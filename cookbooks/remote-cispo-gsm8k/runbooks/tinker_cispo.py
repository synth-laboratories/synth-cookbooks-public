"""Hosted CISPO on GSM8K through Tinker (Thinking Machines), not local MLX.

Rollouts and the LoRA learner both go through `tinker.ServiceClient`. The
objective is Tinker's `cispo` loss (`clip_low_threshold=0`, `clip_high_threshold=4`),
the same packing used by the Craftax Tinker gold runner.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
COOKBOOK = HERE.parent
CONTAINERS = Path.home() / "Documents/GitHub/containers-mlx-local-rl-20260818/src"

SOLVE_SYSTEM = (
    "Solve the grade-school math word problem. Reason step by step, then state the "
    "final numeric answer on its own last line in the form '#### <answer>'. The final "
    "answer must be a bare number with no units, no currency symbol and no words."
)
PREFERRED_MODELS = (
    "Qwen/Qwen3.5-4B",
    "meta-llama/Llama-3.2-3B",
    "Qwen/Qwen3-8B",
    "thinkingmachines/Inkling-Small",
)


def load_tinker_key() -> None:
    if os.environ.get("TINKER_API_KEY", "").strip():
        return
    env_path = Path.home() / "Documents/GitHub/synth-ai/.env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("TINKER_API_KEY="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                if value:
                    os.environ["TINKER_API_KEY"] = value
                break
    if not os.environ.get("TINKER_API_KEY", "").strip():
        raise RuntimeError("TINKER_API_KEY is required")


def configure_tinker_tls() -> None:
    try:
        import pyqwest
        import tinker._base_client as base_client
        from pyqwest.httpx import AsyncPyqwestTransport
    except Exception:
        return
    base_client._default_pyqwest_transport = lambda: AsyncPyqwestTransport(
        transport=pyqwest.HTTPTransport(tls_include_system_certs=True)
    )


def parse_answer(text: str) -> str | None:
    if CONTAINERS.is_dir():
        sys.path.insert(0, str(CONTAINERS))
        from synth_containers.platform.gsm8k_world import parse_answer as world_parse

        parsed = world_parse(text)
        return parsed.value
    match = re.search(r"####\s*([^\n]+)", text)
    if not match:
        boxed = re.search(r"\\boxed\{([^}]+)\}", text)
        if boxed:
            raw = boxed.group(1)
        else:
            numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
            return numbers[-1] if numbers else None
    else:
        raw = match.group(1)
    cleaned = re.sub(r"[,$]", "", raw.strip())
    cleaned = re.sub(r"<\|[^|]+\|>", "", cleaned)
    cleaned = cleaned.rstrip(".")
    return cleaned or None


def reward(text: str, gold: str) -> float:
    parsed = parse_answer(text)
    return 1.0 if parsed is not None and parsed == gold else 0.0


def group_normalize(rewards: list[float]) -> list[float]:
    if len(rewards) < 2:
        return [0.0 for _ in rewards]
    mean = statistics.fmean(rewards)
    if len(rewards) == 1:
        return [0.0]
    var = statistics.pvariance(rewards)
    scale = var**0.5
    if scale < 1e-8:
        return [0.0 for _ in rewards]
    return [(value - mean) / scale for value in rewards]


def load_rows(*, eval_n: int, train_n: int) -> tuple[list[dict], list[dict]]:
    if CONTAINERS.is_dir():
        sys.path.insert(0, str(CONTAINERS))
        from synth_containers.platform.gsm8k_world import (  # type: ignore
            HELDOUT_SPLIT,
            TRAIN_SPLIT,
            load_row,
            source_name,
            split_size,
        )

        source = source_name()
        train_size = split_size(TRAIN_SPLIT)
        held_size = split_size(HELDOUT_SPLIT)
        if train_n > train_size or eval_n > held_size:
            raise RuntimeError(
                f"requested train={train_n}/{train_size} eval={eval_n}/{held_size} source={source}"
            )
        print(f"GSM8K source={source} train={train_size} heldout={held_size}", flush=True)
        train = []
        for i in range(train_n):
            row = load_row(TRAIN_SPLIT, i)
            if row is None:
                raise RuntimeError(f"missing train seed {i}")
            train.append({"seed": i, "question": row.question, "answer": row.answer})
        heldout = []
        for i in range(eval_n):
            row = load_row(HELDOUT_SPLIT, i)
            if row is None:
                raise RuntimeError(f"missing heldout seed {i}")
            heldout.append({"seed": i, "question": row.question, "answer": row.answer})
        return train, heldout
    fixture = json.loads((COOKBOOK / "fixtures" / "gsm8k_tiny.json").read_text())["rows"]
    return fixture[:train_n], fixture[:eval_n]


def messages(question: str) -> list[dict[str, str]]:
    prompt = question
    if CONTAINERS.is_dir():
        sys.path.insert(0, str(CONTAINERS))
        from synth_containers.platform.gsm8k_world import user_prompt

        prompt = user_prompt(question)
    return [
        {"role": "system", "content": SOLVE_SYSTEM},
        {"role": "user", "content": prompt},
    ]


def prompt_ids(tokenizer, question: str) -> list[int]:
    kwargs = {"tokenize": True, "add_generation_prompt": True, "enable_thinking": False}
    try:
        ids = tokenizer.apply_chat_template(messages(question), **kwargs)
    except TypeError:
        ids = tokenizer.apply_chat_template(
            messages(question), tokenize=True, add_generation_prompt=True
        )
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    elif hasattr(ids, "input_ids"):
        ids = ids.input_ids
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(x) for x in ids]


def pick_model(supported: set[str], requested: str | None) -> str:
    if requested:
        if requested not in supported:
            raise RuntimeError(f"Tinker does not advertise {requested}")
        return requested
    for name in PREFERRED_MODELS:
        if name in supported:
            return name
    raise RuntimeError(
        "no small Qwen model advertised by Tinker; pass --model with an advertised id"
    )


async def sample_one(tinker, sampler, tokenizer, question: str, *, temperature: float, max_tokens: int, seed: int):
    ids = prompt_ids(tokenizer, question)
    prompt = tinker.ModelInput.from_ints(tokens=ids)
    response = await sampler.sample_async(
        prompt,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed % (2**31),
        ),
    )
    sequence = response.sequences[0]
    completion_ids = [int(v) for v in sequence.tokens]
    logprobs = [float(v) for v in (sequence.logprobs or [])]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return {
        "prompt_ids": ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "text": text,
    }


def to_datum(tinker, sample: dict, advantage: float):
    ids = list(sample["prompt_ids"]) + list(sample["completion_ids"])
    n = len(sample["completion_ids"])
    if n == 0 or len(sample["logprobs"]) != n:
        raise RuntimeError("Tinker sample omitted aligned completion logprobs")
    targets = ids[1:]
    pad = len(ids) - 1 - n
    logprobs = [0.0] * pad + list(sample["logprobs"])
    advantages = [0.0] * pad + [float(advantage)] * n
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens=ids[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(data=targets, dtype="int64", shape=[len(targets)]),
            "logprobs": tinker.TensorData(data=logprobs, dtype="float32", shape=[len(logprobs)]),
            "advantages": tinker.TensorData(data=advantages, dtype="float32", shape=[len(advantages)]),
        },
    )


async def evaluate(tinker, sampler, tokenizer, rows, *, max_tokens: int, label: str) -> float:
    hits = 0
    t0 = time.time()
    for i, row in enumerate(rows):
        out = await sample_one(
            tinker, sampler, tokenizer, row["question"],
            temperature=0.0, max_tokens=max_tokens, seed=10_000 + int(row["seed"]),
        )
        if i == 0:
            preview = out["text"].replace("\n", " ")[:240]
            print(f"  [{label} sample0] tokens={len(out['completion_ids'])} parsed={parse_answer(out['text'])!r} gold={row['answer']!r} text={preview!r}", flush=True)
        hits += reward(out["text"], row["answer"])
    acc = hits / max(len(rows), 1)
    print(f"  [{label}] held-out {hits}/{len(rows)} = {acc:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    return acc


async def run(args: argparse.Namespace) -> dict:
    if not args.authorize:
        raise SystemExit("refusing paid Tinker CISPO: pass --authorize")
    load_tinker_key()
    configure_tinker_tls()
    import tinker

    train_rows, eval_rows = load_rows(eval_n=args.eval_n, train_n=args.train_prompts)
    service = tinker.ServiceClient(user_metadata={"task": "cookbook_remote_cispo_gsm8k"})
    capabilities = await service.get_server_capabilities_async()
    supported = {row.model_name for row in capabilities.supported_models}
    model = pick_model(supported, args.model)
    print(f"Tinker CISPO  model={model}  lora r={args.rank}", flush=True)

    training = await service.create_lora_training_client_async(
        base_model=model, rank=args.rank, seed=args.seed,
    )
    tokenizer = training.get_tokenizer()
    sampler = await service.create_sampling_client_async(base_model=model)
    baseline = await evaluate(
        tinker, sampler, tokenizer, eval_rows, max_tokens=args.max_tokens, label="baseline",
    )

    adam = tinker.AdamParams(
        learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.01,
    )
    history = []
    step = 0
    for round_index in range(args.rounds):
        start = (round_index * 8) % max(len(train_rows), 1)
        prompts = train_rows[start:start + 8] or train_rows[:8]
        datums, rewards_seen = [], []
        for row in prompts:
            group = []
            rewards = []
            for replica in range(args.group):
                sample = await sample_one(
                    tinker, sampler, tokenizer, row["question"],
                    temperature=1.0, max_tokens=args.max_tokens,
                    seed=args.seed + round_index * 1000 + int(row["seed"]) * 17 + replica,
                )
                group.append(sample)
                rewards.append(reward(sample["text"], row["answer"]))
            advantages = group_normalize(rewards)
            rewards_seen.extend(rewards)
            for sample, adv in zip(group, advantages):
                if abs(adv) < 1e-12:
                    continue
                datums.append(to_datum(tinker, sample, adv))
        mean_reward = statistics.fmean(rewards_seen) if rewards_seen else 0.0
        if datums:
            for offset in range(0, len(datums), args.accum):
                batch = datums[offset:offset + args.accum]
                forward = await training.forward_backward_async(
                    batch,
                    loss_fn="cispo",
                    loss_fn_config={
                        "clip_low_threshold": 0.0,
                        "clip_high_threshold": args.eps_high,
                    },
                )
                await forward.result_async()
                optimizer = await training.optim_step_async(adam)
                await optimizer.result_async()
                step += 1
            sampler_result = await (
                await training.save_weights_for_sampler_async(
                    name=f"gsm8k_cispo_r{round_index:03d}",
                    ttl_seconds=7 * 86400,
                )
            ).result_async()
            path = getattr(sampler_result, "path", None) or getattr(sampler_result, "model_path", None)
            sampler = await service.create_sampling_client_async(model_path=str(path))
        print(
            f"  round {round_index}: train_reward={mean_reward:.3f} datums={len(datums)} steps={step}",
            flush=True,
        )
        history.append({"round": round_index, "train_reward": mean_reward, "datums": len(datums), "steps": step})

    final = await evaluate(
        tinker, sampler, tokenizer, eval_rows, max_tokens=args.max_tokens, label="after cispo",
    )
    receipt = {
        "schema": "synth.cookbook.tinker_cispo.v1",
        "backend": "tinker",
        "loss_fn": "cispo",
        "model": model,
        "baseline": baseline,
        "final": final,
        "uplift": final - baseline,
        "eval_n": args.eval_n,
        "dataset": "openai/gsm8k",
        "split": "test",
        "source": os.environ.get("SYNTH_GSM8K_SOURCE", "fixture"),
        "history": history,
        "args": {k: getattr(args, k) for k in ("eval_n", "train_prompts", "group", "rounds", "lr", "max_tokens", "rank", "eps_high")},
    }
    print("=" * 58)
    print(f"  baseline   {baseline:.4f}")
    print(f"  final      {final:.4f}")
    print(f"  uplift     {final - baseline:+.4f}")
    print("=" * 58)
    if args.out:
        Path(args.out).write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authorize", action="store_true")
    ap.add_argument("--model", default="")
    ap.add_argument("--eval-n", type=int, default=16)
    ap.add_argument("--train-prompts", type=int, default=32)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eps-high", type=float, default=4.0)
    ap.add_argument("--out", default=str(COOKBOOK / "evidence" / "tinker_cispo.json"))
    args = ap.parse_args()
    args.model = args.model or None
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
