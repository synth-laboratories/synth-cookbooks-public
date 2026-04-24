from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import modal
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler

from common_training import (
    DEFAULT_MODEL_ID,
    compute_completion_logprobs,
    example_to_messages,
    expand_paths,
    is_main_process,
    load_text_only_causal_lm,
    load_tokenizer,
    maybe_write_json,
    normalize_text,
    seed_everything,
)
from modal_common import CKPT_DIR, GPU_8, REMOTE_ROOT, training_image, volume_mounts

app = modal.App("nanolong-qwen35-rl-cispo")
image = training_image()



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CISPO-style RL fine-tuning for Qwen3.5-0.8B on Modal")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--train-files", nargs="+", required=True)
    parser.add_argument("--output-dir", default=f"{CKPT_DIR}/rl")
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--prompts-per-device-batch-size", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--offpolicy-updates-per-rollout", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--reward-backend", choices=["exact_match", "synth_task_app"], default="synth_task_app")
    parser.add_argument("--task-app-url", default=None)
    parser.add_argument("--environment-api-key", default=None)
    parser.add_argument("--length-penalty", type=float, default=0.0, help="Subtract alpha * response_tokens from reward")
    parser.add_argument("--is-weight-clip-high", type=float, default=2.0)
    parser.add_argument("--is-weight-clip-low", type=float, default=None, help="Leave unset to match the M1 paper's upper-bound-only tuning")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--nproc-per-node", type=int, default=None)
    parser.add_argument("--no-torchrun", action="store_true")
    return parser.parse_args()



def launch_with_torchrun(args: argparse.Namespace) -> None:
    cmd = [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={args.nproc_per_node or torch.cuda.device_count() or 1}",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
        "--no-torchrun",
    ]
    print("Launching distributed job:", " ".join(shlex.quote(x) for x in cmd), flush=True)
    subprocess.check_call(cmd)



def collate_examples(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch



def render_generation_prompt(tokenizer, example: dict[str, Any]) -> str:
    messages = example_to_messages(example, for_generation=True)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)



def decode_generated_completions(tokenizer, sequences: torch.Tensor, prompt_lengths: torch.Tensor) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    token_counts: list[int] = []
    for seq, prompt_len in zip(sequences, prompt_lengths):
        prompt_len_i = int(prompt_len.item())
        completion_ids = seq[prompt_len_i:]
        # Drop padding tokens that may appear at the tail.
        completion_ids = completion_ids[completion_ids != tokenizer.pad_token_id]
        texts.append(tokenizer.decode(completion_ids, skip_special_tokens=True))
        token_counts.append(int(completion_ids.numel()))
    return texts, token_counts



def exact_match_reward(example: dict[str, Any], completion: str) -> float:
    target = example.get("answer") or example.get("expected_output") or example.get("target")
    if target is None:
        raise ValueError(
            "exact_match reward backend requires one of: answer, expected_output, or target in each training row"
        )
    completion_n = normalize_text(completion)
    target_n = normalize_text(str(target))
    if completion_n == target_n:
        return 1.0
    if target_n and target_n in completion_n:
        return 1.0
    return 0.0



def call_synth_task_app(
    *,
    task_app_url: str,
    example: dict[str, Any],
    completion: str,
    environment_api_key: str | None,
) -> tuple[float, dict[str, Any]]:
    if not task_app_url:
        raise ValueError("reward_backend=synth_task_app requires --task-app-url")

    env_payload = {
        "seed": int(example.get("seed", 0)),
        "task_id": example.get("task_id"),
        "container_url": example.get("container_url"),
        "max_steps": int(example.get("max_steps", 8)),
        "timeout_s": int(example.get("timeout_s", 120)),
        "mode": example.get("mode", "candidate_output"),
    }
    payload = {
        "env": env_payload,
        "inputs": example.get("inputs", {}),
        "policy": {
            "config": {
                "prompt_template": example.get("prompt_template", "{{prompt}}"),
                "candidate_output": completion,
                "inference_url": example.get("inference_url"),
                "model": example.get("rollout_model"),
                "temperature": float(example.get("temperature", 0.0)),
                "max_tokens": int(example.get("max_tokens", 1024)),
            }
        },
    }

    headers = {"Content-Type": "application/json"}
    if environment_api_key:
        headers["x-api-key"] = environment_api_key

    with httpx.Client(timeout=180.0) as client:
        response = client.post(f"{task_app_url.rstrip('/')}/rollout", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
    metrics = data.get("metrics", {})
    reward = metrics.get("mean_return", metrics.get("reward_mean"))
    if reward is None:
        raise ValueError(f"Task app response did not contain mean_return/reward_mean: {data}")
    return float(reward), data



def score_batch(
    *,
    examples: list[dict[str, Any]],
    completions: list[str],
    token_counts: list[int],
    reward_backend: str,
    task_app_url: str | None,
    environment_api_key: str | None,
    length_penalty: float,
) -> tuple[list[float], list[dict[str, Any]]]:
    rewards: list[float] = []
    payloads: list[dict[str, Any]] = []
    for example, completion, token_count in zip(examples, completions, token_counts):
        if reward_backend == "exact_match":
            reward = exact_match_reward(example, completion)
            payload = {"backend": "exact_match"}
        else:
            reward, payload = call_synth_task_app(
                task_app_url=task_app_url or "",
                example=example,
                completion=completion,
                environment_api_key=environment_api_key,
            )
        if length_penalty:
            reward -= length_penalty * float(token_count)
        rewards.append(reward)
        payloads.append(payload)
    return rewards, payloads



def build_advantages(reward_tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Group-relative normalized rewards, GRPO-style.
    mean = reward_tensor.mean(dim=-1, keepdim=True)
    std = reward_tensor.std(dim=-1, keepdim=True, correction=0)
    advantages = (reward_tensor - mean) / (std + eps)
    return advantages



def save_checkpoint(accelerator: Accelerator, model, tokenizer, output_dir: Path, global_step: int) -> None:
    if not accelerator.is_main_process:
        return
    ckpt_dir = output_dir / f"step_{global_step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(ckpt_dir, safe_serialization=True)
    tokenizer.save_pretrained(ckpt_dir)
    maybe_write_json(output_dir / "latest_checkpoint.json", {"global_step": global_step, "path": str(ckpt_dir)})
    latest = output_dir / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(ckpt_dir.name)
    except OSError:
        # Symlinks are nice-to-have only.
        pass



def train_worker(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(log_with=None if args.report_to == "none" else args.report_to)
    tokenizer = load_tokenizer(args.model_id)
    model = load_text_only_causal_lm(
        args.model_id,
        dtype=args.dtype,
        gradient_checkpointing=args.gradient_checkpointing,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        attn_implementation=args.attn_implementation,
    )
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)

    data_files = expand_paths(args.train_files)
    train_dataset = load_dataset("json", data_files=data_files, split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.prompts_per_device_batch_size,
        shuffle=True,
        collate_fn=collate_examples,
        drop_last=False,
    )

    steps_per_epoch = math.ceil(len(train_loader) * args.offpolicy_updates_per_rollout)
    total_train_steps = steps_per_epoch * args.num_train_epochs
    if args.max_steps > 0:
        total_train_steps = min(total_train_steps, args.max_steps)
    warmup_steps = int(total_train_steps * args.warmup_ratio)
    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_train_steps, 1),
    )

    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
    global_step = 0
    rollout_batches_seen = 0

    if accelerator.is_main_process and args.report_to != "none":
        accelerator.init_trackers(
            project_name="nanolong-qwen35-rl",
            config=vars(args),
        )

    for epoch in range(args.num_train_epochs):
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

        for batch in train_loader:
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
            rollout_batches_seen += 1
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.eval()

            prompts = [render_generation_prompt(tokenizer, example) for example in batch]
            with torch.no_grad():
                tokenized = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_prompt_length,
                )
                tokenized = {k: v.to(accelerator.device) for k, v in tokenized.items()}
                prompt_lengths = tokenized["attention_mask"].sum(dim=-1)
                generated = unwrapped.generate(
                    **tokenized,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_new_tokens=args.max_new_tokens,
                    num_return_sequences=args.group_size,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
                prompt_lengths = prompt_lengths.repeat_interleave(args.group_size)
                old_logps, token_mask = compute_completion_logprobs(
                    unwrapped,
                    generated,
                    prompt_lengths,
                    tokenizer.pad_token_id,
                )
                old_logps = old_logps.detach()
                token_mask = token_mask.detach()
                completions, token_counts = decode_generated_completions(tokenizer, generated, prompt_lengths)

            expanded_examples: list[dict[str, Any]] = []
            for example in batch:
                expanded_examples.extend([example] * args.group_size)

            rewards_list, reward_payloads = score_batch(
                examples=expanded_examples,
                completions=completions,
                token_counts=token_counts,
                reward_backend=args.reward_backend,
                task_app_url=args.task_app_url,
                environment_api_key=args.environment_api_key,
                length_penalty=args.length_penalty,
            )
            rewards = torch.tensor(rewards_list, dtype=torch.float32, device=accelerator.device).view(
                len(batch), args.group_size
            )
            advantages = build_advantages(rewards).reshape(-1)

            model.train()
            for _ in range(args.offpolicy_updates_per_rollout):
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break
                curr_logps, curr_mask = compute_completion_logprobs(model, generated, prompt_lengths, tokenizer.pad_token_id)
                curr_logps = curr_logps * curr_mask

                seq_log_ratio = ((curr_logps - old_logps) * token_mask).sum(dim=-1)
                is_weight = torch.exp(torch.clamp(seq_log_ratio, min=-20.0, max=20.0)).detach()
                if args.is_weight_clip_low is None:
                    clipped_weight = torch.clamp(is_weight, max=args.is_weight_clip_high)
                else:
                    clipped_weight = torch.clamp(is_weight, min=args.is_weight_clip_low, max=args.is_weight_clip_high)

                token_mean_logp = (curr_logps.sum(dim=-1) / token_mask.sum(dim=-1).clamp_min(1.0))
                loss = -(clipped_weight * advantages * token_mean_logp).mean()

                optimizer.zero_grad(set_to_none=True)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                global_step += 1

                if accelerator.is_main_process and global_step % args.logging_steps == 0:
                    metrics = {
                        "loss": float(loss.detach().cpu().item()),
                        "reward_mean": float(rewards.mean().detach().cpu().item()),
                        "reward_max": float(rewards.max().detach().cpu().item()),
                        "adv_mean": float(advantages.mean().detach().cpu().item()),
                        "is_weight_mean": float(is_weight.mean().detach().cpu().item()),
                        "rollout_batches_seen": rollout_batches_seen,
                        "global_step": global_step,
                    }
                    print(json.dumps(metrics), flush=True)
                    if args.report_to != "none":
                        accelerator.log(metrics, step=global_step)

                if global_step % args.save_steps == 0:
                    accelerator.wait_for_everyone()
                    save_checkpoint(accelerator, model, tokenizer, output_dir, global_step)

    accelerator.wait_for_everyone()
    save_checkpoint(accelerator, model, tokenizer, output_dir, global_step)

    if accelerator.is_main_process:
        maybe_write_json(
            output_dir / "recipe.rl.cispo.json",
            {
                "stage": "rl",
                "algorithm": "cispo_style",
                "model_id": args.model_id,
                "train_files": args.train_files,
                "group_size": args.group_size,
                "offpolicy_updates_per_rollout": args.offpolicy_updates_per_rollout,
                "reward_backend": args.reward_backend,
                "task_app_url": args.task_app_url,
                "notes": [
                    "This is a practical CISPO-style implementation: it clips sequence importance weights, keeps all completion tokens in the policy-gradient term, and uses group-relative normalized rewards.",
                    "For true multi-turn long-horizon credit assignment, extend the task app to return per-step or concatenated training transcripts plus old-policy token logprobs.",
                ],
                "example_reward_payloads": reward_payloads[: min(2, len(reward_payloads))],
            },
        )

    if args.report_to != "none":
        accelerator.end_training()


@app.function(
    image=image,
    gpu=GPU_8,
    timeout=60 * 60 * 24,
    volumes=volume_mounts(),
)
def run(cmd: str) -> None:
    os.chdir(REMOTE_ROOT)
    argv = shlex.split(cmd)
    full_cmd = [sys.executable, str(Path(REMOTE_ROOT) / "train_rl_cispo_modal.py"), *argv]
    print("Remote entrypoint:", " ".join(shlex.quote(x) for x in full_cmd), flush=True)
    subprocess.check_call(full_cmd)


@app.local_entrypoint()
def main(cmd: str = "") -> None:
    run.remote(cmd)


if __name__ == "__main__":
    args = parse_args()
    if not args.no_torchrun and "LOCAL_RANK" not in os.environ and (torch.cuda.device_count() or 0) > 1:
        launch_with_torchrun(args)
    else:
        train_worker(args)
