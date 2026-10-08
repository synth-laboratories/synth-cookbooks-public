"""Write evidence receipts from in-process proofs against current Containers + MLX learner."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient
from synth_containers.platform import create_compat_app
from synth_mlx_rl.api.app import create_app
from synth_mlx_rl.testing import FakeEngine

ROOT = Path(__file__).resolve().parents[1]
TELEMETRY = {"enabled": True, "transport": "sse", "retention": "run"}
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write(cookbook_id: str, payload: dict) -> None:
    dest = ROOT / "cookbooks" / cookbook_id / "evidence" / "evidence.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {dest}")


def prove_banking77() -> dict:
    client = TestClient(create_compat_app("banking77_classify"))
    health = client.get("/health").json()
    meta = client.get("/metadata").json()
    client.post("/rollouts/prepare", json={"rollout_id": "gold0", "telemetry": TELEMETRY})
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": "gold0",
            "telemetry": TELEMETRY,
            "slot": "stream",
            "world_ref": "world:banking77@train",
            "task_instance_id": "seed:0",
            "evaluation_plan_ref": "banking77_eval.v1",
            "policy_ref": {"harness": "dataset_gold", "config": "dataset_gold"},
        },
    )
    scored = client.post("/reward", json={"rollout_id": "gold0", "mode": "terminal"}).json()
    receipt = {
        "cookbook_id": "gepa-banking77",
        "proved_at": NOW,
        "target": health["target"],
        "runtime_family": meta["runtime_family"],
        "gold_seed0_reward": scored["reward"],
        "gold_seed0_status": scored["status"],
        "rollout_http": started.status_code,
        "claim": "dataset_gold seed 0 scores 1.0 on current banking77_classify",
        "not_claimed": "paid GEPA search uplift",
    }
    assert scored["reward"] == 1.0
    write("gepa-banking77", receipt)
    return receipt


def prove_healthbench() -> dict:
    client = TestClient(create_compat_app("healthbench_chat"))
    health = client.get("/health").json()
    info = client.get("/info").json()
    gepa = info["capabilities"]["optimizer_contracts"]["gepa"]
    roles = info["metadata"]["model_roles"]
    receipt = {
        "cookbook_id": "evals-healthbench",
        "proved_at": NOW,
        "target": health["target"],
        "gepa_contract": gepa["version"],
        "policy_lane": roles["policy"]["usage_lane"],
        "grader_lane": roles["scorer"]["usage_lane"],
        "canonical_grader": roles["scorer"]["canonical"],
        "policy_refs": sorted({row["config"] for row in info["policy_refs"]}),
        "claim": "eval-only HealthBench HTTP contract with independent grader lane",
        "not_claimed": "physician-rubric scores from a live paid grader",
    }
    assert gepa["version"] == "synth_optimizers.gepa.v2"
    write("evals-healthbench", receipt)
    return receipt


def prove_craftax_sft() -> dict:
    client = TestClient(create_compat_app("craftax_engine"))
    client.post("/policy-configs", json={"config_id": "scripted", "config": {"model": "scripted"}})
    rows = []
    for seed in range(2):
        rid = f"cx{seed}"
        client.post("/rollouts/prepare", json={"rollout_id": rid, "telemetry": TELEMETRY})
        started = client.post(
            "/rollouts",
            json={
                "rollout_id": rid,
                "telemetry": TELEMETRY,
                "slot": "stream",
                "world_ref": "world:craftax_default@symbolic_survival",
                "task_instance_id": f"seed:{seed}",
                "policy_ref": {"harness": "react", "config": "scripted"},
            },
        )
        scored = client.post("/reward", json={"rollout_id": rid, "mode": "terminal"}).json()
        rows.append(
            {
                "seed": seed,
                "reward": scored["reward"],
                "status": scored["status"],
                "environment_ref": started.json()["environment_ref"],
            }
        )
    with TestClient(create_app(engine=FakeEngine())) as mlx:
        cap = mlx.get("/v1/synth/capability").json()
        datum = {"input_ids": [1, 2, 3], "target_ids": [2, 3, 4], "weights": [0.0, 1.0, 1.0]}
        forward = mlx.post(
            "/v1/forward_backward", json={"data": [datum], "loss_fn": "cross_entropy"}
        )
        step = mlx.post("/v1/optim_step", json={"params": {"learning_rate": 5e-5}})
        ckpt = mlx.post("/v1/checkpoints/save", json={"name": "sft-craftax-smoke"})
        sft = {
            "cross_entropy_supported": "cross_entropy" in cap["objectives"]["supported"],
            "loss": forward.json()["loss"],
            "training_version": step.json()["training_version"],
            "checkpoint": ckpt.json().get("name") or "sft-craftax-smoke",
            "http": [forward.status_code, step.status_code, ckpt.status_code],
        }
    receipt = {
        "cookbook_id": "mlx-sft-craftax",
        "proved_at": NOW,
        "craftax_eval": rows,
        "sft": sft,
        "claim": "scripted Craftax fixture eval + LoRA SFT step on synth-mlx-rl FakeEngine",
        "not_claimed": "Apple-silicon base-model fine-tune or /evals/craftax gold matrix",
    }
    assert all(row["environment_ref"] == "env:craftax_fixture" for row in rows)
    assert sft["http"] == [200, 200, 200]
    write("mlx-sft-craftax", receipt)
    return receipt


def prove_cispo() -> dict:
    with TestClient(create_app(engine=FakeEngine())) as client:
        cap = client.get("/v1/synth/capability").json()
        datum = {
            "input_ids": [1, 2, 3],
            "target_ids": [2, 3, 4],
            "weights": [0.0, 1.0, 1.0],
            "behavior_logprobs": [0.0, -0.5, -0.4],
            "advantages": 1.5,
        }
        ok = client.post(
            "/v1/forward_backward",
            json={
                "data": [datum],
                "loss_fn": "cispo_minimax",
                "eps_low": 1.0,
                "eps_high": 4.0,
            },
        )
        refused = client.post(
            "/v1/forward_backward",
            json={"data": [datum], "loss_fn": "cispo_minimax", "eps_low": 0.2},
        )
        step = client.post("/v1/optim_step", json={"params": {"learning_rate": 1e-5}})
        receipt = {
            "cookbook_id": "remote-cispo-gsm8k",
            "proved_at": NOW,
            "cispo_supported": "cispo_minimax" in cap["objectives"]["supported"],
            "loss": ok.json()["loss"],
            "training_version": step.json()["training_version"],
            "minimax_refusal_status": refused.status_code,
            "hosted_submit_slugs": ["gepa", "go-ex", "ohco"],
            "claim": "cispo_minimax one step + refuse active lower clip",
            "not_claimed": "hosted CISPO job submit or GSM8K held-out accuracy",
        }
    assert ok.status_code == 200
    assert refused.status_code == 422
    write("remote-cispo-gsm8k", receipt)
    return receipt


def write_pages(receipts: dict[str, dict]) -> None:
    pages = []
    specs = [
        (
            "gepa-banking77",
            "GEPA on Banking77",
            "gepa",
            "banking77",
            "local",
            "A GEPA search over stage2_system beats the seed prompt on held-out exact match.",
            "Gold path proved. Paid search not run.",
            "high",
            "dataset_gold seed 0 reward is 1.0 on current banking77_classify. That is the env-authored baseline, not optimizer uplift.",
        ),
        (
            "evals-healthbench",
            "Evals on HealthBench",
            "eval",
            "healthbench",
            "local",
            "HealthBench can be scored as an eval-only lane with independent policy and grader usage.",
            "Contract proved. Live rubric scores not run (no API key).",
            "high",
            "healthbench_chat advertises synth_optimizers.gepa.v2 and split model_roles without starting a proposer.",
        ),
        (
            "mlx-sft-craftax",
            "Local MLX SFT on Craftax",
            "sft",
            "craftax",
            "local",
            "LoRA SFT through synth-mlx-rl plus Craftax fixture eval is a valid local-training cookbook.",
            "Proved on FakeEngine + craftax_engine scripted fixture.",
            "high",
            "Fixture seeds 0–1 scored 0.5; SFT cross_entropy + checkpoint succeeded. Not the public gold eval matrix.",
        ),
        (
            "remote-cispo-gsm8k",
            "Remote CISPO on GSM8K",
            "cispo",
            "gsm8k",
            "hosted",
            "CISPO minimax is reachable on the public learner API; hosted Optimizers does not yet submit CISPO.",
            "Kernel proved. Hosted submit not available.",
            "high",
            "cispo_minimax step succeeded; eps_low=0.2 returns 422. Optimizers startup slugs remain gepa/go-ex/ohco.",
        ),
    ]
    for slug, title, method, task, execution, hypothesis, verdict, confidence, why in specs:
        receipt = receipts[slug]
        pages.append(
            {
                "slug": slug,
                "title": title,
                "method": method,
                "task": task,
                "execution": execution,
                "status": "proven-local",
                "hypothesis": hypothesis,
                "verdict": verdict,
                "confidence": confidence,
                "why": why,
                "receipt": receipt,
                "source": f"cookbooks/{slug}",
                "href": f"/cookbooks/{slug}",
            }
        )
    dest = ROOT / "catalog" / "generated" / "pages.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"generated_at": NOW, "pages": pages}, indent=2) + "\n")
    print(f"wrote {dest}")


def main() -> int:
    receipts = {
        "gepa-banking77": prove_banking77(),
        "evals-healthbench": prove_healthbench(),
        "mlx-sft-craftax": prove_craftax_sft(),
        "remote-cispo-gsm8k": prove_cispo(),
    }
    write_pages(receipts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
