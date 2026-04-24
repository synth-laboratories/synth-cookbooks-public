from __future__ import annotations

import argparse
import json
import os

import httpx


def _reward(payload: dict[str, object]) -> object:
    reward_info = payload.get("reward_info")
    if isinstance(reward_info, dict):
        return reward_info.get("outcome_reward")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        return summary.get("outcome_reward")
    return payload.get("reward")


def _policy_config() -> dict[str, object]:
    api_key = str(os.environ.get("OPENAI_API_KEY") or "").strip()
    config: dict[str, object] = {
        "inference_url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4.1-nano",
        "temperature": 0.0,
        "max_tokens": 96,
    }
    if api_key:
        config["api_key"] = api_key
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8922")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-steps", type=int, default=3)
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    with httpx.Client(timeout=120.0) as client:
        health = client.get(f"{base}/health")
        health.raise_for_status()

        task_info = client.get(f"{base}/task_info", params={"seed": args.seed})
        task_info.raise_for_status()

        rollout = client.post(
            f"{base}/rollout",
            json={
                "trace_correlation_id": "minigrid-smoke",
                "env": {
                    "seed": args.seed,
                    "config": {"split_group": "train", "max_steps": args.max_steps},
                },
                "policy": {"config": _policy_config()},
                "submission_mode": "sync",
            },
        )
        rollout.raise_for_status()
        rollout_payload = rollout.json()
        rollout_id = str(rollout_payload["rollout_id"])

        checkpoint = client.post(
            f"{base}/rollouts/{rollout_id}/checkpoints",
            json={"label": "smoke_checkpoint"},
        )
        checkpoint.raise_for_status()
        checkpoint_payload = checkpoint.json()

        resume = client.post(
            f"{base}/rollouts/{rollout_id}/resume",
            json={
                "checkpoint_id": checkpoint_payload["checkpoint_id"],
                "target_rollout_id": "minigrid_smoke_resume",
                "submission_mode": "sync",
                "overrides": {"segment_steps": 1},
            },
        )
        resume.raise_for_status()

    print(
        json.dumps(
            {
                "health_status": health.json().get("status"),
                "task_preview": (task_info.json().get("task_metadata") or {}).get("task_preview"),
                "rollout_reward": _reward(rollout_payload),
                "checkpoint_id": checkpoint_payload.get("checkpoint_id"),
                "resume_reward": _reward(resume.json()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
