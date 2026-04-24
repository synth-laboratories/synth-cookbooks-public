from __future__ import annotations

import argparse
import json
import os

import httpx


def _observation(state_payload: dict[str, object]) -> dict[str, object]:
    env_state = state_payload.get("env_state")
    if isinstance(env_state, dict):
        values = env_state.get("values")
        if isinstance(values, dict):
            return values
    observation = state_payload.get("observation")
    return dict(observation) if isinstance(observation, dict) else {}


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
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    with httpx.Client(timeout=120.0) as client:
        rollout = client.post(
            f"{base}/rollout",
            json={
                "trace_correlation_id": "minigrid-snapshot-proof",
                "env": {"seed": args.seed, "config": {"split_group": "train", "max_steps": 1}},
                "policy": {"config": _policy_config()},
                "submission_mode": "sync",
            },
        )
        rollout.raise_for_status()
        rollout_payload = rollout.json()
        rollout_id = str(rollout_payload["rollout_id"])
        original_state = client.get(f"{base}/rollouts/{rollout_id}/state")
        original_state.raise_for_status()
        original_observation = _observation(original_state.json())

        checkpoint = client.post(
            f"{base}/rollouts/{rollout_id}/checkpoints",
            json={"label": "snapshot_proof"},
        )
        checkpoint.raise_for_status()
        checkpoint_id = checkpoint.json()["checkpoint_id"]

        resumed = client.post(
            f"{base}/rollouts/{rollout_id}/resume",
            json={
                "checkpoint_id": checkpoint_id,
                "target_rollout_id": "minigrid_snapshot_resume",
                "submission_mode": "sync",
                "overrides": {"segment_steps": 1},
            },
        )
        resumed.raise_for_status()
        resumed_state = client.get(f"{base}/rollouts/minigrid_snapshot_resume/state")
        resumed_state.raise_for_status()
        restored_observation = _observation(resumed_state.json())

    proof = {
        "checkpoint_id": checkpoint_id,
        "mission_match": original_observation.get("mission") == restored_observation.get("mission"),
        "grid_match": original_observation.get("grid") == restored_observation.get("grid"),
        "agent_pos_match": original_observation.get("agent_pos")
        == restored_observation.get("agent_pos"),
        "carrying_match": original_observation.get("carrying")
        == restored_observation.get("carrying"),
        "admissible_actions_match": original_observation.get("admissible_actions")
        == restored_observation.get("admissible_actions"),
    }
    print(json.dumps(proof, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
