from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

CONTAINER_ROOT = Path(__file__).resolve().parent
if str(CONTAINER_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTAINER_ROOT))

from pool_runner import ensure_pool, load_secret, wait_for_rollout
from smoke import DEFAULT_RUNNER_PATH, _build_open_thoughts_request
try:
    from synth_ai.sdk.container_pools import ContainerPoolsClient
except ModuleNotFoundError:
    from synth_ai.sdk.pools import ContainerPoolsClient


DEFAULT_BACKEND_BASE = "http://127.0.0.1:8001"
DEFAULT_OPEN_THOUGHTS_TASK = "application-debug"
DEFAULT_MODEL_NAME = "openai/gpt-5.4-nano"
DEFAULT_REASONING_EFFORT = "low"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one Rhodes Harbor TBLite rollout.")
    parser.add_argument("--backend-base", default=DEFAULT_BACKEND_BASE)
    parser.add_argument("--open-thoughts-task", default=DEFAULT_OPEN_THOUGHTS_TASK)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--pool-suffix", default=str(int(time.time())))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)

    api_key = load_secret("SYNTH_API_KEY", fallback_files=[])
    client = ContainerPoolsClient(
        api_key=api_key,
        backend_base=args.backend_base,
        timeout=max(args.timeout_s, 900.0),
    )
    request = _build_open_thoughts_request(
        task_id=args.open_thoughts_task,
        pool_id_suffix=args.pool_suffix,
        model_name=str(args.model),
        reasoning_effort=str(args.reasoning_effort),
    )
    pool = ensure_pool(client, request)
    pool_id = str(request["pool_id"])
    task = dict((request.get("tasks") or [])[0])
    task_id = str(task["task_id"])
    rollout_request = {
        "task_id": task_id,
        "trace_correlation_id": f"{pool_id}-seed-{args.seed}",
        "policy": {"model": str(args.model)},
        "messages": [{"role": "user", "content": "Solve the Terminal-Bench Lite task."}],
        "env": {"seed": args.seed},
        "metadata": {
            "open_thoughts_task": args.open_thoughts_task,
            "runner_path": str(DEFAULT_RUNNER_PATH),
            "runner_mode": "one_shot",
        },
    }
    created = client.create_rollout(pool_id, rollout_request)
    rollout = wait_for_rollout(
        client,
        pool_id=pool_id,
        rollout_id=str(created["rollout_id"]),
        timeout_s=args.timeout_s,
    )

    output_root = args.output_root.resolve()
    artifacts_root = output_root / "artifacts"
    report_root = output_root / "reports"
    _write_json(artifacts_root / "pool_request.json", request)
    _write_json(artifacts_root / "rollout_request.json", rollout_request)
    _write_json(artifacts_root / "pool_response.json", pool if isinstance(pool, dict) else {})
    _write_json(
        artifacts_root / "rollout_summary.json",
        {
            "backend_base": args.backend_base,
            "pool_id": pool_id,
            "task_id": task_id,
            "rollout_id": rollout.get("rollout_id"),
            "requested_model": args.model,
            "effective_model": ((rollout.get("result") or {}).get("metadata") or {}).get(
                "effective_model_name"
            ),
            "status": rollout.get("status"),
            "success": rollout.get("success"),
            "score": rollout.get("score"),
            "error": rollout.get("error"),
            "metadata": rollout.get("metadata"),
        },
    )
    report_root.mkdir(parents=True, exist_ok=True)
    report_root.joinpath("summary.md").write_text(
        "\n".join(
            [
                "# Harbor TBLite One Shot",
                "",
                f"- backend_base: `{args.backend_base}`",
                f"- open_thoughts_task: `{args.open_thoughts_task}`",
                f"- pool_id: `{pool_id}`",
                f"- rollout_id: `{rollout.get('rollout_id')}`",
                f"- requested_model: `{args.model}`",
                f"- effective_model: `{((rollout.get('result') or {}).get('metadata') or {}).get('effective_model_name')}`",
                f"- success: `{bool(rollout.get('success'))}`",
                f"- score: `{rollout.get('score')}`",
                f"- status: `{rollout.get('status')}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0 if rollout.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
