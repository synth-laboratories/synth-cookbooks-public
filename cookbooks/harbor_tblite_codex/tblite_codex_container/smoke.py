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
from tb_lite_dataset import package_open_thoughts_task
try:
    from synth_ai.sdk.container_pools import ContainerPoolsClient
except ModuleNotFoundError:
    from synth_ai.sdk.pools import ContainerPoolsClient


DEFAULT_BACKEND_BASE = "http://127.0.0.1:8001"
DEFAULT_EXAMPLE_PATH = CONTAINER_ROOT / "tb_lite_harbor_pool.json"
DEFAULT_OPEN_THOUGHTS_TASK = "application-debug"
DEFAULT_RUNNER_PATH = CONTAINER_ROOT / "codex_harbor_runner.py"
DEFAULT_MODEL_NAME = "openai/gpt-5.4-mini"
DEFAULT_REASONING_EFFORT = "medium"


def _load_request(path: Path, *, pool_id_suffix: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["pool_id"] = f"{payload['pool_id']}_{pool_id_suffix}"
    payload["name"] = f"{payload.get('name', 'TBLite Harbor')} {pool_id_suffix}"
    payload["assembly_id"] = str(payload.get("assembly_id") or payload["pool_id"])
    return payload


def _build_open_thoughts_request(
    *,
    task_id: str,
    pool_id_suffix: str,
    model_name: str,
    reasoning_effort: str,
) -> dict[str, Any]:
    packaged = package_open_thoughts_task(task_id, runner_path=DEFAULT_RUNNER_PATH)
    return {
        "pool_id": f"pool_tb_lite_harbor_{packaged.task_id.replace('-', '_')}_{pool_id_suffix}",
        "assembly_id": f"pool_tb_lite_harbor_{packaged.task_id.replace('-', '_')}_{pool_id_suffix}",
        "name": f"TBLite Harbor {packaged.task_id} {pool_id_suffix}",
        "backend": "harbor",
        "metadata": {
            "suite": "tb_lite",
            "benchmark_name": "terminal_bench_lite",
            "default_task_id": packaged.task_id,
            "source_dataset": packaged.metadata["source_dataset"],
            "source_task_id": packaged.metadata["source_task_id"],
        },
        "tasks": [
            {
                "task_id": packaged.task_id,
                "backend": "harbor",
                "dockerfile": packaged.dockerfile,
                "context_tar_base64": packaged.context_tar_base64,
                "entrypoint": (
                    "python /app/run_codex_harbor_rollout.py --input /tmp/rollout.json "
                    "--output /tmp/result.json --task-root /app/task"
                ),
                "metadata": packaged.metadata,
                "harbor_agent": {
                    "name": "codex",
                    "model_name": model_name,
                    "kwargs": {"reasoning_effort": reasoning_effort},
                    "env": {},
                },
            }
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-base", default=DEFAULT_BACKEND_BASE)
    parser.add_argument("--example-path", type=Path, default=DEFAULT_EXAMPLE_PATH)
    parser.add_argument("--open-thoughts-task", default=DEFAULT_OPEN_THOUGHTS_TASK)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--seed", action="append", type=int, dest="seeds", default=[])
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--pool-suffix", default=str(int(time.time())))
    args = parser.parse_args(argv)

    api_key = load_secret("SYNTH_API_KEY", fallback_files=[])
    client = ContainerPoolsClient(
        api_key=api_key,
        backend_base=args.backend_base,
        timeout=max(args.timeout_s, 900.0),
    )
    if args.open_thoughts_task:
        print(
            json.dumps(
                {
                    "stage": "package_task",
                    "task_id": args.open_thoughts_task,
                    "pool_suffix": args.pool_suffix,
                }
            ),
            flush=True,
        )
        request = _build_open_thoughts_request(
            task_id=args.open_thoughts_task,
            pool_id_suffix=args.pool_suffix,
            model_name=str(args.model),
            reasoning_effort=str(args.reasoning_effort),
        )
    else:
        request = _load_request(args.example_path, pool_id_suffix=args.pool_suffix)
    print(
        json.dumps(
            {
                "stage": "create_pool",
                "pool_id": request["pool_id"],
                "backend": request.get("backend"),
            }
        ),
        flush=True,
    )
    pool = ensure_pool(client, request)
    pool_id = str(request["pool_id"])
    task_id = str((request.get("tasks") or [])[0]["task_id"])

    print(json.dumps({"stage": "pool_ready", "pool_id": pool_id, "status": pool.get("status")}, indent=2), flush=True)

    seeds = args.seeds or [0, 1]

    exit_code = 0
    for seed in seeds:
        print(
            json.dumps(
                {
                    "stage": "create_rollout",
                    "pool_id": pool_id,
                    "task_id": task_id,
                    "seed": seed,
                }
            ),
            flush=True,
        )
        created = client.create_rollout(
            pool_id,
            {
                "task_id": task_id,
                "trace_correlation_id": f"{pool_id}-seed-{seed}",
                "policy": {"model": str(args.model)},
                "messages": [{"role": "user", "content": "Solve the Terminal-Bench Lite task."}],
                "env": {"seed": seed},
            },
        )
        print(
            json.dumps(
                {
                    "stage": "wait_rollout",
                    "pool_id": pool_id,
                    "seed": seed,
                    "rollout_id": created["rollout_id"],
                }
            ),
            flush=True,
        )
        rollout = wait_for_rollout(
            client,
            pool_id=pool_id,
            rollout_id=str(created["rollout_id"]),
            timeout_s=args.timeout_s,
        )
        summary = {
            "seed": seed,
            "rollout_id": rollout["rollout_id"],
            "status": rollout.get("status"),
            "success": rollout.get("success"),
            "score": rollout.get("score"),
            "error": rollout.get("error"),
            "metadata": rollout.get("metadata"),
        }
        print(json.dumps(summary, indent=2), flush=True)
        if not rollout.get("success"):
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
