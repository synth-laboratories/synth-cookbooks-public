from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from synth_containers.compat.archipelago import ArchipelagoProxyRuntime, ArchipelagoTarget
from synth_containers.formats import (
    execution_to_rollout_payload,
    metadata_to_http_payload,
    task_info_to_http_payload,
)


def _load_json_arg(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    candidate = Path(value)
    if candidate.exists():
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    else:
        payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("--task-config-json must resolve to a JSON object")
    return payload


def _redact(payload: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(payload)
    for key in ("auth_token", "container_worker_token", "api_key"):
        if redacted.get(key):
            redacted[key] = "<present>"
    return redacted


async def _run(args: argparse.Namespace) -> int:
    service_url = str(args.service_url or os.environ.get("ARCHIPELAGO_SERVICE_URL") or "").strip()
    task_config = _load_json_arg(args.task_config_json)
    if not service_url:
        print(
            json.dumps(
                {
                    "status": "missing_service_url",
                    "required": ["ARCHIPELAGO_SERVICE_URL or --service-url"],
                    "task_config": _redact(task_config),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    target = ArchipelagoTarget(
        service_url=service_url.rstrip("/"),
        provider=str(task_config.get("provider") or args.provider),
        interface_mode=str(task_config.get("interface_mode") or "synth_http"),
        world_id=str(task_config.get("world_id") or args.world_id or "").strip() or None,
        dataset_name=str(task_config.get("dataset_name") or args.dataset_name or "").strip()
        or None,
        limits=dict(task_config.get("limits") or {"timeout_s": args.timeout_s}),
        auth_token=str(args.auth_token or os.environ.get("ARCHIPELAGO_AUTH_TOKEN") or "").strip()
        or None,
    )
    runtime = ArchipelagoProxyRuntime(
        target=target,
        task_path=str(task_config.get("task_path") or args.task_path or "").strip(),
    )
    payload: dict[str, Any] = {
        "runtime": metadata_to_http_payload(runtime.metadata()),
        "task_info": task_info_to_http_payload(runtime.task_info()),
        "task_catalog": runtime.task_catalog().to_dict(),
    }
    if args.rollout:
        request = {
            "trace_correlation_id": args.trace_correlation_id,
            "env": {
                "seed": args.seed,
                "config": {
                    **task_config,
                    "task_path": str(task_config.get("task_path") or args.task_path or "").strip(),
                    "world_id": target.world_id,
                    "dataset_name": target.dataset_name,
                },
            },
            "policy": {"config": {"model": args.model}},
        }
        execution = await runtime.submit_rollout(request)
        payload["rollout"] = execution_to_rollout_payload(execution)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an Archipelago synth-containers eval wrapper."
    )
    parser.add_argument("--service-url", default="")
    parser.add_argument("--task-path", default="")
    parser.add_argument("--task-config-json", default="")
    parser.add_argument("--world-id", default="")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--provider", default="archipelago")
    parser.add_argument("--auth-token", default="")
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="openai/gpt-5.4-nano")
    parser.add_argument("--trace-correlation-id", default="archipelago_eval_smoke")
    parser.add_argument("--rollout", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        args.rollout = False
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
