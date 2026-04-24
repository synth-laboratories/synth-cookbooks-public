from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

try:
    from synth_ai.sdk.container_pools import ContainerPoolsClient
except ModuleNotFoundError:
    from synth_ai.sdk.pools import ContainerPoolsClient


def load_secret(name: str, *, fallback_files: list[Path]) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    for path in fallback_files:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith(f"{name}="):
                continue
            _, _, raw = line.partition("=")
            secret = raw.strip()
            if secret:
                return secret
    raise SystemExit(f"{name} is required.")


def ensure_pool(client: ContainerPoolsClient, request: dict[str, Any]) -> dict[str, Any]:
    pool_id = str(request["pool_id"])
    try:
        return client.create_pool(request)
    except Exception as exc:
        if "409" not in str(exc):
            raise
    return client.update_pool(pool_id, request)


def wait_for_rollout(
    client: ContainerPoolsClient,
    *,
    pool_id: str,
    rollout_id: str,
    timeout_s: float,
) -> dict[str, Any]:
    started = time.time()
    while time.time() - started < timeout_s:
        rollout = client.get_rollout(pool_id, rollout_id)
        status = str(rollout.get("status") or "")
        if status in {"completed", "failed", "cancelled"}:
            return rollout
        time.sleep(2.0)
    raise TimeoutError(f"Timed out waiting for rollout {rollout_id} after {timeout_s:.0f}s")
