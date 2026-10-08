"""Live E2E against current Containers using keys from a local .env (never printed)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient
from synth_containers.platform import create_compat_app

ROOT = Path(__file__).resolve().parents[1]
ENV_CANDIDATES = [
    Path("/Users/joshuapurtell/Documents/GitHub/backend/.env.local"),
    Path("/Users/joshuapurtell/Documents/GitHub/synth-ai/.env"),
]
TELEMETRY = {"enabled": True, "transport": "sse", "retention": "run"}
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_env() -> Path:
    for path in ENV_CANDIDATES:
        if not path.is_file():
            continue
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and value and key not in os.environ:
                os.environ[key] = value
        return path
    raise SystemExit("no .env with provider keys found")


def _present(name: str) -> bool:
    return bool(os.environ.get(name, "").strip())


def live_banking77() -> dict:
    client = TestClient(create_compat_app("banking77_classify"))
    registered = client.post(
        "/policy-configs",
        json={
            "config_id": "live_nano",
            "harness": "classify",
            "config": {
                "provider": "openai",
                "model": "gpt-4.1-nano",
                "api_key_env": "OPENAI_API_KEY",
                "base_url": "https://api.openai.com/v1",
                "temperature": 0,
                "max_tokens": 32,
            },
        },
    )
    assert registered.status_code == 200, registered.text
    rid = "b77_live_0"
    client.post("/rollouts/prepare", json={"rollout_id": rid, "telemetry": TELEMETRY})
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": rid,
            "telemetry": TELEMETRY,
            "slot": "stream",
            "world_ref": "world:banking77@train",
            "task_instance_id": "seed:0",
            "evaluation_plan_ref": "banking77_eval.v1",
            "policy_ref": {"harness": "classify", "config": "live_nano"},
        },
    )
    body = started.json()
    scored = client.post("/reward", json={"rollout_id": rid, "mode": "terminal"}).json()
    return {
        "http": started.status_code,
        "status": body.get("status") or scored.get("status"),
        "reward": scored.get("reward"),
        "reward_status": scored.get("status"),
        "usage_tokens": (body.get("usage") or {}).get("total_tokens"),
        "model": "gpt-4.1-nano",
    }


def live_healthbench() -> dict:
    os.environ["HEALTHBENCH_GRADER_MODEL"] = "gpt-4.1-mini-2025-04-14"
    client = TestClient(create_compat_app("healthbench_chat"))
    rid = "hb_live_0"
    client.post("/rollouts/prepare", json={"rollout_id": rid, "telemetry": TELEMETRY})
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": rid,
            "slot": "stream",
            "telemetry": TELEMETRY,
            "task_instance_id": "seed:0",
            "world_ref": "world:healthbench@eval",
            "evaluation_plan_ref": "healthbench_scaled_grader.v1",
            "policy_ref": {"harness": "chat_completion", "config": "openai_gpt41_mini"},
        },
    )
    scored = client.post("/reward", json={"rollout_id": rid, "mode": "terminal"}).json()
    usage = started.json().get("usage") or {}
    return {
        "http": started.status_code,
        "start_status": started.json().get("status"),
        "reward": scored.get("reward"),
        "reward_status": scored.get("status"),
        "grader": "gpt-4.1-mini-2025-04-14",
        "plan": "healthbench_scaled_grader.v1",
        "cost_kind": usage.get("cost_kind"),
        "cost_usd": usage.get("cost_usd"),
        "detail": started.text[:400] if started.status_code != 200 else None,
    }


def hosted_probe() -> dict:
    import urllib.error
    import urllib.request

    key = os.environ.get("SYNTH_API_KEY", "").strip()
    base = os.environ.get("SYNTH_BACKEND_URL", "https://api.usesynth.ai").rstrip("/")
    if base.startswith("http://127.0.0.1") or "localhost" in base:
        base = "https://api.usesynth.ai"
    if not key:
        return {"ok": False, "reason": "SYNTH_API_KEY missing"}
    request = urllib.request.Request(
        f"{base}/api/v1/optimizers/startup",
        headers={"authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        return {"ok": False, "http": exc.code, "backend": base}
    except urllib.error.URLError as exc:
        return {"ok": False, "error_type": type(exc).__name__, "backend": base}
    algorithms = [
        row.get("algorithm")
        for row in payload.get("available_algorithms") or []
        if isinstance(row, dict)
    ]
    return {
        "ok": True,
        "backend": base,
        "algorithms": algorithms,
        "cispo_listed": any("cispo" in str(item) for item in algorithms),
    }


def merge_evidence(cookbook_id: str, live: dict) -> None:
    path = ROOT / "cookbooks" / cookbook_id / "evidence" / "evidence.json"
    data = json.loads(path.read_text()) if path.is_file() else {"cookbook_id": cookbook_id}
    data["live_e2e"] = {"proved_at": NOW, **live}
    path.write_text(json.dumps(data, indent=2) + "\n")


def main() -> int:
    env_path = load_env()
    print(f"loaded_env_file={env_path.name} openai={_present('OPENAI_API_KEY')} groq={_present('GROQ_API_KEY')} synth={_present('SYNTH_API_KEY')}")
    banking = live_banking77()
    print(f"banking77_live reward={banking.get('reward')} status={banking.get('reward_status')} http={banking.get('http')}")
    merge_evidence("gepa-banking77", banking)
    health = live_healthbench()
    print(
        f"healthbench_live reward={health.get('reward')} status={health.get('reward_status')} http={health.get('http')}"
    )
    merge_evidence("evals-healthbench", health)
    hosted = hosted_probe()
    print(f"hosted_probe ok={hosted.get('ok')} cispo_listed={hosted.get('cispo_listed')} http={hosted.get('http')}")
    merge_evidence("remote-cispo-gsm8k", hosted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
