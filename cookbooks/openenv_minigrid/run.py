from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COOKBOOK_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS_DIR = COOKBOOK_ROOT / "run_artifacts" / "dry_run"
DEFAULT_BASE_URL = "http://127.0.0.1:8922"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_summary(path: Path, plan: dict[str, Any]) -> None:
    lines = [
        "# MiniGrid PipelineRL/Modal Dry Run",
        "",
        f"- generated_at: `{plan['generated_at']}`",
        f"- mode: `{plan['mode']}`",
        f"- container: `{plan['container']['path']}`",
        f"- trainer: `{plan['optimizer']['trainer']}`",
        f"- base_url: `{plan['execution']['base_url']}`",
        f"- result_status: `{plan['result_status']}`",
        "",
        "This artifact scopes the local smoke, checkpoint proof, and Modal",
        "training boundary. It does not launch Modal by default.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "cookbook": "openenv_minigrid",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "execute_local_smoke" if args.execute_local_smoke else "dry_run",
        "container": {
            "path": "minigrid_container",
            "service_command": [
                "python",
                "minigrid_container/synth_service_app.py",
            ],
            "source_files": [
                "synth_service_app.py",
                "container_spec.json",
                "task_registry.json",
                "service_app.py",
                "synth_service_app.py",
                "smoke.py",
                "snapshot_proof.py",
            ],
        },
        "optimizer": {
            "path": "pipelinerl_modal",
            "trainer": "PipelineRL via Modal",
            "plan": "modal_pipelinerl_plan.json",
        },
        "execution": {
            "base_url": args.base_url,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "smoke_command": [
                "python",
                "minigrid_container/smoke.py",
                "--base-url",
                args.base_url,
                "--seed",
                str(args.seed),
                "--max-steps",
                str(args.max_steps),
            ],
            "snapshot_command": [
                "python",
                "minigrid_container/snapshot_proof.py",
                "--base-url",
                args.base_url,
                "--seed",
                str(args.seed),
            ],
        },
        "expected_artifacts": [
            "plan.json",
            "summary.md",
            "local_smoke_command_result.json",
            "snapshot_proof_command_result.json",
        ],
        "result_status": "inconclusive",
    }


def _run_and_capture(command: list[str], cwd: Path, output_path: Path) -> int:
    completed = subprocess.run(command, cwd=cwd, check=False, text=True, capture_output=True)
    payload = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    _write_json(output_path, payload)
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare or run the MiniGrid PipelineRL cookbook.")
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--execute-local-smoke", action="store_true")
    args = parser.parse_args(argv)

    artifacts_dir = Path(args.artifacts_dir)
    plan = build_plan(args)
    _write_json(artifacts_dir / "plan.json", plan)
    _write_summary(artifacts_dir / "summary.md", plan)
    if not args.execute_local_smoke:
        print(f"wrote dry-run artifacts to {artifacts_dir}")
        return 0

    smoke_rc = _run_and_capture(
        plan["execution"]["smoke_command"],
        COOKBOOK_ROOT,
        artifacts_dir / "local_smoke_command_result.json",
    )
    snapshot_rc = _run_and_capture(
        plan["execution"]["snapshot_command"],
        COOKBOOK_ROOT,
        artifacts_dir / "snapshot_proof_command_result.json",
    )
    return 0 if smoke_rc == 0 and snapshot_rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
