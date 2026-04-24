from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COOKBOOK_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS_DIR = COOKBOOK_ROOT / "run_artifacts" / "dry_run"
DEFAULT_BACKEND_BASE = "http://127.0.0.1:8001"
DEFAULT_MODEL = "openai/gpt-5.4-nano"
SYNTH_AI_VERSION = "0.9.11"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_summary(path: Path, plan: dict[str, Any]) -> None:
    lines = [
        "# Harbor/TBLite Codex PipelineRL/Tinker Dry Run",
        "",
        f"- generated_at: `{plan['generated_at']}`",
        f"- mode: `{plan['mode']}`",
        f"- container: `{plan['container']['path']}`",
        f"- optimizer: `{plan['optimizer']['path']}`",
        f"- backend_base: `{plan['execution']['backend_base']}`",
        f"- model: `{plan['execution']['model']}`",
        f"- result_status: `{plan['result_status']}`",
        "",
        "This artifact is a public-safe execution plan. It does not claim an",
        "uplift until a real baseline/train/heldout run has been executed.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "cookbook": "harbor_tblite_codex",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "execute_one_shot" if args.execute_one_shot else "dry_run",
        "container": {
            "path": "tblite_codex_container",
            "public_entrypoint": "tblite_codex_container/synth_service_app.py",
            "source_files": [
                "synth_service_app.py",
                "tb_lite_harbor_pool.json",
                "tb_lite_dataset.py",
                "run_tblite_codex_nano_once.py",
                "smoke.py",
            ],
        },
        "optimizer": {
            "path": "pipelinerl_tinker",
            "trainer": "PipelineRL via Tinker",
            "plan": "pipelinerl_tinker_plan.json",
        },
        "execution": {
            "backend_base": args.backend_base,
            "model": args.model,
            "seed": args.seed,
            "synth_ai_version": SYNTH_AI_VERSION,
            "dry_run_default": True,
            "contract_smoke_command": [
                "PYTHONPATH=packages/synth-containers/src:$(pwd)/cookbooks/harbor_tblite_codex/tblite_codex_container",
                "PORT=8952",
                "python",
                "cookbooks/harbor_tblite_codex/tblite_codex_container/synth_service_app.py",
            ],
            "one_shot_command": [
                "uv",
                "run",
                "--with",
                f"synth-ai=={SYNTH_AI_VERSION}",
                "python",
                "tblite_codex_container/run_tblite_codex_nano_once.py",
                "--backend-base",
                args.backend_base,
                "--model",
                args.model,
                "--seed",
                str(args.seed),
                "--output-root",
                str(Path("run_artifacts") / "live_one_shot"),
            ],
        },
        "expected_artifacts": [
            "plan.json",
            "summary.md",
            "live_one_shot/artifacts/rollout_summary.json",
            "live_one_shot/reports/summary.md",
        ],
        "result_status": "inconclusive",
        "public_runtime_note": "Use synth_service_app.py for the public synth-containers contract. The one-shot Harbor backend path is migration reference only.",
    }


def execute_one_shot(plan: dict[str, Any], cwd: Path) -> int:
    completed = subprocess.run(
        plan["execution"]["one_shot_command"],
        cwd=cwd,
        check=False,
        text=True,
        capture_output=True,
    )
    live_dir = cwd / Path(plan["execution"]["one_shot_command"][-1])
    _write_json(
        live_dir / "command_result.json",
        {
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        },
    )
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare or run the Harbor/TBLite Codex cookbook.")
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--backend-base", default=DEFAULT_BACKEND_BASE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--execute-one-shot", action="store_true")
    args = parser.parse_args(argv)

    artifacts_dir = Path(args.artifacts_dir)
    plan = build_plan(args)
    _write_json(artifacts_dir / "plan.json", plan)
    _write_summary(artifacts_dir / "summary.md", plan)
    if not args.execute_one_shot:
        print(f"wrote dry-run artifacts to {artifacts_dir}")
        return 0
    return execute_one_shot(plan, COOKBOOK_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
