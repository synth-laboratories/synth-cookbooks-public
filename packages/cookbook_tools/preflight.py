from __future__ import annotations

import json
import os
import platform
import shutil
from pathlib import Path
from typing import Any

from .manifest import CookbookManifest


def _check(
    check_id: str,
    *,
    ok: bool,
    required: Any,
    observed: Any,
    remediation: str,
) -> dict[str, Any]:
    return {
        "id": check_id,
        "status": "passed" if ok else "failed",
        "required": required,
        "observed": observed,
        "remediation": remediation,
    }


def run_preflight(manifest: CookbookManifest) -> dict[str, Any]:
    requirements = dict(manifest.data.get("requirements") or {})
    credentials = list(requirements.get("credentials") or [])
    hardware = dict(requirements.get("hardware") or {})
    platforms = list((manifest.data.get("compatibility") or {}).get("platforms") or [])
    checks: list[dict[str, Any]] = []

    machine = platform.machine().lower()
    system = platform.system().lower()
    if platforms:
        darwin_arm = system == "darwin" and machine in {"arm64", "aarch64"}
        linux_ok = system == "linux"
        allowed = False
        if "macos-arm64" in platforms and darwin_arm:
            allowed = True
        if "linux" in platforms and linux_ok:
            allowed = True
        checks.append(
            _check(
                "platform",
                ok=allowed,
                required=platforms,
                observed=f"{system}/{machine}",
                remediation="Use a listed platform or a different cookbook profile.",
            )
        )

    mem_min = hardware.get("memory_gb_min")
    if mem_min is not None:
        # Observed RAM is best-effort; do not fail closed on unknown.
        checks.append(
            _check(
                "hardware.memory_gb",
                ok=True,
                required=mem_min,
                observed="unmeasured",
                remediation=f"Use a machine with at least {mem_min} GB if the smoke run OOMs.",
            )
        )

    for env_name in credentials:
        present = bool(os.environ.get(env_name))
        checks.append(
            _check(
                f"credentials.{env_name}",
                ok=present,
                required="set",
                observed="present" if present else "missing",
                remediation=f"Export {env_name} without printing its value.",
            )
        )

    python = shutil.which("python3") or shutil.which("python")
    checks.append(
        _check(
            "python",
            ok=python is not None,
            required="python3",
            observed=python or "missing",
            remediation="Install Python 3.11+.",
        )
    )

    compatible = all(row["status"] == "passed" for row in checks)
    return {
        "schema_version": "synth.cookbook.preflight.v1",
        "cookbook_id": manifest.cookbook_id,
        "compatible": compatible,
        "checks": checks,
    }


def dumps(result: dict[str, Any]) -> str:
    return json.dumps(result, indent=2) + "\n"
