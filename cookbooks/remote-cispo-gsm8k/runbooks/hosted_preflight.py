"""Hosted catalog probe. CISPO is not a HostedOptimizer slug yet (gepa/go-ex/ohco)."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


def main() -> int:
    key = os.environ.get("SYNTH_API_KEY")
    base = os.environ.get("SYNTH_BACKEND_URL", "https://api.usesynth.ai").rstrip("/")
    if not key:
        print(
            json.dumps(
                {
                    "compatible": False,
                    "reason": "SYNTH_API_KEY missing",
                    "hosted_algorithms_expected": ["gepa", "go-ex", "ohco"],
                    "cispo_path": "synth-mlx-rl cispo_minimax until hosted training graduates",
                },
                indent=2,
            )
        )
        return 2
    request = urllib.request.Request(
        f"{base}/api/v1/optimizers/startup",
        headers={"authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.URLError as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    algorithms = [
        row.get("algorithm")
        for row in payload.get("available_algorithms") or []
        if isinstance(row, dict)
    ]
    print(
        json.dumps(
            {
                "backend": base,
                "algorithms": algorithms,
                "cispo_listed": "cispo" in algorithms or "cispo_minimax" in algorithms,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
