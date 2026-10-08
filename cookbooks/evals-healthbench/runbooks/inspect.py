from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def _json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8114")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    health = _json(f"{base}/health")
    info = _json(f"{base}/info")
    gepa = info["capabilities"]["optimizer_contracts"]["gepa"]
    receipt = {
        "target": health.get("target"),
        "runtime_family": info.get("runtime_family") or info.get("metadata", {}).get("runtime_family"),
        "gepa_contract": gepa.get("version"),
        "policy_ready": info["capabilities"]["metadata"].get("policy_ready"),
        "grader_ready": info["capabilities"]["metadata"].get("grader_ready"),
        "note": "eval-only cookbook; do not start a proposer",
    }
    print(json.dumps(receipt, indent=2))
    if health.get("status") != "ok":
        return 1
    if gepa.get("version") != "synth_optimizers.gepa.v2":
        print("unexpected GEPA contract", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
