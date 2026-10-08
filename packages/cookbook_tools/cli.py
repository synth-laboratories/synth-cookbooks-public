from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .manifest import iter_manifests, load_manifest, repo_root
from .preflight import dumps, run_preflight


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="synth-cookbooks")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("catalog")
    pre = sub.add_parser("preflight")
    pre.add_argument("--cookbook", required=True)
    args = parser.parse_args(argv)
    root = repo_root()
    if args.command == "catalog":
        rows = [
            {
                "id": manifest.cookbook_id,
                "title": manifest.data["title"],
                "status": manifest.data["status"],
                "method": manifest.classification.get("method"),
                "task": manifest.classification.get("task"),
            }
            for manifest in iter_manifests(root)
        ]
        print(json.dumps({"cookbooks": rows}, indent=2))
        return 0
    path = Path(args.cookbook)
    if not path.is_file():
        path = root / "cookbooks" / args.cookbook / "cookbook.yaml"
    result = run_preflight(load_manifest(path))
    sys.stdout.write(dumps(result))
    return 0 if result["compatible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
