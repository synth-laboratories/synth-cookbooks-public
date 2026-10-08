"""Loopback synth_containers façade used by cookbook runbooks."""

from __future__ import annotations

import argparse

import uvicorn

from synth_containers.platform import create_compat_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--storage-root", default=None)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("compat serve binds loopback only")
    app = create_compat_app(args.target, storage_root=args.storage_root)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
