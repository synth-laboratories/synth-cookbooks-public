"""Canonical public entrypoint for the normalized HealthBench 2 container."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from synth_containers.platform import create_compat_app


def create_app(*, storage_root: str | Path | None = None):
    """Build an isolated service; callers own its durable storage directory."""

    return create_compat_app("healthbench_chat", storage_root=storage_root)


# ASGI import target for local inspection. Production/dev launchers should call
# main() so storage ownership is explicit rather than process-global magic.
app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the normalized HealthBench 2 target")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8114)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--grader-model", default="gpt-4.1-2025-04-14")
    parser.add_argument("--grader-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--grader-base-url", default="https://api.openai.com/v1")
    args = parser.parse_args()
    args.storage_root.mkdir(parents=True, exist_ok=True)
    os.environ["HEALTHBENCH_GRADER_MODEL"] = args.grader_model
    os.environ["HEALTHBENCH_GRADER_API_KEY_ENV"] = args.grader_api_key_env
    os.environ["HEALTHBENCH_GRADER_BASE_URL"] = args.grader_base_url
    uvicorn.run(
        create_app(storage_root=args.storage_root),
        host=args.host,
        port=args.port,
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    main()
