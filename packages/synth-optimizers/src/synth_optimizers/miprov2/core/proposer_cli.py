"""CLI stepper for interactive MIPRO proposer sessions."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from synth_optimizers.miprov2.core.checkpointing import load_proposer_checkpoint
from synth_optimizers.miprov2.core.proposer_environment import MiproProposerEnvironment
from synth_optimizers.miprov2.core.proposer_openenv import MiproOpenEnvProposerVariant


def _emit(payload: dict[str, Any], *, pretty: bool = False) -> None:
    indent = 2 if pretty else None
    print(json.dumps(payload, indent=indent, sort_keys=True, ensure_ascii=True))


def _load_variant(path: Path | None) -> MiproOpenEnvProposerVariant | None:
    if path is None:
        return None
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    return MiproOpenEnvProposerVariant.from_dict(payload)


def _load_args_json(text: str | None) -> dict[str, Any]:
    if text is None or not str(text).strip():
        return {}
    payload = json.loads(str(text))
    if not isinstance(payload, dict):
        raise ValueError("--args-json must decode to a JSON object")
    return dict(payload)


def _load_env(args: argparse.Namespace) -> MiproProposerEnvironment:
    return MiproProposerEnvironment.load(
        session_root=Path(args.session_root).expanduser().resolve(),
        session_id=str(args.session),
    )


def _cmd_start(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = load_proposer_checkpoint(checkpoint_path)
    env = MiproProposerEnvironment.from_checkpoint(
        checkpoint,
        session_root=Path(args.session_root).expanduser().resolve(),
        source_ref=str(checkpoint_path),
        variant=_load_variant(args.variant_json),
        actor_id=str(args.actor_id),
    )
    return {
        "status": "ok",
        "session_id": env.session.session_id,
        "session_dir": env.session.session_dir,
        "event_log_path": env.session.event_log_path,
        "current_version": env.session.current_version,
    }


def _cmd_tools(args: argparse.Namespace) -> dict[str, Any]:
    return _load_env(args).list_tools()


def _cmd_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    return _load_env(args).snapshot()


async def _cmd_call(args: argparse.Namespace) -> dict[str, Any]:
    env = _load_env(args)
    return await env.call_tool(
        str(args.tool),
        _load_args_json(args.args_json),
        actor_id=str(args.actor_id),
        expected_version=args.expected_version,
    )


def _cmd_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    return _load_env(args).checkpoint(actor_id=str(args.actor_id))


def _cmd_commit(args: argparse.Namespace) -> dict[str, Any]:
    return _load_env(args).commit(actor_id=str(args.actor_id))


def _cmd_discard(args: argparse.Namespace) -> dict[str, Any]:
    return _load_env(args).discard(actor_id=str(args.actor_id))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step through a MIPRO proposer session.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="Start a session from a checkpoint.")
    start.add_argument("--checkpoint", type=Path, required=True)
    start.add_argument("--session-root", type=Path, required=True)
    start.add_argument("--variant-json", type=Path)
    start.add_argument("--actor-id", default="interactive")

    for name in ("tools", "snapshot", "checkpoint", "commit", "discard"):
        command = subparsers.add_parser(name)
        command.add_argument("--session", required=True)
        command.add_argument("--session-root", type=Path, required=True)
        command.add_argument("--actor-id", default="interactive")

    call = subparsers.add_parser("call", help="Call one proposer tool.")
    call.add_argument("--session", required=True)
    call.add_argument("--session-root", type=Path, required=True)
    call.add_argument("--tool", required=True)
    call.add_argument("--args-json", default="{}")
    call.add_argument("--actor-id", default="interactive")
    call.add_argument("--expected-version", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "start":
            payload = _cmd_start(args)
        elif args.command == "tools":
            payload = _cmd_tools(args)
        elif args.command == "snapshot":
            payload = _cmd_snapshot(args)
        elif args.command == "call":
            payload = asyncio.run(_cmd_call(args))
        elif args.command == "checkpoint":
            payload = _cmd_checkpoint(args)
        elif args.command == "commit":
            payload = _cmd_commit(args)
        elif args.command == "discard":
            payload = _cmd_discard(args)
        else:
            raise ValueError(f"unknown command: {args.command}")
        _emit(payload, pretty=bool(args.pretty))
        return 0
    except Exception as exc:
        _emit({"status": "error", "reason": str(exc)}, pretty=bool(args.pretty))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
