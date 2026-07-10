"""Read one Reflexion Factory through typed owner routes; never mutate it."""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from synth_ai.managed_research.sdk.client import ManagedResearchClient


def _wire(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _wire(dataclasses.asdict(value))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _wire(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_wire(item) for item in value]
    to_wire = getattr(value, "to_wire", None)
    return _wire(to_wire()) if callable(to_wire) else value


def inspect_factory(
    *,
    client: ManagedResearchClient,
    factory_id: str,
    project_id: str,
    history_limit: int,
) -> dict[str, Any]:
    status = client.factories.status(factory_id)
    history = client.factories.experiment_history(project_id, limit=history_limit)
    history_wire = _wire(history)
    bundles = list(history_wire.get("bundles") or [])
    experiment_ids = [
        str(bundle.get("experiment_id"))
        for bundle in bundles
        if isinstance(bundle, Mapping) and bundle.get("experiment_id")
    ]
    comparison = (
        _wire(client.factories.compare_experiments(project_id, experiment_ids))
        if len(experiment_ids) >= 2
        else None
    )
    champion_deployments = _wire(
        client.cloud_deployments.list(project_id=project_id, limit=25)
    )
    hosted_artifacts = _wire(client.list_project_hosted_artifacts(project_id))
    return {
        "schema_version": "research_factory_reflexion.inspection.v1",
        "factory_id": factory_id,
        "project_id": project_id,
        "factory_status": _wire(status),
        "experiment_history": history_wire,
        "experiment_comparison": comparison,
        "champion_deployments": champion_deployments,
        "hosted_artifacts": hosted_artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory-id", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--history-limit", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = inspect_factory(
        client=ManagedResearchClient(),
        factory_id=args.factory_id,
        project_id=args.project_id,
        history_limit=args.history_limit,
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
