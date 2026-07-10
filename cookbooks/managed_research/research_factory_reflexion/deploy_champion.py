"""Preview or launch the accepted Reflexion instance on the exe.dev service lane."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from synth_ai.managed_research.sdk.client import ManagedResearchClient


CANDIDATE_SCHEMA = "research_factory_reflexion.champion_candidate.v1"
RECEIPT_SCHEMA = "research_factory_reflexion.champion_deployment_receipt.v1"
TOPOLOGY_ID = "research-reflexion-service"
TOPOLOGY_VERSION = "2026-07-10.v1"
HOST_KIND = "exe_dev"
_ALLOWED_FIELDS = {
    "schema_version",
    "project_id",
    "factory_id",
    "effort_id",
    "instance_id",
    "source_commit_sha",
    "evidence_commit_sha",
    "deployment_name",
}


def _error(message: str) -> ValueError:
    return ValueError(f"reflexion_champion_deployment_error: {message}")


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(f"{field} must be an object")
    return dict(value)


def _text(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise _error(f"{field} is required")
    return text


def _git_sha(value: Any, *, field: str) -> str:
    commit_sha = _text(value, field=field).lower()
    if len(commit_sha) != 40 or any(
        character not in "0123456789abcdef" for character in commit_sha
    ):
        raise _error(f"{field} must be a full git SHA")
    return commit_sha


def _deployment_name(value: Any, *, instance_id: str) -> str:
    requested = str(value or "").strip()
    if requested:
        if len(requested) > 120:
            raise _error("deployment_name cannot exceed 120 characters")
        return requested
    slug = re.sub(r"[^a-z0-9]+", "-", instance_id.lower()).strip("-")
    if not slug:
        raise _error("instance_id cannot produce a deployment name")
    return f"reflexion-{slug}"[:120].rstrip("-")


def _instance_id(value: Any) -> str:
    instance_id = _text(value, field="instance_id")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,239}", instance_id):
        raise _error("instance_id must be a stable non-secret identifier")
    return instance_id


def build_request(candidate: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(candidate)
    if payload.get("schema_version") != CANDIDATE_SCHEMA:
        raise _error(f"schema_version must be {CANDIDATE_SCHEMA}")
    unexpected = sorted(set(payload) - _ALLOWED_FIELDS)
    if unexpected:
        raise _error("unexpected candidate fields: " + ", ".join(unexpected))
    project_id = _text(payload.get("project_id"), field="project_id")
    factory_id = _text(payload.get("factory_id"), field="factory_id")
    effort_id = _text(payload.get("effort_id"), field="effort_id")
    instance_id = _instance_id(payload.get("instance_id"))
    source_commit_sha = _git_sha(
        payload.get("source_commit_sha"), field="source_commit_sha"
    )
    evidence_commit_sha = _git_sha(
        payload.get("evidence_commit_sha"), field="evidence_commit_sha"
    )
    if source_commit_sha == evidence_commit_sha:
        raise _error("source and evidence commits must be distinct")
    return {
        "project_id": project_id,
        "name": _deployment_name(
            payload.get("deployment_name"), instance_id=instance_id
        ),
        "topology_id": TOPOLOGY_ID,
        "topology_version": TOPOLOGY_VERSION,
        "host_kind": HOST_KIND,
        "source": {
            "kind": "project_git",
            "source_commit_sha": source_commit_sha,
            "evidence_commit_sha": evidence_commit_sha,
            "instance_id": instance_id,
        },
        "metadata": {
            "program": "research_factory_reflexion",
            "factory_id": factory_id,
            "effort_id": effort_id,
            "instance_id": instance_id,
        },
    }


def candidate_from_release_evidence(packet: Mapping[str, Any]) -> dict[str, Any]:
    factory = _mapping(packet.get("factory"), field="factory")
    acceptance = _mapping(
        factory.get("b0_c1_c2_acceptance"), field="factory.b0_c1_c2_acceptance"
    )
    if acceptance.get("accepted") is not True or list(
        acceptance.get("cycle_labels") or []
    ) != ["B0", "C1", "C2"]:
        raise _error("release evidence has no accepted B0/C1/C2 chain")
    instance_id = _instance_id(acceptance.get("reflexion_instance_id"))
    instance_head = _git_sha(
        acceptance.get("reflexion_instance_head"),
        field="factory.b0_c1_c2_acceptance.reflexion_instance_head",
    )
    experiment_ids = list(acceptance.get("experiment_ids") or [])
    run_ids = list(acceptance.get("run_ids") or [])
    if (
        len(experiment_ids) != 3
        or len(set(experiment_ids)) != 3
        or len(run_ids) != 3
        or len(set(run_ids)) != 3
    ):
        raise _error("release evidence has no exact B0/C1/C2 IDs")
    audit = _mapping(packet.get("release_audit"), field="release_audit")
    if (
        audit.get("accepted") is not True
        or audit.get("consumed") is not True
        or int(audit.get("use_index") or 0) != 1
        or audit.get("seed_role") != "release_audit"
        or int(audit.get("seed_count") or 0) != 64
        or float(audit.get("ci_lo") or 0.0) <= 0.0
    ):
        raise _error("release evidence has no accepted positive 64-seed audit")
    if (
        audit.get("reflexion_instance_id") != instance_id
        or audit.get("experiment_id") != experiment_ids[2]
        or audit.get("run_id") != run_ids[2]
        or _git_sha(
            audit.get("source_commit_sha"), field="release_audit.source_commit_sha"
        )
        != instance_head
        or not math.isfinite(float(audit.get("ci_lo") or 0.0))
    ):
        raise _error("release audit is not bound to the accepted instance")
    git_receipts = packet.get("git_receipts")
    if not isinstance(git_receipts, list) or not git_receipts:
        raise _error("release evidence has no git receipts")
    latest_git = _mapping(git_receipts[-1], field="git_receipts[-1]")
    if latest_git.get("repo_state_advanced") is not True:
        raise _error("latest git receipt did not advance project git")
    if (
        _git_sha(
            latest_git.get("source_commit_sha"),
            field="git_receipts[-1].source_commit_sha",
        )
        != instance_head
    ):
        raise _error("latest git receipt does not contain the accepted instance head")
    return {
        "schema_version": CANDIDATE_SCHEMA,
        "project_id": _text(factory.get("project_id"), field="factory.project_id"),
        "factory_id": _text(factory.get("factory_id"), field="factory.factory_id"),
        "effort_id": _text(factory.get("effort_id"), field="factory.effort_id"),
        "instance_id": instance_id,
        "source_commit_sha": instance_head,
        "evidence_commit_sha": _git_sha(
            latest_git.get("commit_sha"), field="git_receipts[-1].commit_sha"
        ),
    }


def deploy_champion(
    *,
    client: ManagedResearchClient | None,
    request: Mapping[str, Any],
    launch: bool,
    wait: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    request_payload = dict(request)
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "dry_run": not launch,
        "request": request_payload,
        "deployment": None,
        "observation": None,
    }
    if not launch:
        return receipt
    if client is None:
        raise _error("a ManagedResearchClient is required for --launch")
    if not wait:
        raise _error("--launch requires a final --wait observation receipt")
    created = client.cloud_deployments.create(**request_payload)
    receipt["deployment"] = created
    if wait:
        running = client.cloud_deployments.wait_until_running(
            deployment_id=_text(
                created.get("deployment_id"), field="deployment.deployment_id"
            ),
            timeout_seconds=timeout_seconds,
        )
        receipt["deployment"] = running
        receipt["observation"] = client.cloud_deployments.observe(
            deployment_id=_text(
                running.get("deployment_id"), field="deployment.deployment_id"
            )
        )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--candidate", type=Path)
    input_group.add_argument(
        "--release-evidence",
        type=Path,
        help="Derive the candidate from accepted B0/C1/C2 and release-audit evidence.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Create the retained exe.dev CloudDeployment; default is dry-run.",
    )
    parser.add_argument(
        "--confirm-source-commit",
        help="Required with --launch; must equal candidate.source_commit_sha.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="After launch, wait for running and record a fresh health observation.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    args = parser.parse_args()
    input_path = args.candidate or args.release_evidence
    assert input_path is not None
    input_payload = _mapping(
        json.loads(input_path.read_text(encoding="utf-8")), field=str(input_path)
    )
    candidate = (
        input_payload
        if args.candidate
        else candidate_from_release_evidence(input_payload)
    )
    request = build_request(candidate)
    source = _mapping(request.get("source"), field="request.source")
    if args.wait and not args.launch:
        raise _error("--wait requires --launch")
    if args.launch and not args.wait:
        raise _error("--launch requires --wait")
    if args.launch:
        confirmed = _git_sha(
            args.confirm_source_commit,
            field="--confirm-source-commit",
        )
        if confirmed != source["source_commit_sha"]:
            raise _error("confirmed source commit does not match the candidate")
    receipt = deploy_champion(
        client=ManagedResearchClient() if args.launch else None,
        request=request,
        launch=args.launch,
        wait=args.wait,
        timeout_seconds=args.timeout_seconds,
    )
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_request",
    "candidate_from_release_evidence",
    "deploy_champion",
]
