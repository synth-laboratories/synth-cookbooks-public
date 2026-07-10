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

try:
    from .release_gate import validate_predeployment_evidence
except ImportError:
    from release_gate import validate_predeployment_evidence


CANDIDATE_SCHEMA = "research_factory_reflexion.champion_candidate.v1"
RECEIPT_SCHEMA = "research_factory_reflexion.champion_deployment_receipt.v1"
TOPOLOGY_ID = "research-reflexion-service"
TOPOLOGY_VERSION = "2026-07-10.v1"
HOST_KIND = "exe_dev"
RELEASE_AUDIT_SPLIT = "craftax_release_audit_v1_64"
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


def _sequence(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise _error(f"{field} must be an array")
    return list(value)


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


def _sha256(value: Any, *, field: str) -> str:
    digest = _text(value, field=field).lower()
    if not digest.startswith("sha256:"):
        digest = "sha256:" + digest
    if len(digest) != 71 or any(char not in "0123456789abcdef" for char in digest[7:]):
        raise _error(f"{field} must be a SHA-256 digest")
    return digest


def _reflexion_instance(value: Any, *, field: str) -> dict[str, Any]:
    instance = _mapping(value, field=field)
    if (
        instance.get("created_from_scratch") is not True
        or _text(
            instance.get("implementation_root"),
            field=f"{field}.implementation_root",
        )
        != "reflexion_instance"
    ):
        raise _error(f"{field} is not a Factory-built Reflexion instance")
    cookbook = _mapping(
        instance.get("cookbook_reference"), field=f"{field}.cookbook_reference"
    )
    if (
        cookbook.get("role") != "reference_design_only"
        or cookbook.get("source_mounted") is not False
        or cookbook.get("source_copied") is not False
    ):
        raise _error(f"{field} treats the cookbook as runtime source")
    if not _sequence(instance.get("source_files"), field=f"{field}.source_files"):
        raise _error(f"{field}.source_files must not be empty")
    return instance


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
    try:
        predeployment = validate_predeployment_evidence(
            packet, require_operating_proof=False
        )
    except ValueError as exc:
        raise _error(
            f"release evidence failed predeployment validation: {exc}"
        ) from exc
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
    try:
        audit_statistics = {
            field: float(audit.get(field)) for field in ("mean_delta", "ci_lo", "ci_hi")
        }
    except (TypeError, ValueError) as exc:
        raise _error("release audit statistics must be numeric") from exc
    if (
        audit.get("accepted") is not True
        or audit.get("consumed") is not True
        or int(audit.get("use_index") or 0) != 1
        or audit.get("seed_role") != "release_audit"
        or int(audit.get("seed_count") or 0) != 64
        or audit_statistics["ci_lo"] <= 0.0
    ):
        raise _error("release evidence has no accepted positive 64-seed audit")
    if (
        audit.get("reflexion_instance_id") != instance_id
        or _git_sha(
            audit.get("source_commit_sha"), field="release_audit.source_commit_sha"
        )
        != instance_head
        or any(not math.isfinite(value) for value in audit_statistics.values())
    ):
        raise _error("release audit is not bound to the accepted instance")
    audit_experiment_id = _text(
        audit.get("experiment_id"), field="release_audit.experiment_id"
    )
    audit_run_id = _text(audit.get("run_id"), field="release_audit.run_id")
    if audit_experiment_id in set(experiment_ids) or audit_run_id in set(run_ids):
        raise _error("release audit must be distinct from the B0/C1/C2 chain")
    experiments = _mapping(packet.get("experiments"), field="experiments")
    bundles = experiments.get("bundles")
    if not isinstance(bundles, list):
        raise _error("experiments.bundles must be an array")
    audit_bundles = [
        _mapping(raw, field=f"experiments.bundles[{index}]")
        for index, raw in enumerate(bundles)
        if isinstance(raw, Mapping)
        and str(raw.get("experiment_id") or "").strip() == audit_experiment_id
    ]
    if len(audit_bundles) != 1:
        raise _error("release audit must have exactly one experiment bundle")
    audit_bundle = audit_bundles[0]
    audit_experiment = _mapping(
        audit_bundle.get("experiment"), field="release_audit.bundle.experiment"
    )
    if audit_experiment.get("status") != "accepted" or str(
        audit_experiment.get("verdict") or ""
    ).lower() not in {"accept", "promote"}:
        raise _error("release audit experiment is not accepted")
    audit_run_ids = audit_bundle.get("run_ids")
    if not isinstance(audit_run_ids, list) or audit_run_id not in {
        _text(item, field="release_audit.bundle.run_ids[]") for item in audit_run_ids
    }:
        raise _error("release audit run does not belong to its experiment bundle")
    audit_integrity = _mapping(
        audit_bundle.get("integrity"), field="release_audit.bundle.integrity"
    )
    if audit_integrity.get("accepted_cycle") is not True or audit_integrity.get(
        "missing"
    ) not in (None, []):
        raise _error("release audit experiment bundle is not evidence-complete")
    audit_candidate = _mapping(
        audit_bundle.get("candidate"), field="release_audit.bundle.candidate"
    )
    audit_snapshot = _mapping(
        audit_candidate.get("snapshot"),
        field="release_audit.bundle.candidate.snapshot",
    )
    audit_instance = _reflexion_instance(
        audit_snapshot.get("reflexion_instance"),
        field="release_audit.bundle.reflexion_instance",
    )
    if (
        _instance_id(audit_instance.get("instance_id")) != instance_id
        or _git_sha(
            audit_instance.get("git_commit_sha"),
            field="release_audit.bundle.reflexion_instance.git_commit_sha",
        )
        != instance_head
    ):
        raise _error("release audit bundle does not target the accepted instance head")
    audit_evaluations = [
        _mapping(raw, field=f"release_audit.bundle.evaluations[{index}]")
        for index, raw in enumerate(
            _sequence(
                audit_bundle.get("evaluations"),
                field="release_audit.bundle.evaluations",
            )
        )
        if isinstance(raw, Mapping)
        and str(raw.get("run_id") or "").strip() == audit_run_id
        and str(raw.get("split_name") or "").strip() == RELEASE_AUDIT_SPLIT
    ]
    if len(audit_evaluations) != 1:
        raise _error("release audit requires exactly one run-specific evaluation")
    audit_evaluation = audit_evaluations[0]
    try:
        audit_seed_set = {
            int(item)
            for item in _sequence(
                audit_evaluation.get("seed_set"),
                field="release_audit.bundle.evaluation.seed_set",
            )
        }
        evaluation_value = float(audit_evaluation.get("value"))
        evaluation_baseline = float(audit_evaluation.get("baseline_value"))
        evaluation_delta = float(audit_evaluation.get("delta"))
    except (TypeError, ValueError) as exc:
        raise _error("release audit evaluation has invalid seed or delta data") from exc
    if (
        int(audit_evaluation.get("sample_size") or 0) != 64
        or len(audit_seed_set) != 64
        or audit_evaluation.get("evidence_grade") != "release_evidence"
        or audit_evaluation.get("truth_status") not in {"attested", "verified"}
        or not all(
            math.isfinite(value)
            for value in (evaluation_value, evaluation_baseline, evaluation_delta)
        )
        or not math.isclose(
            evaluation_value - evaluation_baseline,
            evaluation_delta,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or evaluation_delta != audit_statistics["mean_delta"]
        or not audit_evaluation.get("result_id")
        or not audit_evaluation.get("scorer_id")
        or not audit_evaluation.get("per_example_artifact_id")
        or not audit_evaluation.get("summary_artifact_id")
    ):
        raise _error("release audit evaluation is not 64-seed release evidence")
    evaluation_metadata = _mapping(
        audit_evaluation.get("metadata"),
        field="release_audit.bundle.evaluation.metadata",
    )
    embedded_audit = _mapping(
        evaluation_metadata.get("release_audit"),
        field="release_audit.bundle.evaluation.metadata.release_audit",
    )
    embedded_identity = {
        "schema_version": "craftax_factory.release_audit.v1",
        "consumed": True,
        "use_index": 1,
        "seed_role": "release_audit",
        "seed_count": 64,
        "experiment_id": audit_experiment_id,
        "run_id": audit_run_id,
        "reflexion_instance_id": instance_id,
        "source_commit_sha": instance_head,
        "accepted": True,
        "registry_version": _text(
            audit.get("registry_version"), field="release_audit.registry_version"
        ),
    }
    if any(
        embedded_audit.get(key) != value for key, value in embedded_identity.items()
    ):
        raise _error("run-specific evaluation embeds a different release audit")
    for field, value in audit_statistics.items():
        try:
            embedded_value = float(embedded_audit.get(field))
        except (TypeError, ValueError) as exc:
            raise _error(f"embedded release audit {field} must be numeric") from exc
        if not math.isfinite(embedded_value) or embedded_value != value:
            raise _error(f"embedded release audit {field} differs from the claim")
    scorecard_digest = _sha256(
        audit.get("scorecard_digest"), field="release_audit.scorecard_digest"
    )
    if (
        _sha256(
            embedded_audit.get("scorecard_digest"),
            field="release_audit.bundle.evaluation.scorecard_digest",
        )
        != scorecard_digest
    ):
        raise _error("release audit evaluation cites a different scorecard")
    container_run_id = _text(
        audit_evaluation.get("container_run_id"),
        field="release_audit.bundle.evaluation.container_run_id",
    )
    audit_executions = [
        _mapping(raw, field=f"release_audit.bundle.executions[{index}]")
        for index, raw in enumerate(
            _sequence(
                audit_bundle.get("executions"),
                field="release_audit.bundle.executions",
            )
        )
        if isinstance(raw, Mapping)
        and str(raw.get("run_id") or "").strip() == audit_run_id
        and str(raw.get("container_run_id") or "").strip() == container_run_id
    ]
    audit_execution = audit_executions[0] if len(audit_executions) == 1 else {}
    if (
        audit_execution.get("status") not in {"completed", "done"}
        or _sha256(
            audit_execution.get("container_digest"),
            field="release_audit.bundle.execution.container_digest",
        )
        == ""
        or audit_execution.get("scorer_id") != audit_evaluation.get("scorer_id")
        or not list(audit_execution.get("task_ids") or [])
    ):
        raise _error("release audit evaluation has no completed matching execution")
    provenance = _mapping(
        audit_bundle.get("provenance"), field="release_audit.bundle.provenance"
    )
    audit_git = _mapping(
        provenance.get("git_server"),
        field="release_audit.bundle.provenance.git_server",
    )
    if (
        audit_git.get("repo_state_advanced") is not True
        or audit_git.get("source_run_id") != audit_run_id
        or _git_sha(
            audit_git.get("source_commit_sha"),
            field="release_audit.bundle.git_server.source_commit_sha",
        )
        != instance_head
    ):
        raise _error("release audit git receipt is not source/run bound")
    audit_evidence_commit = _git_sha(
        audit_git.get("commit_sha"), field="release_audit.bundle.git_server.commit_sha"
    )
    if audit_evidence_commit == instance_head:
        raise _error("release audit evidence commit cannot equal the source commit")
    predeployment_knowledge = _mapping(
        predeployment.get("knowledge_git"), field="predeployment.knowledge_git"
    )
    if (
        _git_sha(
            predeployment_knowledge.get("evidence_commit"),
            field="predeployment.knowledge_git.evidence_commit",
        )
        != audit_evidence_commit
    ):
        raise _error("deployment evidence commit differs from the release gate")
    git_receipts = _sequence(packet.get("git_receipts"), field="git_receipts")
    matching_audit_git = [
        _mapping(raw, field=f"git_receipts[{index}]")
        for index, raw in enumerate(git_receipts)
        if isinstance(raw, Mapping)
        and str(raw.get("source_run_id") or "").strip() == audit_run_id
    ]
    if len(matching_audit_git) != 1 or any(
        matching_audit_git[0].get(field) != audit_git.get(field)
        for field in (
            "commit_sha",
            "source_commit_sha",
            "remote_repo",
            "branch",
            "source_run_id",
            "repo_state_advanced",
        )
    ):
        raise _error("top-level audit git receipt differs from experiment history")
    return {
        "schema_version": CANDIDATE_SCHEMA,
        "project_id": _text(factory.get("project_id"), field="factory.project_id"),
        "factory_id": _text(factory.get("factory_id"), field="factory.factory_id"),
        "effort_id": _text(factory.get("effort_id"), field="factory.effort_id"),
        "instance_id": instance_id,
        "source_commit_sha": instance_head,
        "evidence_commit_sha": audit_evidence_commit,
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
