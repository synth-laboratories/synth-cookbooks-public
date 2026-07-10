"""Fail closed unless a Reflexion Factory is adoptable, publishable, and 24/7-proven."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKET_SCHEMA = "research_factory_reflexion.release_evidence.v1"
RECEIPT_SCHEMA = "research_factory_reflexion.release_receipt.v1"
REQUIRED_SOURCE_COMPONENTS = {"backend", "evals", "synth-dev", "synth-ai", "gamebench"}
REQUIRED_CYCLES = ("B0", "C1", "C2")
REQUIRED_RAILWAY_ENVIRONMENTS = ("dev", "staging", "prod")
RELEASE_AUDIT_SPLIT = "craftax_release_audit_v1_64"


def _error(message: str) -> ValueError:
    return ValueError(f"research_factory_release_error: {message}")


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(f"{field} must be an object")
    return dict(value)


def _sequence(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise _error(f"{field} must be an array")
    return list(value)


def _text(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise _error(f"{field} is required")
    return text


def _sha256(value: Any, *, field: str) -> str:
    digest = _text(value, field=field).lower()
    if not digest.startswith("sha256:"):
        digest = "sha256:" + digest
    if len(digest) != 71 or any(char not in "0123456789abcdef" for char in digest[7:]):
        raise _error(f"{field} must be a SHA-256 digest")
    return digest


def _git_sha(value: Any, *, field: str) -> str:
    commit_sha = _text(value, field=field).lower()
    if len(commit_sha) != 40 or any(
        char not in "0123456789abcdef" for char in commit_sha
    ):
        raise _error(f"{field} must be a full git SHA")
    return commit_sha


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
    source_files = _sequence(
        instance.get("source_files"), field=f"{field}.source_files"
    )
    if not source_files:
        raise _error(f"{field}.source_files must not be empty")
    return instance


def _source_evidence(packet: Mapping[str, Any]) -> dict[str, Any]:
    source = _mapping(packet.get("source"), field="source")
    manifest_digest = _sha256(
        source.get("manifest_digest"), field="source.manifest_digest"
    )
    runtime_image_digest = _sha256(
        source.get("runtime_image_digest"), field="source.runtime_image_digest"
    )
    components = _mapping(source.get("components"), field="source.components")
    if set(components) != REQUIRED_SOURCE_COMPONENTS:
        raise _error(
            "source.components must contain exactly the five runtime authorities"
        )
    for name, raw in components.items():
        component = _mapping(raw, field=f"source.components.{name}")
        _git_sha(
            component.get("commit_sha"), field=f"source.components.{name}.commit_sha"
        )
        if component.get("clean") is not True:
            raise _error(f"source component is not clean: {name}")
    return {
        "manifest_digest": manifest_digest,
        "runtime_image_digest": runtime_image_digest,
        "components": components,
    }


def _factory_evidence(
    packet: Mapping[str, Any], *, require_operating_proof: bool
) -> dict[str, Any]:
    factory = _mapping(packet.get("factory"), field="factory")
    factory_id = _text(factory.get("factory_id"), field="factory.factory_id")
    project_id = _text(factory.get("project_id"), field="factory.project_id")
    effort_id = _text(factory.get("effort_id"), field="factory.effort_id")
    acceptance = _mapping(
        factory.get("b0_c1_c2_acceptance"), field="factory.b0_c1_c2_acceptance"
    )
    if acceptance.get("accepted") is not True:
        raise _error("B0/C1/C2 acceptance is not accepted")
    if tuple(acceptance.get("cycle_labels") or ()) != REQUIRED_CYCLES:
        raise _error("accepted cycle chain must be exactly B0, C1, C2")
    experiment_ids = [
        _text(item, field="factory.b0_c1_c2_acceptance.experiment_ids[]")
        for item in _sequence(
            acceptance.get("experiment_ids"),
            field="factory.b0_c1_c2_acceptance.experiment_ids",
        )
    ]
    run_ids = [
        _text(item, field="factory.b0_c1_c2_acceptance.run_ids[]")
        for item in _sequence(
            acceptance.get("run_ids"), field="factory.b0_c1_c2_acceptance.run_ids"
        )
    ]
    if len(experiment_ids) != 3 or len(set(experiment_ids)) != 3:
        raise _error("B0/C1/C2 experiment IDs must be three unique values")
    if len(run_ids) != 3 or len(set(run_ids)) != 3:
        raise _error("B0/C1/C2 run IDs must be three unique values")
    instance_id = _text(
        acceptance.get("reflexion_instance_id"),
        field="factory.b0_c1_c2_acceptance.reflexion_instance_id",
    )
    instance_head = _git_sha(
        acceptance.get("reflexion_instance_head"),
        field="factory.b0_c1_c2_acceptance.reflexion_instance_head",
    )
    operating_raw = factory.get("operating_proof")
    operating = (
        _mapping(operating_raw, field="factory.operating_proof")
        if operating_raw is not None
        else {}
    )
    if require_operating_proof:
        if operating.get("accepted") is not True:
            raise _error("operating proof is not accepted")
        if int(operating.get("accepted_cycles") or 0) < 12:
            raise _error("operating proof contains fewer than 12 accepted cycles")
        if int(operating.get("window_days") or 0) != 30:
            raise _error("operating proof must use the 30-day window")
        if operating.get("health_status") != "healthy":
            raise _error("Factory health is not healthy")
    return {
        "factory_id": factory_id,
        "project_id": project_id,
        "effort_id": effort_id,
        "instance_id": instance_id,
        "instance_head": instance_head,
        "accepted_cycles": int(operating.get("accepted_cycles") or 0),
        "window_days": int(operating.get("window_days") or 0),
        "experiment_ids": experiment_ids,
        "run_ids": run_ids,
    }


def _experiment_evidence(
    packet: Mapping[str, Any],
    *,
    instance_id: str,
    instance_head: str,
    expected_experiment_ids: Sequence[str],
    expected_run_ids: Sequence[str],
) -> dict[str, Any]:
    experiments = _mapping(packet.get("experiments"), field="experiments")
    bundles = _sequence(experiments.get("bundles"), field="experiments.bundles")
    bundle_by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(bundles):
        bundle = _mapping(raw, field=f"experiments.bundles[{index}]")
        experiment_id = _text(
            bundle.get("experiment_id"),
            field=f"experiments.bundles[{index}].experiment_id",
        )
        if experiment_id in bundle_by_id:
            raise _error(f"duplicate experiment bundle: {experiment_id}")
        bundle_by_id[experiment_id] = bundle
    missing_ids = [
        experiment_id
        for experiment_id in expected_experiment_ids
        if experiment_id not in bundle_by_id
    ]
    if missing_ids:
        raise _error("B0/C1/C2 bundles missing: " + ", ".join(missing_ids))

    observed_origin_run_id = ""
    previous_commit = ""
    source_commits: list[str] = []
    for index, (experiment_id, run_id) in enumerate(
        zip(expected_experiment_ids, expected_run_ids, strict=True)
    ):
        bundle = bundle_by_id[experiment_id]
        experiment = _mapping(
            bundle.get("experiment"),
            field=f"experiments.bundles.{experiment_id}.experiment",
        )
        if experiment.get("status") != "accepted" or str(
            experiment.get("verdict") or ""
        ).lower() not in {"accept", "promote"}:
            raise _error(f"{REQUIRED_CYCLES[index]} experiment is not accepted")
        integrity = _mapping(
            bundle.get("integrity"),
            field=f"experiments.bundles.{experiment_id}.integrity",
        )
        if integrity.get("accepted_cycle") is not True or integrity.get(
            "missing"
        ) not in (None, []):
            raise _error(f"{REQUIRED_CYCLES[index]} bundle is not evidence-complete")
        candidate = _mapping(
            bundle.get("candidate"),
            field=f"experiments.bundles.{experiment_id}.candidate",
        )
        snapshot = _mapping(
            candidate.get("snapshot"),
            field=f"experiments.bundles.{experiment_id}.candidate.snapshot",
        )
        instance = _reflexion_instance(
            snapshot.get("reflexion_instance"),
            field=f"experiments.bundles.{experiment_id}.reflexion_instance",
        )
        if (
            _text(instance.get("instance_id"), field="reflexion_instance.instance_id")
            != instance_id
        ):
            raise _error("experiment bundle replaced the accepted Reflexion instance")
        bundle_run_ids = {
            _text(item, field=f"experiments.bundles.{experiment_id}.run_ids[]")
            for item in _sequence(
                bundle.get("run_ids"),
                field=f"experiments.bundles.{experiment_id}.run_ids",
            )
        }
        if run_id not in bundle_run_ids:
            raise _error(f"{REQUIRED_CYCLES[index]} bundle is not bound to its run")
        origin_run_id = _text(
            instance.get("origin_run_id"),
            field=f"experiments.bundles.{experiment_id}.origin_run_id",
        )
        created_in_run_id = _text(
            instance.get("created_in_run_id"),
            field=f"experiments.bundles.{experiment_id}.created_in_run_id",
        )
        if index == 0:
            if origin_run_id != run_id or created_in_run_id != run_id:
                raise _error("B0 bundle did not originate the Reflexion instance")
            observed_origin_run_id = origin_run_id
        elif (
            origin_run_id != observed_origin_run_id
            or created_in_run_id != observed_origin_run_id
        ):
            raise _error(f"{REQUIRED_CYCLES[index]} changed the instance origin")
        commit_sha = _git_sha(
            instance.get("git_commit_sha"), field="reflexion_instance.git_commit_sha"
        )
        parent_commit = str(instance.get("parent_git_commit_sha") or "").strip().lower()
        if (index == 0 and parent_commit) or (
            index > 0 and parent_commit != previous_commit
        ):
            raise _error(f"{REQUIRED_CYCLES[index]} source parent is not canonical")
        if commit_sha == previous_commit:
            raise _error(f"{REQUIRED_CYCLES[index]} did not advance source")
        provenance = _mapping(
            bundle.get("provenance"),
            field=f"experiments.bundles.{experiment_id}.provenance",
        )
        git_receipt = _mapping(
            provenance.get("git_server"),
            field=f"experiments.bundles.{experiment_id}.provenance.git_server",
        )
        if (
            _git_sha(
                git_receipt.get("source_commit_sha"),
                field=f"experiments.bundles.{experiment_id}.git_server.source_commit_sha",
            )
            != commit_sha
            or git_receipt.get("repo_state_advanced") is not True
        ):
            raise _error(f"{REQUIRED_CYCLES[index]} git receipt is not source-bound")
        if index == 0:
            if provenance.get("prior_findings_consumed") not in (None, {}):
                raise _error("B0 bundle cannot consume prior experiment findings")
        else:
            prior = _mapping(
                provenance.get("prior_findings_consumed"),
                field=f"experiments.bundles.{experiment_id}.prior_findings_consumed",
            )
            if prior.get("source_experiment_id") != expected_experiment_ids[index - 1]:
                raise _error(
                    f"{REQUIRED_CYCLES[index]} did not consume the prior experiment"
                )
        previous_commit = commit_sha
        source_commits.append(commit_sha)
    if previous_commit != instance_head:
        raise _error(
            "latest experiment source commit is not the accepted instance head"
        )
    comparison = _mapping(experiments.get("comparison"), field="experiments.comparison")
    if comparison.get("comparable") is not True:
        raise _error("experiment history is not comparable")
    if not set(expected_experiment_ids).issubset(
        set(comparison.get("experiment_ids") or [])
    ):
        raise _error("experiment comparison omits the B0/C1/C2 chain")
    return {
        "experiment_ids": list(expected_experiment_ids),
        "run_ids": list(expected_run_ids),
        "source_commits": source_commits,
        "bundle_count": 3,
    }


def _knowledge_and_git(
    packet: Mapping[str, Any],
    *,
    expected_experiment_ids: Sequence[str],
    expected_run_ids: Sequence[str],
    expected_source_commits: Sequence[str],
) -> dict[str, Any]:
    wiki = _sequence(packet.get("wiki_receipts"), field="wiki_receipts")
    wiki_by_experiment: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(wiki):
        receipt = _mapping(raw, field=f"wiki_receipts[{index}]")
        experiment_id = _text(
            receipt.get("source_experiment_id"),
            field=f"wiki_receipts[{index}].source_experiment_id",
        )
        if experiment_id in wiki_by_experiment:
            raise _error(f"duplicate Wiki receipt: {experiment_id}")
        wiki_by_experiment[experiment_id] = receipt
    wiki_urls: list[str] = []
    for index, (experiment_id, run_id) in enumerate(
        zip(expected_experiment_ids, expected_run_ids, strict=True)
    ):
        receipt = _mapping(
            wiki_by_experiment.get(experiment_id),
            field=f"wiki_receipts.{experiment_id}",
        )
        if str(receipt.get("review_state") or "").lower() not in {
            "accepted",
            "applied",
        }:
            raise _error(f"Wiki receipt {index} is not accepted/applied truth")
        if receipt.get("source_run_id") != run_id:
            raise _error(f"Wiki receipt {index} is not bound to its B0/C1/C2 run")
        wiki_urls.append(_text(receipt.get("url"), field=f"wiki_receipts[{index}].url"))

    git_receipts = _sequence(packet.get("git_receipts"), field="git_receipts")
    git_by_run: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(git_receipts):
        receipt = _mapping(raw, field=f"git_receipts[{index}]")
        run_id = _text(
            receipt.get("source_run_id"), field=f"git_receipts[{index}].source_run_id"
        )
        if run_id in git_by_run:
            raise _error(f"duplicate git-server receipt: {run_id}")
        git_by_run[run_id] = receipt
    ordered_git: list[dict[str, Any]] = []
    remote_repo = ""
    branch = ""
    for index, (run_id, source_commit) in enumerate(
        zip(expected_run_ids, expected_source_commits, strict=True)
    ):
        receipt = _mapping(git_by_run.get(run_id), field=f"git_receipts.{run_id}")
        if (
            receipt.get("repo_state_advanced") is not True
            or _git_sha(
                receipt.get("source_commit_sha"),
                field=f"git_receipts.{run_id}.source_commit_sha",
            )
            != source_commit
        ):
            raise _error(f"git-server receipt {index} is not source-bound")
        current_remote = _text(
            receipt.get("remote_repo"), field=f"git_receipts.{run_id}.remote_repo"
        )
        current_branch = _text(
            receipt.get("branch"), field=f"git_receipts.{run_id}.branch"
        )
        if index == 0:
            remote_repo = current_remote
            branch = current_branch
        elif current_remote != remote_repo or current_branch != branch:
            raise _error("B0/C1/C2 git receipts cross a repo or branch boundary")
        ordered_git.append(receipt)
    last = ordered_git[-1]
    evidence_commit = _git_sha(
        last.get("commit_sha"), field="git_receipts[-1].commit_sha"
    )
    return {
        "wiki_urls": wiki_urls,
        "remote_repo": remote_repo,
        "branch": branch,
        "evidence_commit": evidence_commit,
    }


def _release_and_artifacts(
    packet: Mapping[str, Any],
    *,
    instance_id: str,
    instance_head: str,
    cycle_experiment_ids: Sequence[str],
    cycle_run_ids: Sequence[str],
    remote_repo: str,
    branch: str,
) -> dict[str, Any]:
    audit = _mapping(packet.get("release_audit"), field="release_audit")
    if (
        audit.get("accepted") is not True
        or audit.get("consumed") is not True
        or int(audit.get("use_index") or 0) != 1
        or audit.get("seed_role") != "release_audit"
        or int(audit.get("seed_count") or 0) != 64
    ):
        raise _error("one accepted 64-seed release audit is required")
    expected_audit_identity = {
        "reflexion_instance_id": instance_id,
        "source_commit_sha": instance_head,
    }
    if any(audit.get(key) != value for key, value in expected_audit_identity.items()):
        raise _error("release audit is not bound to the accepted C2 instance")
    try:
        statistics = {
            field: float(audit.get(field)) for field in ("mean_delta", "ci_lo", "ci_hi")
        }
    except (TypeError, ValueError) as exc:
        raise _error("release audit statistics must be numeric") from exc
    if any(not math.isfinite(value) for value in statistics.values()):
        raise _error("release audit statistics must be finite")
    if statistics["ci_lo"] <= 0.0:
        raise _error("release audit lower confidence bound must be positive")
    audit_experiment_id = _text(
        audit.get("experiment_id"), field="release_audit.experiment_id"
    )
    audit_run_id = _text(audit.get("run_id"), field="release_audit.run_id")
    if audit_experiment_id in set(cycle_experiment_ids) or audit_run_id in set(
        cycle_run_ids
    ):
        raise _error("release audit must be distinct from the B0/C1/C2 chain")
    experiments = _mapping(packet.get("experiments"), field="experiments")
    bundles = _sequence(experiments.get("bundles"), field="experiments.bundles")
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
    audit_bundle_run_ids = {
        _text(item, field="release_audit.bundle.run_ids[]")
        for item in _sequence(
            audit_bundle.get("run_ids"), field="release_audit.bundle.run_ids"
        )
    }
    if audit_run_id not in audit_bundle_run_ids:
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
        _text(
            audit_instance.get("instance_id"),
            field="release_audit.bundle.reflexion_instance.instance_id",
        )
        != instance_id
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
        or evaluation_delta != statistics["mean_delta"]
        or not audit_evaluation.get("result_id")
        or not audit_evaluation.get("scorer_id")
        or not audit_evaluation.get("per_example_artifact_id")
        or not audit_evaluation.get("summary_artifact_id")
    ):
        raise _error("release audit evaluation is not 64-seed release evidence")
    audit_evaluation_metadata = _mapping(
        audit_evaluation.get("metadata"),
        field="release_audit.bundle.evaluation.metadata",
    )
    embedded_audit = _mapping(
        audit_evaluation_metadata.get("release_audit"),
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
    for field, value in statistics.items():
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
        or _text(
            audit_git.get("remote_repo"),
            field="release_audit.bundle.git_server.remote_repo",
        )
        != remote_repo
        or _text(
            audit_git.get("branch"), field="release_audit.bundle.git_server.branch"
        )
        != branch
    ):
        raise _error("release audit git receipt is not bound to the accepted lineage")
    audit_evidence_commit = _git_sha(
        audit_git.get("commit_sha"),
        field="release_audit.bundle.git_server.commit_sha",
    )
    if audit_evidence_commit == instance_head:
        raise _error("release audit evidence commit cannot equal the source commit")
    top_level_git_receipts = _sequence(packet.get("git_receipts"), field="git_receipts")
    matching_audit_git = [
        _mapping(raw, field=f"git_receipts[{index}]")
        for index, raw in enumerate(top_level_git_receipts)
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
    artifacts = _mapping(packet.get("artifacts"), field="artifacts")
    hosted = _mapping(artifacts.get("hosted_site"), field="artifacts.hosted_site")
    hosted_id = _text(
        hosted.get("hosted_artifact_id"),
        field="artifacts.hosted_site.hosted_artifact_id",
    )
    hosted_url = _text(
        hosted.get("hosted_url"), field="artifacts.hosted_site.hosted_url"
    )
    _sha256(hosted.get("sha256"), field="artifacts.hosted_site.sha256")
    if hosted.get("visibility") != "org":
        raise _error("hosted Artifact Site must be org-visible")
    if RELEASE_AUDIT_SPLIT not in list(hosted.get("splits_cited") or []):
        raise _error("hosted Artifact Site does not cite the release-audit split")
    return {
        **statistics,
        "audit_experiment_id": audit_experiment_id,
        "audit_run_id": audit_run_id,
        "scorecard_digest": scorecard_digest,
        "evidence_commit": audit_evidence_commit,
        "hosted_artifact_id": hosted_id,
        "hosted_url": hosted_url,
    }


def _infrastructure(
    packet: Mapping[str, Any],
    *,
    source_manifest_digest: str,
    runtime_image_digest: str,
    project_id: str,
    instance_id: str,
    instance_head: str,
    evidence_commit: str,
    remote_repo: str,
    branch: str,
    cycle_run_ids: Sequence[str],
    cycle_source_commits: Sequence[str],
) -> dict[str, Any]:
    infra = _mapping(packet.get("infrastructure"), field="infrastructure")
    railway = _mapping(infra.get("railway"), field="infrastructure.railway")
    railway_ids: dict[str, str] = {}
    for environment in REQUIRED_RAILWAY_ENVIRONMENTS:
        deployment = _mapping(
            railway.get(environment), field=f"infrastructure.railway.{environment}"
        )
        if deployment.get("health_status") != "healthy":
            raise _error(f"Railway {environment} is not healthy")
        if (
            _sha256(
                deployment.get("source_manifest_digest"),
                field=f"infrastructure.railway.{environment}.source_manifest_digest",
            )
            != source_manifest_digest
        ):
            raise _error(f"Railway {environment} uses a different source manifest")
        _git_sha(
            deployment.get("git_sha"),
            field=f"infrastructure.railway.{environment}.git_sha",
        )
        railway_ids[environment] = _text(
            deployment.get("deployment_id"),
            field=f"infrastructure.railway.{environment}.deployment_id",
        )

    daytona = _sequence(infra.get("daytona_runs"), field="infrastructure.daytona_runs")
    daytona_by_label: dict[str, dict[str, Any]] = {}
    for raw in daytona:
        item = _mapping(raw, field="infrastructure.daytona_runs[]")
        label = _text(item.get("cycle_label"), field="daytona_runs[].cycle_label")
        if label in daytona_by_label:
            raise _error(f"duplicate Daytona cycle receipt: {label}")
        daytona_by_label[label] = item
    for index, label in enumerate(REQUIRED_CYCLES):
        item = _mapping(
            daytona_by_label.get(label), field=f"infrastructure.daytona_runs.{label}"
        )
        if (
            item.get("terminal_state") != "done"
            or item.get("cleanup_passed") is not True
            or not item.get("sandbox_id")
            or item.get("run_id") != cycle_run_ids[index]
            or _sha256(
                item.get("runtime_image"),
                field=f"infrastructure.daytona_runs.{label}.runtime_image",
            )
            != runtime_image_digest
            or _git_sha(
                item.get("source_commit_sha"),
                field=f"infrastructure.daytona_runs.{label}.source_commit_sha",
            )
            != cycle_source_commits[index]
        ):
            raise _error(f"Daytona {label} receipt is not runtime/source bound")

    champion_receipt = _mapping(infra.get("exe_dev"), field="infrastructure.exe_dev")
    if champion_receipt.get("schema_version") != (
        "research_factory_reflexion.champion_deployment_receipt.v1"
    ):
        raise _error("exe.dev champion deployment receipt schema mismatch")
    if champion_receipt.get("dry_run") is not False:
        raise _error("exe.dev champion evidence is a dry-run")
    requested = _mapping(
        champion_receipt.get("request"),
        field="infrastructure.exe_dev.request",
    )
    deployment = _mapping(
        champion_receipt.get("deployment"),
        field="infrastructure.exe_dev.deployment",
    )
    exe = _mapping(
        champion_receipt.get("observation"),
        field="infrastructure.exe_dev.observation",
    )
    if exe.get("deployment_id") != deployment.get("deployment_id"):
        raise _error("exe.dev deployment and observation IDs differ")
    if exe.get("project_id") != project_id:
        raise _error("exe.dev champion belongs to a different project")
    if (
        exe.get("host_kind") != "exe_dev"
        or exe.get("state") != "running"
        or exe.get("topology_id") != "research-reflexion-service"
        or exe.get("topology_version") != "2026-07-10.v1"
    ):
        raise _error("exe.dev champion CloudDeployment is not running")
    health = _mapping(exe.get("health"), field="infrastructure.exe_dev.health")
    http = _mapping(
        health.get("http") or {}, field="infrastructure.exe_dev.health.http"
    )
    if (
        health.get("observed") is not True
        or http.get("ok") is not True
        or http.get("identity_ok") is not True
        or http.get("status") != "healthy"
        or http.get("instance_id") != instance_id
        or http.get("source_commit_sha") != instance_head
    ):
        raise _error("exe.dev champion service is not healthy")
    provision_receipts = _mapping(
        exe.get("provision_receipts"),
        field="infrastructure.exe_dev.observation.provision_receipts",
    )
    deploy_steps = _sequence(
        provision_receipts.get("deploy_steps"),
        field="infrastructure.exe_dev.observation.provision_receipts.deploy_steps",
    )
    steps_by_id = {
        str(step.get("step_id") or ""): dict(step)
        for step in deploy_steps
        if isinstance(step, Mapping)
    }
    for required_step in ("bootstrap:project_git", "deploy_reflexion_service"):
        step = _mapping(
            steps_by_id.get(required_step),
            field=f"infrastructure.exe_dev.deploy_steps.{required_step}",
        )
        if step.get("exit_code") != 0:
            raise _error(f"exe.dev deployment step failed: {required_step}")
    request_payload = _mapping(
        exe.get("request_payload"),
        field="infrastructure.exe_dev.observation.request_payload",
    )
    source_request = _mapping(
        requested.get("source"), field="infrastructure.exe_dev.request.source"
    )
    server_source_request = _mapping(
        request_payload.get("source"),
        field="infrastructure.exe_dev.observation.request_payload.source",
    )
    resolved_source = _mapping(
        request_payload.get("resolved_source"),
        field="infrastructure.exe_dev.observation.request_payload.resolved_source",
    )
    expected_request_source = {
        "kind": "project_git",
        "source_commit_sha": instance_head,
        "evidence_commit_sha": evidence_commit,
        "instance_id": instance_id,
    }
    if source_request != expected_request_source:
        raise _error("exe.dev caller source binding is not the accepted instance")
    if server_source_request != expected_request_source:
        raise _error("exe.dev server source request differs from the accepted instance")
    expected_resolved_source_keys = {
        "kind",
        "repo_id",
        "remote_repo",
        "branch",
        "source_commit_sha",
        "evidence_commit_sha",
        "instance_id",
        "checkout_name",
        "subdirectory",
    }
    if set(resolved_source) != expected_resolved_source_keys:
        raise _error("exe.dev resolved source binding fields are not canonical")
    expected_resolved_identity = {
        "kind": "smr_project_git",
        "source_commit_sha": instance_head,
        "evidence_commit_sha": evidence_commit,
        "instance_id": instance_id,
        "checkout_name": "reflexion-source",
        "subdirectory": "reflexion_instance",
    }
    if any(
        resolved_source.get(key) != value
        for key, value in expected_resolved_identity.items()
    ):
        raise _error("exe.dev champion service is not the accepted instance head")
    _text(
        resolved_source.get("repo_id"),
        field="infrastructure.exe_dev.resolved_source.repo_id",
    )
    if (
        _text(
            resolved_source.get("remote_repo"),
            field="infrastructure.exe_dev.resolved_source.remote_repo",
        )
        != remote_repo
        or _text(
            resolved_source.get("branch"),
            field="infrastructure.exe_dev.resolved_source.branch",
        )
        != branch
    ):
        raise _error("exe.dev resolved source crosses the accepted git lineage")
    return {
        "railway_deployment_ids": railway_ids,
        "daytona_cycle_labels": list(REQUIRED_CYCLES),
        "exe_deployment_id": _text(
            exe.get("deployment_id"), field="infrastructure.exe_dev.deployment_id"
        ),
        "exe_service_url": _text(
            exe.get("service_url"), field="infrastructure.exe_dev.service_url"
        ),
        "exe_source_commit_sha": instance_head,
        "exe_evidence_commit_sha": evidence_commit,
    }


def validate_predeployment_evidence(
    packet: Mapping[str, Any], *, require_operating_proof: bool = True
) -> dict[str, Any]:
    source_packet = dict(packet)
    if source_packet.get("schema_version") != PACKET_SCHEMA:
        raise _error(f"schema_version must be {PACKET_SCHEMA}")
    source = _source_evidence(source_packet)
    factory = _factory_evidence(
        source_packet, require_operating_proof=require_operating_proof
    )
    experiments = _experiment_evidence(
        source_packet,
        instance_id=factory["instance_id"],
        instance_head=factory["instance_head"],
        expected_experiment_ids=factory["experiment_ids"],
        expected_run_ids=factory["run_ids"],
    )
    knowledge_git = _knowledge_and_git(
        source_packet,
        expected_experiment_ids=experiments["experiment_ids"],
        expected_run_ids=experiments["run_ids"],
        expected_source_commits=experiments["source_commits"],
    )
    release = _release_and_artifacts(
        source_packet,
        instance_id=factory["instance_id"],
        instance_head=factory["instance_head"],
        cycle_experiment_ids=experiments["experiment_ids"],
        cycle_run_ids=experiments["run_ids"],
        remote_repo=knowledge_git["remote_repo"],
        branch=knowledge_git["branch"],
    )
    knowledge_git["cycle_evidence_commit"] = knowledge_git["evidence_commit"]
    knowledge_git["evidence_commit"] = release["evidence_commit"]
    return {
        "source": source,
        "factory": factory,
        "experiments": experiments,
        "knowledge_git": knowledge_git,
        "release": release,
    }


def validate_release_evidence(packet: Mapping[str, Any]) -> dict[str, Any]:
    source_packet = dict(packet)
    predeployment = validate_predeployment_evidence(source_packet)
    source = _mapping(predeployment.get("source"), field="predeployment.source")
    factory = _mapping(predeployment.get("factory"), field="predeployment.factory")
    experiments = _mapping(
        predeployment.get("experiments"), field="predeployment.experiments"
    )
    knowledge_git = _mapping(
        predeployment.get("knowledge_git"), field="predeployment.knowledge_git"
    )
    release = _mapping(predeployment.get("release"), field="predeployment.release")
    infrastructure = _infrastructure(
        source_packet,
        source_manifest_digest=source["manifest_digest"],
        runtime_image_digest=source["runtime_image_digest"],
        project_id=factory["project_id"],
        instance_id=factory["instance_id"],
        instance_head=factory["instance_head"],
        evidence_commit=release["evidence_commit"],
        remote_repo=knowledge_git["remote_repo"],
        branch=knowledge_git["branch"],
        cycle_run_ids=experiments["run_ids"],
        cycle_source_commits=experiments["source_commits"],
    )
    return {
        "schema_version": RECEIPT_SCHEMA,
        "accepted": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "factory": factory,
        "experiments": experiments,
        "knowledge_git": knowledge_git,
        "release": release,
        "infrastructure": infrastructure,
    }


def render_blog(receipt: Mapping[str, Any]) -> str:
    source = _mapping(receipt.get("source"), field="receipt.source")
    factory = _mapping(receipt.get("factory"), field="receipt.factory")
    experiments = _mapping(receipt.get("experiments"), field="receipt.experiments")
    knowledge_git = _mapping(
        receipt.get("knowledge_git"), field="receipt.knowledge_git"
    )
    release = _mapping(receipt.get("release"), field="receipt.release")
    infrastructure = _mapping(
        receipt.get("infrastructure"), field="receipt.infrastructure"
    )
    railway = _mapping(
        infrastructure.get("railway_deployment_ids"),
        field="receipt.infrastructure.railway_deployment_ids",
    )
    return f"""# Building a Reflexion System with a 24/7 Research Factory

We ran one Research Factory with one continuous objective: build a Reflexion
implementation from scratch in project git, then improve that same instance.
The reference cookbook was never mounted or copied into the worker workspace.

The runtime used immutable image `{source["runtime_image_digest"]}` built from
source manifest `{source["manifest_digest"]}`. The release receipt records the
full clean commit SHA for each of the five runtime authorities.

The Factory preserved instance `{factory["instance_id"]}` through B0, C1, and
C2, ending at source commit `{factory["instance_head"]}`. It recorded
{experiments["bundle_count"]} evidence-complete experiments and then accumulated
{factory["accepted_cycles"]} accepted cycles in a {factory["window_days"]}-day
operating window.

The one-use 64-seed release audit measured a paired mean delta of
{release["mean_delta"]:.6f}, with a 95% interval from {release["ci_lo"]:.6f} to
{release["ci_hi"]:.6f}. The lower bound stayed above zero.

Every experiment exposes the candidate prompt/model/config, per-seed scorecard,
trace and cost indexes, Seraph judgment, Gardener carry, Synth Wiki record, and
project git receipt. The final evidence commit is
`{knowledge_git["evidence_commit"]}` on `{knowledge_git["remote_repo"]}`.

The same source manifest ran through Railway dev `{railway["dev"]}`, staging
`{railway["staging"]}`, and production `{railway["prod"]}`. Daytona supplied the
disposable B0/C1/C2 workers with terminal cleanup; exe.dev deployment
`{infrastructure["exe_deployment_id"]}` serves the accepted instance at
{infrastructure["exe_service_url"]}.

Explore the release-audited Artifact Site: {release["hosted_url"]}.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blog-output", type=Path)
    args = parser.parse_args()
    payload = _mapping(
        json.loads(args.evidence.read_text(encoding="utf-8")), field=str(args.evidence)
    )
    receipt = validate_release_evidence(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.blog_output:
        args.blog_output.parent.mkdir(parents=True, exist_ok=True)
        args.blog_output.write_text(render_blog(receipt), encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["render_blog", "validate_release_evidence"]
