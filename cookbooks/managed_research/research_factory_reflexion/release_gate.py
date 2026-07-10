"""Fail closed unless a Reflexion Factory is adoptable, publishable, and 24/7-proven."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKET_SCHEMA = "research_factory_reflexion.release_evidence.v1"
RECEIPT_SCHEMA = "research_factory_reflexion.release_receipt.v1"
REQUIRED_SOURCE_COMPONENTS = {"backend", "evals", "synth-dev", "synth-ai", "gamebench"}
REQUIRED_CYCLES = ("B0", "C1", "C2")
REQUIRED_RAILWAY_ENVIRONMENTS = ("dev", "staging", "prod")


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


def _factory_evidence(packet: Mapping[str, Any]) -> dict[str, Any]:
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
    instance_id = _text(
        acceptance.get("reflexion_instance_id"),
        field="factory.b0_c1_c2_acceptance.reflexion_instance_id",
    )
    instance_head = _git_sha(
        acceptance.get("reflexion_instance_head"),
        field="factory.b0_c1_c2_acceptance.reflexion_instance_head",
    )
    operating = _mapping(
        factory.get("operating_proof"), field="factory.operating_proof"
    )
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
        "accepted_cycles": int(operating["accepted_cycles"]),
        "window_days": int(operating["window_days"]),
    }


def _experiment_evidence(
    packet: Mapping[str, Any], *, instance_id: str, instance_head: str
) -> dict[str, Any]:
    experiments = _mapping(packet.get("experiments"), field="experiments")
    bundles = _sequence(experiments.get("bundles"), field="experiments.bundles")
    if len(bundles) < 3:
        raise _error("at least B0, C1, and C2 experiment bundles are required")
    accepted_ids: list[str] = []
    observed_head = ""
    for index, raw in enumerate(bundles):
        bundle = _mapping(raw, field=f"experiments.bundles[{index}]")
        integrity = _mapping(
            bundle.get("integrity"), field=f"experiments.bundles[{index}].integrity"
        )
        if integrity.get("accepted_cycle") is not True or integrity.get(
            "missing"
        ) not in (
            None,
            [],
        ):
            raise _error(f"experiment bundle {index} is not evidence-complete")
        candidate = _mapping(
            bundle.get("candidate"), field=f"experiments.bundles[{index}].candidate"
        )
        snapshot = _mapping(
            candidate.get("snapshot"),
            field=f"experiments.bundles[{index}].candidate.snapshot",
        )
        instance = _mapping(
            snapshot.get("reflexion_instance"),
            field=f"experiments.bundles[{index}].candidate.snapshot.reflexion_instance",
        )
        if (
            _text(instance.get("instance_id"), field="reflexion_instance.instance_id")
            != instance_id
        ):
            raise _error("experiment bundle replaced the accepted Reflexion instance")
        observed_head = _git_sha(
            instance.get("git_commit_sha"), field="reflexion_instance.git_commit_sha"
        )
        accepted_ids.append(
            _text(
                bundle.get("experiment_id"),
                field=f"experiments.bundles[{index}].experiment_id",
            )
        )
    if observed_head != instance_head:
        raise _error(
            "latest experiment source commit is not the accepted instance head"
        )
    comparison = _mapping(experiments.get("comparison"), field="experiments.comparison")
    if comparison.get("comparable") is not True:
        raise _error("experiment history is not comparable")
    return {"experiment_ids": accepted_ids, "bundle_count": len(bundles)}


def _knowledge_and_git(
    packet: Mapping[str, Any], *, instance_head: str
) -> dict[str, Any]:
    wiki = _sequence(packet.get("wiki_receipts"), field="wiki_receipts")
    if len(wiki) < 3:
        raise _error("B0/C1/C2 Wiki receipts are required")
    wiki_urls: list[str] = []
    for index, raw in enumerate(wiki):
        receipt = _mapping(raw, field=f"wiki_receipts[{index}]")
        if str(receipt.get("review_state") or "").lower() not in {
            "accepted",
            "applied",
        }:
            raise _error(f"Wiki receipt {index} is not accepted/applied truth")
        wiki_urls.append(_text(receipt.get("url"), field=f"wiki_receipts[{index}].url"))

    git_receipts = _sequence(packet.get("git_receipts"), field="git_receipts")
    if len(git_receipts) < 3:
        raise _error("B0/C1/C2 git-server receipts are required")
    last = _mapping(git_receipts[-1], field="git_receipts[-1]")
    if last.get("repo_state_advanced") is not True:
        raise _error("latest project git-server receipt did not advance")
    if (
        _git_sha(
            last.get("source_commit_sha"), field="git_receipts[-1].source_commit_sha"
        )
        != instance_head
    ):
        raise _error("latest git-server receipt does not contain the instance head")
    evidence_commit = _git_sha(
        last.get("commit_sha"), field="git_receipts[-1].commit_sha"
    )
    return {
        "wiki_urls": wiki_urls,
        "remote_repo": _text(
            last.get("remote_repo"), field="git_receipts[-1].remote_repo"
        ),
        "branch": _text(last.get("branch"), field="git_receipts[-1].branch"),
        "evidence_commit": evidence_commit,
    }


def _release_and_artifacts(packet: Mapping[str, Any]) -> dict[str, Any]:
    audit = _mapping(packet.get("release_audit"), field="release_audit")
    if audit.get("accepted") is not True or int(audit.get("seed_count") or 0) != 64:
        raise _error("one accepted 64-seed release audit is required")
    ci_lo = float(audit.get("ci_lo") or 0.0)
    if ci_lo <= 0.0:
        raise _error("release audit lower confidence bound must be positive")
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
    if "craftax_release_audit_v1_64" not in list(hosted.get("splits_cited") or []):
        raise _error("hosted Artifact Site does not cite the release-audit split")
    return {
        "mean_delta": float(audit.get("mean_delta") or 0.0),
        "ci_lo": ci_lo,
        "ci_hi": float(audit.get("ci_hi") or 0.0),
        "hosted_artifact_id": hosted_id,
        "hosted_url": hosted_url,
    }


def _infrastructure(
    packet: Mapping[str, Any], *, source_manifest_digest: str, instance_head: str
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
    labels = {
        _text(item.get("cycle_label"), field="daytona_runs[].cycle_label")
        for item in daytona
        if isinstance(item, Mapping)
        and item.get("terminal_state") == "done"
        and item.get("cleanup_passed") is True
        and item.get("sandbox_id")
    }
    if not set(REQUIRED_CYCLES).issubset(labels):
        raise _error("Daytona does not prove terminal cleanup for B0, C1, and C2")

    exe = _mapping(infra.get("exe_dev"), field="infrastructure.exe_dev")
    if exe.get("host_kind") != "exe_dev" or exe.get("state") != "running":
        raise _error("exe.dev champion CloudDeployment is not running")
    health = _mapping(exe.get("health"), field="infrastructure.exe_dev.health")
    if health.get("status") != "healthy":
        raise _error("exe.dev champion service is not healthy")
    if (
        _git_sha(
            exe.get("source_commit_sha"),
            field="infrastructure.exe_dev.source_commit_sha",
        )
        != instance_head
    ):
        raise _error("exe.dev champion service is not the accepted instance head")
    return {
        "railway_deployment_ids": railway_ids,
        "daytona_cycle_labels": sorted(labels),
        "exe_deployment_id": _text(
            exe.get("deployment_id"), field="infrastructure.exe_dev.deployment_id"
        ),
        "exe_service_url": _text(
            exe.get("service_url"), field="infrastructure.exe_dev.service_url"
        ),
    }


def validate_release_evidence(packet: Mapping[str, Any]) -> dict[str, Any]:
    source_packet = dict(packet)
    if source_packet.get("schema_version") != PACKET_SCHEMA:
        raise _error(f"schema_version must be {PACKET_SCHEMA}")
    source = _source_evidence(source_packet)
    factory = _factory_evidence(source_packet)
    experiments = _experiment_evidence(
        source_packet,
        instance_id=factory["instance_id"],
        instance_head=factory["instance_head"],
    )
    knowledge_git = _knowledge_and_git(
        source_packet, instance_head=factory["instance_head"]
    )
    release = _release_and_artifacts(source_packet)
    infrastructure = _infrastructure(
        source_packet,
        source_manifest_digest=source["manifest_digest"],
        instance_head=factory["instance_head"],
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
