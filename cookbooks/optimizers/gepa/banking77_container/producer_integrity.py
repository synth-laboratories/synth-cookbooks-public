"""C-6 Banking77 producer integrity: leakage, split identity, execution, forbid-by-default."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Iterable, Mapping

DATASET_ID = "banking77_public_rows"
PROTECTED_SPLIT = "test"
SPAN_DIGEST_ROUTE = "/leakage/span_digests"


def literal_training_targets_policy(raw: str | None = None) -> str:
    """Default forbid. Explicit opt-in via recipe-materialized config or env."""
    text = str(
        raw if raw is not None else os.environ.get("BANKING77_LITERAL_TRAINING_TARGETS", "forbid")
    ).strip().lower().replace("-", "_")
    if text in {"allow", "allowed", "permit"}:
        return "allow"
    return "forbid"


def example_span_digest(row: Mapping[str, Any]) -> str:
    blob = json.dumps(
        {
            "split": row.get("split"),
            "seed": row.get("seed"),
            "source_index": row.get("source_index"),
            "text": row.get("text"),
            "label": row.get("label"),
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def dataset_digest(rows: Iterable[Mapping[str, Any]]) -> str:
    payload = [example_span_digest(row) for row in rows]
    blob = json.dumps(payload, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def split_identity(
    rows: list[Mapping[str, Any]],
    *,
    train_seed: int,
    test_seed: int,
    train_sample: int,
    test_sample: int,
) -> dict[str, Any]:
    train = sum(1 for row in rows if row.get("split") == "train")
    test = sum(1 for row in rows if row.get("split") == "test")
    return {
        "dataset_id": DATASET_ID,
        "dataset_digest": dataset_digest(rows),
        "train_seed": int(train_seed),
        "test_seed": int(test_seed),
        "samples": {"train": train, "test": test, "train_requested": train_sample, "test_requested": test_sample},
        "sampling_mode": "fixed",
    }


def leakage_contract(*, policy: str | None = None) -> dict[str, Any]:
    resolved = literal_training_targets_policy(policy)
    return {
        "policy": resolved,
        "protected_split": PROTECTED_SPLIT,
        "span_digest_route": SPAN_DIGEST_ROUTE,
    }


def execution_block(*, policy_concurrency: int, timeout: float, retries: int) -> dict[str, Any]:
    return {
        "policy_concurrency": int(policy_concurrency),
        "timeout": float(timeout),
        "retries": int(retries),
    }


def span_digests_for_split(rows: Iterable[Mapping[str, Any]], *, split: str) -> dict[str, str]:
    digests: dict[str, str] = {}
    for row in rows:
        if str(row.get("split")) != split:
            continue
        example_id = str(row.get("example_id") or f"{row.get('split')}:{row.get('seed')}")
        digests[example_id] = example_span_digest(row)
    return digests
