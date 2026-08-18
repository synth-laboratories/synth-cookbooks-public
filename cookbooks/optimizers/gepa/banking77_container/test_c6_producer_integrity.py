"""C-6 producer integrity helpers. Does not import the live HuggingFace dataset."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).with_name("producer_integrity.py")
_SPEC = importlib.util.spec_from_file_location("banking77_producer_integrity", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
integrity = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(integrity)

ROWS = [
    {"seed": 0, "source_index": 10, "split": "train", "text": "I lost my card", "label": "lost_or_stolen_card"},
    {"seed": 1, "source_index": 11, "split": "train", "text": "change pin", "label": "change_pin"},
    {"seed": 0, "source_index": 90, "split": "test", "text": "where is my salary", "label": "receiving_money"},
]


def test_literal_training_targets_default_forbid(monkeypatch) -> None:
    monkeypatch.delenv("BANKING77_LITERAL_TRAINING_TARGETS", raising=False)
    assert integrity.literal_training_targets_policy() == "forbid"
    monkeypatch.setenv("BANKING77_LITERAL_TRAINING_TARGETS", "allow")
    assert integrity.literal_training_targets_policy() == "allow"
    assert integrity.literal_training_targets_policy("forbid") == "forbid"


def test_split_identity_and_per_example_digests() -> None:
    identity = integrity.split_identity(
        ROWS, train_seed=1009, test_seed=2003, train_sample=24, test_sample=200
    )
    assert identity["dataset_id"] == "banking77_public_rows"
    assert identity["dataset_digest"].startswith("sha256:")
    assert identity["train_seed"] == 1009
    assert identity["test_seed"] == 2003
    assert identity["samples"]["train"] == 2
    assert identity["samples"]["test"] == 1
    assert identity["sampling_mode"] == "fixed"
    assert integrity.dataset_digest(ROWS) == identity["dataset_digest"]
    first = integrity.example_span_digest(ROWS[0])
    again = integrity.example_span_digest(ROWS[0])
    other = integrity.example_span_digest(ROWS[1])
    assert first == again
    assert first != other
    protected = integrity.span_digests_for_split(ROWS, split="test")
    assert list(protected) == ["test:0"]
    assert protected["test:0"].startswith("sha256:")


def test_leakage_and_execution_blocks() -> None:
    leakage = integrity.leakage_contract()
    assert leakage == {
        "policy": "forbid",
        "protected_split": "test",
        "span_digest_route": "/leakage/span_digests",
    }
    execution = integrity.execution_block(policy_concurrency=30, timeout=20.0, retries=1)
    assert execution["policy_concurrency"] == 30
    assert execution["timeout"] == 20.0
    assert execution["retries"] == 1
