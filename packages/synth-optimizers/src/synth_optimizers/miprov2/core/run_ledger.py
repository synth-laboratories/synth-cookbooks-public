"""SQLite-backed run ledger + evidence/read-model helpers for MIPROv2."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from synth_optimizers.miprov2.transform_contract import (
    TransformExecutionSummary,
    TransformFailure,
)
from synth_optimizers.miprov2.core.contracts import (
    MiproModelTransformRecord,
    MiproRunEvent,
)
from synth_optimizers.miprov2.core.optimizer import MiproTrialResult
from synth_optimizers.miprov2.core.program_model import MiproProgramCandidate

_DEFAULT_RUN_ROOT = Path(".out") / "miprov2" / "runs"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _json_loads(value: str) -> Any:
    return json.loads(value)


def _safe_path_component(value: str) -> str:
    text = "".join(
        ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in str(value)
    ).strip("._")
    return text or "item"


def _hash_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _json_field(raw: Any, *, default: Any) -> Any:
    if raw in (None, ""):
        return default
    try:
        return _json_loads(str(raw))
    except Exception:
        return default


@dataclass(slots=True, frozen=True)
class LedgerObservation:
    seq: int
    round_idx: int
    trial: MiproTrialResult


@dataclass(slots=True, frozen=True)
class LedgerHeldoutSnapshot:
    seq: int
    snapshot: RunLedgerHeldoutSnapshot


@dataclass(slots=True, frozen=True)
class RunLedgerHeldoutSnapshot:
    round_idx: int
    best_candidate_id: str
    best_score: float
    baseline_score: float | None
    lift: float | None


@dataclass(slots=True, frozen=True)
class LedgerProposerRound:
    seq: int
    round_idx: int
    summary: dict[str, Any]
    diagnostics: dict[str, Any]
    stop_reason: str
    skipped_tabu_delta: int


@dataclass(slots=True, frozen=True)
class MiproRunResumeState:
    run_id: str
    ledger_path: str
    observations: list[LedgerObservation] = field(default_factory=list)
    heldout_snapshots: list[LedgerHeldoutSnapshot] = field(default_factory=list)
    proposer_rounds: list[LedgerProposerRound] = field(default_factory=list)
    run_state: dict[str, Any] = field(default_factory=dict)
    next_observation_seq: int = 1
    next_heldout_seq: int = 1
    next_proposer_round_seq: int = 1


class HeldoutSnapshotLike(Protocol):
    round_idx: int
    best_candidate_id: str
    best_score: float
    baseline_score: float | None
    lift: float | None


class SQLiteMiproRunLedger:
    """Append-only run ledger with evidence materialization and query helpers."""

    def __init__(
        self,
        *,
        run_id: str,
        ledger_path: str,
        program_id: str,
        mode: str,
        resume: bool,
    ) -> None:
        self.run_id = str(run_id).strip()
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        self.ledger_path = str(ledger_path).strip()
        if not self.ledger_path:
            raise ValueError("ledger_path must be non-empty")
        self.program_id = str(program_id).strip()
        if not self.program_id:
            raise ValueError("program_id must be non-empty")
        self.mode = str(mode).strip()
        if not self.mode:
            raise ValueError("mode must be non-empty")

        path = Path(self.ledger_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.workspace_root = (
            path.parent / f"{_safe_path_component(path.stem)}_workspace"
        ).resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        if resume and not self._run_exists():
            raise ValueError(
                f"resume=True requested but run_id '{self.run_id}' was not found in {self.ledger_path}"
            )
        self._upsert_run(status="running")

    def close(self) -> None:
        self._conn.close()

    def set_status(self, status: str) -> None:
        self._upsert_run(status=str(status).strip() or "running")

    def get_run_status(self) -> dict[str, Any]:
        row = self._conn.execute(
            """
            SELECT run_id, program_id, mode, status, ledger_path, workspace_root, created_at, updated_at
            FROM runs
            WHERE run_id = ?
            """,
            (self.run_id,),
        ).fetchone()
        if row is None:
            return {}
        return {
            "run_id": str(row["run_id"]),
            "program_id": str(row["program_id"]),
            "mode": str(row["mode"]),
            "status": str(row["status"]),
            "ledger_path": str(row["ledger_path"]),
            "workspace_root": str(row["workspace_root"]),
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
        }

    def append_event(
        self,
        *,
        event_type: str,
        payload: dict[str, Any],
        round_idx: int | None = None,
        candidate_id: str | None = None,
    ) -> MiproRunEvent:
        normalized_event_type = str(event_type).strip()
        if not normalized_event_type:
            raise ValueError("event_type must be non-empty")
        now_ts = float(time.time())
        next_seq = int(
            self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq FROM run_events WHERE run_id = ?",
                (self.run_id,),
            ).fetchone()["next_seq"]
        )
        self._conn.execute(
            """
            INSERT INTO run_events(
                run_id, seq, event_type, payload_json, round_idx, candidate_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.run_id,
                next_seq,
                normalized_event_type,
                _json_dumps(dict(payload)),
                int(round_idx) if round_idx is not None else None,
                str(candidate_id).strip() or None,
                now_ts,
            ),
        )
        self._conn.commit()
        return MiproRunEvent(
            seq=next_seq,
            event_type=normalized_event_type,
            payload=dict(payload),
            round_idx=int(round_idx) if round_idx is not None else None,
            candidate_id=str(candidate_id).strip() or None,
            created_at=now_ts,
        )

    def append_observation(
        self, *, seq: int, round_idx: int, trial: MiproTrialResult
    ) -> bool:
        cur = self._conn.execute(
            """
            INSERT OR IGNORE INTO observations(
                run_id, seq, round_idx, config_json, score, details_json, candidate_id,
                lever_bundle_hash, latency_ms, cost_proxy, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.run_id,
                int(seq),
                int(round_idx),
                _json_dumps(dict(trial.config)),
                float(trial.score),
                _json_dumps(dict(trial.details)),
                str(trial.candidate_id or ""),
                str(trial.lever_bundle_hash or ""),
                float(trial.latency_ms),
                float(trial.cost_proxy),
                float(trial.timestamp),
            ),
        )
        self._conn.commit()
        return int(cur.rowcount) > 0

    def append_heldout_snapshot(
        self, *, seq: int, snapshot: HeldoutSnapshotLike
    ) -> bool:
        cur = self._conn.execute(
            """
            INSERT OR IGNORE INTO heldout_snapshots(
                run_id, seq, round_idx, best_candidate_id, best_score, baseline_score, lift
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.run_id,
                int(seq),
                int(snapshot.round_idx),
                str(snapshot.best_candidate_id),
                float(snapshot.best_score),
                (
                    float(snapshot.baseline_score)
                    if snapshot.baseline_score is not None
                    else None
                ),
                float(snapshot.lift) if snapshot.lift is not None else None,
            ),
        )
        self._conn.commit()
        return int(cur.rowcount) > 0

    def append_proposer_round(
        self,
        *,
        seq: int,
        round_idx: int,
        summary: dict[str, Any],
        diagnostics: dict[str, Any],
        stop_reason: str,
        skipped_tabu_delta: int,
    ) -> bool:
        cur = self._conn.execute(
            """
            INSERT OR IGNORE INTO proposer_rounds(
                run_id, seq, round_idx, summary_json, diagnostics_json, stop_reason, skipped_tabu_delta, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.run_id,
                int(seq),
                int(round_idx),
                _json_dumps(dict(summary)),
                _json_dumps(dict(diagnostics)),
                str(stop_reason),
                int(skipped_tabu_delta),
                float(time.time()),
            ),
        )
        self._conn.commit()
        return int(cur.rowcount) > 0

    def upsert_checkpoint(
        self,
        *,
        checkpoint_id: str,
        stage: str,
        round_idx: int,
        path: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        normalized_checkpoint_id = str(checkpoint_id).strip()
        if not normalized_checkpoint_id:
            raise ValueError("checkpoint_id must be non-empty")
        normalized_stage = str(stage).strip()
        if not normalized_stage:
            raise ValueError("checkpoint stage must be non-empty")
        normalized_path = str(path).strip()
        if not normalized_path:
            raise ValueError("checkpoint path must be non-empty")
        now_ts = float(time.time())
        self._conn.execute(
            """
            INSERT INTO checkpoints(
                run_id, checkpoint_id, stage, round_idx, path, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, checkpoint_id) DO UPDATE SET
                stage=excluded.stage,
                round_idx=excluded.round_idx,
                path=excluded.path,
                metadata_json=excluded.metadata_json
            """,
            (
                self.run_id,
                normalized_checkpoint_id,
                normalized_stage,
                int(round_idx),
                normalized_path,
                _json_dumps(dict(metadata or {})),
                now_ts,
            ),
        )
        self._conn.commit()

    def upsert_rollout_queue(
        self,
        *,
        queue_id: str,
        round_idx: int,
        queue_kind: str,
        queue_payload: dict[str, Any],
        artifact_path: str | None = None,
    ) -> dict[str, Any]:
        normalized_queue_id = str(queue_id).strip()
        if not normalized_queue_id:
            raise ValueError("queue_id must be non-empty")
        artifact_dir = self.workspace_root / "rollout_queues"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        resolved_artifact_path = (
            Path(artifact_path)
            if artifact_path is not None and str(artifact_path).strip()
            else artifact_dir / f"{_safe_path_component(normalized_queue_id)}.json"
        )
        payload = dict(queue_payload)
        payload.setdefault("queue_id", normalized_queue_id)
        payload.setdefault("round_idx", int(round_idx))
        resolved_artifact_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        self._conn.execute(
            """
            INSERT INTO rollout_queues(
                run_id, queue_id, round_idx, queue_kind, queue_json, artifact_path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, queue_id) DO UPDATE SET
                round_idx=excluded.round_idx,
                queue_kind=excluded.queue_kind,
                queue_json=excluded.queue_json,
                artifact_path=excluded.artifact_path
            """,
            (
                self.run_id,
                normalized_queue_id,
                int(round_idx),
                str(queue_kind),
                _json_dumps(payload),
                str(resolved_artifact_path),
                float(time.time()),
            ),
        )
        self._conn.commit()
        return {"queue": payload, "artifact_path": str(resolved_artifact_path)}

    def query_rollout_queues(self, *, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT queue_id, round_idx, queue_kind, queue_json, artifact_path, created_at
            FROM rollout_queues
            WHERE run_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (self.run_id, int(limit)),
        ).fetchall()
        return [
            {
                "queue_id": str(row["queue_id"]),
                "round_idx": int(row["round_idx"]),
                "queue_kind": str(row["queue_kind"]),
                "queue": dict(_json_loads(str(row["queue_json"]))),
                "artifact_path": str(row["artifact_path"]),
                "created_at": float(row["created_at"]),
            }
            for row in rows
        ]

    def upsert_state(self, *, key: str, value: Any) -> None:
        state_key = str(key).strip()
        if not state_key:
            raise ValueError("run_state key must be non-empty")
        self._conn.execute(
            """
            INSERT INTO run_state(run_id, key, value_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(run_id, key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at
            """,
            (
                self.run_id,
                state_key,
                _json_dumps(value),
                float(time.time()),
            ),
        )
        self._conn.commit()

    def upsert_candidate(
        self,
        *,
        candidate: MiproProgramCandidate,
        round_idx: int,
        candidate_metadata: dict[str, Any] | None = None,
    ) -> None:
        existing_row = self._conn.execute(
            """
            SELECT candidate_metadata_json, first_seen_round_idx
            FROM candidates
            WHERE run_id = ? AND candidate_id = ?
            """,
            (self.run_id, str(candidate.candidate_id or "")),
        ).fetchone()
        prompt_text = "\n\n".join(
            str(text).strip()
            for _, text in sorted(candidate.selected_instructions.items())
            if str(text).strip()
        )
        payload = candidate.to_dict()
        existing_metadata = (
            _json_field(existing_row["candidate_metadata_json"], default={})
            if existing_row is not None
            else {}
        )
        metadata = {
            **(
                dict(existing_metadata)
                if isinstance(existing_metadata, dict)
                else {}
            ),
            **dict(candidate_metadata or {}),
        }
        now_ts = float(time.time())
        self._conn.execute(
            """
            INSERT INTO candidates(
                run_id, candidate_id, parent_candidate_id, lever_bundle_hash, candidate_json,
                prompt_text, source_config_json, candidate_metadata_json, first_seen_round_idx,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, candidate_id) DO UPDATE SET
                parent_candidate_id=excluded.parent_candidate_id,
                lever_bundle_hash=excluded.lever_bundle_hash,
                candidate_json=excluded.candidate_json,
                prompt_text=excluded.prompt_text,
                source_config_json=excluded.source_config_json,
                candidate_metadata_json=excluded.candidate_metadata_json,
                updated_at=excluded.updated_at
            """,
            (
                self.run_id,
                str(candidate.candidate_id or ""),
                str(candidate.parent_candidate_id or "").strip() or None,
                str(candidate.lever_bundle_hash or ""),
                _json_dumps(payload),
                prompt_text,
                _json_dumps(candidate.source_config),
                _json_dumps(metadata),
                (
                    int(existing_row["first_seen_round_idx"])
                    if existing_row is not None
                    else int(round_idx)
                ),
                now_ts,
                now_ts,
            ),
        )
        self._conn.commit()

    def append_rollout(
        self,
        *,
        rollout_id: str,
        candidate_id: str,
        lever_bundle_hash: str,
        split: str,
        round_idx: int,
        task_row_ref: str | None,
        seed: int | None,
        prompt_id: str | None,
        score: float,
        score_components: dict[str, Any] | None,
        rollout_summary: dict[str, Any],
        trace_payload: Any,
        verifier_verdict: dict[str, Any] | None,
        evidence_artifacts: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        normalized_rollout_id = str(rollout_id).strip()
        if not normalized_rollout_id:
            raise ValueError("rollout_id must be non-empty")
        normalized_candidate_id = str(candidate_id).strip()
        if not normalized_candidate_id:
            raise ValueError("candidate_id must be non-empty")
        normalized_split = str(split).strip() or "train"
        summary_payload = dict(rollout_summary)
        trace_value = trace_payload
        if isinstance(trace_value, str):
            trace_value = {"trace_text": trace_value}
        elif trace_value is None:
            trace_value = {}
        verdict_payload = (
            dict(verifier_verdict)
            if isinstance(verifier_verdict, dict)
            else {
                "status": "unsupported",
                "reason": "verifier verdict not provided by evaluator",
            }
        )
        workspace_entry = self._materialize_rollout_evidence(
            rollout_id=normalized_rollout_id,
            candidate_id=normalized_candidate_id,
            split=normalized_split,
            summary_payload=summary_payload,
            trace_payload=trace_value,
            verifier_verdict=verdict_payload,
            evidence_artifacts=list(evidence_artifacts or []),
        )
        self._conn.execute(
            """
            DELETE FROM rollouts WHERE run_id = ? AND rollout_id = ?
            """,
            (self.run_id, normalized_rollout_id),
        )
        self._conn.execute(
            """
            DELETE FROM rollout_evidence_files WHERE run_id = ? AND rollout_id = ?
            """,
            (self.run_id, normalized_rollout_id),
        )
        self._conn.execute(
            """
            INSERT INTO rollouts(
                run_id, rollout_id, candidate_id, compare_parent_candidate_id, lever_bundle_hash,
                split, round_idx, task_row_ref, seed, prompt_id, score, score_components_json,
                rollout_summary_json, trace_payload_json, verifier_verdict_json, workspace_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.run_id,
                normalized_rollout_id,
                normalized_candidate_id,
                (
                    str(summary_payload.get("parent_candidate_id") or "").strip() or None
                ),
                str(lever_bundle_hash or ""),
                normalized_split,
                int(round_idx),
                str(task_row_ref or "").strip() or None,
                int(seed) if seed is not None else None,
                str(prompt_id or "").strip() or None,
                float(score),
                _json_dumps(dict(score_components or {})),
                _json_dumps(summary_payload),
                _json_dumps(trace_value),
                _json_dumps(verdict_payload),
                _json_dumps(workspace_entry),
                float(time.time()),
            ),
        )
        for evidence in workspace_entry.get("files") or []:
            self._conn.execute(
                """
                INSERT INTO rollout_evidence_files(
                    run_id, evidence_id, rollout_id, candidate_id, split, kind, path, relative_path,
                    size_bytes, content_hash, description, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.run_id,
                    str(evidence["evidence_id"]),
                    normalized_rollout_id,
                    normalized_candidate_id,
                    normalized_split,
                    str(evidence["kind"]),
                    str(evidence["path"]),
                    str(evidence["relative_path"]),
                    int(evidence["size_bytes"]),
                    str(evidence["content_hash"]),
                    str(evidence["description"]),
                    float(evidence["created_at"]),
                ),
            )
        self._conn.commit()
        return workspace_entry

    def upsert_candidate_delta(
        self,
        *,
        candidate_id: str,
        compare_to_candidate_id: str,
        comparison_kind: str,
        split: str,
        summary: dict[str, Any],
    ) -> dict[str, Any]:
        delta_id = (
            f"delta_{_safe_path_component(candidate_id)}_"
            f"{_safe_path_component(compare_to_candidate_id)}_{_safe_path_component(split)}"
        )
        artifact_dir = self.workspace_root / "candidate_deltas"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"{delta_id}.json"
        payload = {
            "delta_id": delta_id,
            "candidate_id": str(candidate_id),
            "compare_to_candidate_id": str(compare_to_candidate_id),
            "comparison_kind": str(comparison_kind),
            "split": str(split),
            **dict(summary),
        }
        artifact_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        now_ts = float(time.time())
        self._conn.execute(
            """
            INSERT INTO candidate_deltas(
                run_id, delta_id, candidate_id, compare_to_candidate_id, comparison_kind,
                split, summary_json, artifact_path, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, delta_id) DO UPDATE SET
                summary_json=excluded.summary_json,
                artifact_path=excluded.artifact_path,
                updated_at=excluded.updated_at
            """,
            (
                self.run_id,
                delta_id,
                str(candidate_id),
                str(compare_to_candidate_id),
                str(comparison_kind),
                str(split),
                _json_dumps(payload),
                str(artifact_path),
                now_ts,
                now_ts,
            ),
        )
        self._conn.commit()
        return payload

    def upsert_candidate_verdict_digest(
        self,
        *,
        candidate_id: str,
        split: str,
        summary: dict[str, Any],
    ) -> dict[str, Any]:
        digest_id = (
            f"verdict_{_safe_path_component(candidate_id)}_{_safe_path_component(split)}"
        )
        artifact_dir = self.workspace_root / "verdict_digests"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"{digest_id}.json"
        payload = {
            "digest_id": digest_id,
            "candidate_id": str(candidate_id),
            "split": str(split),
            **dict(summary),
        }
        artifact_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        now_ts = float(time.time())
        self._conn.execute(
            """
            INSERT INTO candidate_verdict_digests(
                run_id, digest_id, candidate_id, split, summary_json, artifact_path, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, digest_id) DO UPDATE SET
                summary_json=excluded.summary_json,
                artifact_path=excluded.artifact_path,
                updated_at=excluded.updated_at
            """,
            (
                self.run_id,
                digest_id,
                str(candidate_id),
                str(split),
                _json_dumps(payload),
                str(artifact_path),
                now_ts,
                now_ts,
            ),
        )
        self._conn.commit()
        return payload

    def upsert_model_transform(
        self,
        *,
        record: MiproModelTransformRecord,
        summary: TransformExecutionSummary | None = None,
    ) -> None:
        now_ts = float(time.time())
        summary_payload = summary.to_dict() if summary is not None else {}
        self._conn.execute(
            """
            INSERT INTO candidate_model_transforms(
                run_id, transform_id, transform_type, training_backend, parent_candidate_id,
                child_candidate_id, finetune_ref, status, training_summary_path,
                holdout_model_compare_path, contract_path, failure_path,
                baseline_holdout_score, finetuned_holdout_score, holdout_delta,
                num_train_samples, optimizer_steps, transform_stage,
                cost_proxy_train, cost_proxy_heldout, cost_proxy_total,
                estimated_cost_usd, estimate_source,
                inference_target_json, summary_json, metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, transform_id) DO UPDATE SET
                transform_type=excluded.transform_type,
                training_backend=excluded.training_backend,
                parent_candidate_id=excluded.parent_candidate_id,
                child_candidate_id=excluded.child_candidate_id,
                finetune_ref=excluded.finetune_ref,
                status=excluded.status,
                training_summary_path=excluded.training_summary_path,
                holdout_model_compare_path=excluded.holdout_model_compare_path,
                contract_path=excluded.contract_path,
                failure_path=excluded.failure_path,
                baseline_holdout_score=excluded.baseline_holdout_score,
                finetuned_holdout_score=excluded.finetuned_holdout_score,
                holdout_delta=excluded.holdout_delta,
                num_train_samples=excluded.num_train_samples,
                optimizer_steps=excluded.optimizer_steps,
                transform_stage=excluded.transform_stage,
                cost_proxy_train=excluded.cost_proxy_train,
                cost_proxy_heldout=excluded.cost_proxy_heldout,
                cost_proxy_total=excluded.cost_proxy_total,
                estimated_cost_usd=excluded.estimated_cost_usd,
                estimate_source=excluded.estimate_source,
                inference_target_json=excluded.inference_target_json,
                summary_json=excluded.summary_json,
                metadata_json=excluded.metadata_json,
                updated_at=excluded.updated_at
            """,
            (
                self.run_id,
                record.transform_id,
                record.transform_type,
                record.training_backend,
                record.parent_candidate_id,
                record.child_candidate_id,
                record.finetune_ref,
                record.status,
                record.training_summary_path,
                record.holdout_model_compare_path,
                record.contract_path,
                record.failure_path,
                record.baseline_holdout_score,
                record.finetuned_holdout_score,
                record.holdout_delta,
                record.num_train_samples,
                record.optimizer_steps,
                record.transform_stage,
                record.cost_proxy_train,
                record.cost_proxy_heldout,
                record.cost_proxy_total,
                record.estimated_cost_usd,
                record.estimate_source,
                _json_dumps(record.to_dict().get("inference_target")),
                _json_dumps(summary_payload),
                _json_dumps(dict(record.metadata)),
                now_ts,
                now_ts,
            ),
        )
        self._conn.commit()

    def upsert_transform_failure(
        self,
        *,
        transform_id: str,
        failure: TransformFailure,
    ) -> None:
        now_ts = float(time.time())
        self._conn.execute(
            """
            INSERT INTO transform_failures(
                run_id, transform_id, failure_kind, phase, retriable, message,
                request_path, result_path, stdout_path, stderr_path, traceback_text,
                metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, transform_id) DO UPDATE SET
                failure_kind=excluded.failure_kind,
                phase=excluded.phase,
                retriable=excluded.retriable,
                message=excluded.message,
                request_path=excluded.request_path,
                result_path=excluded.result_path,
                stdout_path=excluded.stdout_path,
                stderr_path=excluded.stderr_path,
                traceback_text=excluded.traceback_text,
                metadata_json=excluded.metadata_json,
                updated_at=excluded.updated_at
            """,
            (
                self.run_id,
                str(transform_id),
                failure.failure_kind,
                failure.phase,
                1 if failure.retriable else 0,
                failure.message,
                failure.request_path,
                failure.result_path,
                failure.stdout_path,
                failure.stderr_path,
                failure.traceback_text,
                _json_dumps(dict(failure.metadata)),
                now_ts,
                now_ts,
            ),
        )
        self._conn.commit()

    def query_model_transforms(
        self,
        *,
        candidate_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT
                transform_id, transform_type, training_backend, parent_candidate_id,
                child_candidate_id, finetune_ref, status, training_summary_path,
                holdout_model_compare_path, contract_path, failure_path,
                baseline_holdout_score, finetuned_holdout_score, holdout_delta,
                num_train_samples, optimizer_steps, transform_stage,
                cost_proxy_train, cost_proxy_heldout, cost_proxy_total,
                estimated_cost_usd, estimate_source,
                inference_target_json,
                summary_json, metadata_json, created_at, updated_at
            FROM candidate_model_transforms
            WHERE run_id = ?
              AND (? IS NULL OR parent_candidate_id = ? OR child_candidate_id = ?)
            ORDER BY updated_at DESC, transform_id DESC
            LIMIT ?
            """,
            (self.run_id, candidate_id, candidate_id, candidate_id, int(limit)),
        ).fetchall()
        return [
            {
                "transform_id": str(row["transform_id"]),
                "transform_type": str(row["transform_type"]),
                "training_backend": str(row["training_backend"]),
                "parent_candidate_id": str(row["parent_candidate_id"]),
                "child_candidate_id": str(row["child_candidate_id"] or "").strip() or None,
                "finetune_ref": str(row["finetune_ref"] or "").strip() or None,
                "status": str(row["status"]),
                "training_summary_path": str(row["training_summary_path"] or "").strip() or None,
                "holdout_model_compare_path": str(row["holdout_model_compare_path"] or "").strip() or None,
                "contract_path": str(row["contract_path"] or "").strip() or None,
                "failure_path": str(row["failure_path"] or "").strip() or None,
                "baseline_holdout_score": float(row["baseline_holdout_score"])
                if row["baseline_holdout_score"] is not None
                else None,
                "finetuned_holdout_score": float(row["finetuned_holdout_score"])
                if row["finetuned_holdout_score"] is not None
                else None,
                "holdout_delta": float(row["holdout_delta"])
                if row["holdout_delta"] is not None
                else None,
                "num_train_samples": int(row["num_train_samples"])
                if row["num_train_samples"] is not None
                else None,
                "optimizer_steps": int(row["optimizer_steps"])
                if row["optimizer_steps"] is not None
                else None,
                "transform_stage": str(row["transform_stage"] or "").strip() or None,
                "cost_proxy_train": float(row["cost_proxy_train"])
                if row["cost_proxy_train"] is not None
                else None,
                "cost_proxy_heldout": float(row["cost_proxy_heldout"])
                if row["cost_proxy_heldout"] is not None
                else None,
                "cost_proxy_total": float(row["cost_proxy_total"])
                if row["cost_proxy_total"] is not None
                else None,
                "estimated_cost_usd": float(row["estimated_cost_usd"])
                if row["estimated_cost_usd"] is not None
                else None,
                "estimate_source": str(row["estimate_source"] or "").strip() or None,
                "inference_target": _json_field(row["inference_target_json"], default=None),
                "summary": _json_field(row["summary_json"], default={}),
                "metadata": _json_field(row["metadata_json"], default={}),
                "created_at": float(row["created_at"]),
                "updated_at": float(row["updated_at"]),
            }
            for row in rows
        ]

    def query_transform_failures(
        self,
        *,
        transform_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT
                transform_id, failure_kind, phase, retriable, message,
                request_path, result_path, stdout_path, stderr_path, traceback_text,
                metadata_json, created_at, updated_at
            FROM transform_failures
            WHERE run_id = ?
              AND (? IS NULL OR transform_id = ?)
            ORDER BY updated_at DESC, transform_id DESC
            LIMIT ?
            """,
            (self.run_id, transform_id, transform_id, int(limit)),
        ).fetchall()
        return [
            {
                "transform_id": str(row["transform_id"]),
                "failure_kind": str(row["failure_kind"]),
                "phase": str(row["phase"]),
                "retriable": bool(int(row["retriable"] or 0)),
                "message": str(row["message"]),
                "request_path": str(row["request_path"] or "").strip() or None,
                "result_path": str(row["result_path"] or "").strip() or None,
                "stdout_path": str(row["stdout_path"] or "").strip() or None,
                "stderr_path": str(row["stderr_path"] or "").strip() or None,
                "traceback_text": str(row["traceback_text"] or ""),
                "metadata": _json_field(row["metadata_json"], default={}),
                "created_at": float(row["created_at"]),
                "updated_at": float(row["updated_at"]),
            }
            for row in rows
        ]

    def query_candidates(
        self,
        *,
        candidate_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT
                c.candidate_id,
                c.parent_candidate_id,
                c.lever_bundle_hash,
                c.prompt_text,
                c.candidate_json,
                c.source_config_json,
                c.candidate_metadata_json,
                c.first_seen_round_idx,
                c.created_at,
                c.updated_at,
                COUNT(r.rollout_id) AS rollout_count,
                AVG(r.score) AS average_score
            FROM candidates AS c
            LEFT JOIN rollouts AS r
              ON r.run_id = c.run_id
             AND r.candidate_id = c.candidate_id
             AND r.split = 'train'
            WHERE c.run_id = ?
              AND (? IS NULL OR c.candidate_id = ?)
            GROUP BY
                c.candidate_id,
                c.parent_candidate_id,
                c.lever_bundle_hash,
                c.prompt_text,
                c.candidate_json,
                c.source_config_json,
                c.candidate_metadata_json,
                c.first_seen_round_idx,
                c.created_at,
                c.updated_at
            ORDER BY
                CASE WHEN AVG(r.score) IS NULL THEN 1 ELSE 0 END,
                AVG(r.score) DESC,
                c.updated_at DESC,
                c.candidate_id ASC
            LIMIT ?
            """,
            (self.run_id, candidate_id, candidate_id, int(limit)),
        ).fetchall()
        payload: list[dict[str, Any]] = []
        for row in rows:
            candidate_payload = _json_field(row["candidate_json"], default={})
            selected_instruction_base_option_ids = dict(
                candidate_payload.get("selected_instruction_base_option_ids") or {}
            )
            selected_instruction_transform_ids = {
                str(module_id): list(transform_ids or [])
                for module_id, transform_ids in dict(
                    candidate_payload.get("selected_instruction_transform_ids") or {}
                ).items()
            }
            payload.append(
                {
                    "candidate_id": str(row["candidate_id"]),
                    "parent_candidate_id": str(row["parent_candidate_id"] or "").strip()
                    or None,
                    "lever_bundle_hash": str(row["lever_bundle_hash"]),
                    "prompt_text_preview": str(row["prompt_text"])[:240],
                    "candidate": candidate_payload,
                    "source_config": _json_field(row["source_config_json"], default={}),
                    "candidate_metadata": _json_field(
                        row["candidate_metadata_json"], default={}
                    ),
                    "selected_instruction_base_option_ids": selected_instruction_base_option_ids,
                    "selected_instruction_transform_ids": selected_instruction_transform_ids,
                    "transform_count_total": sum(
                        len(transform_ids)
                        for transform_ids in selected_instruction_transform_ids.values()
                    ),
                    "first_seen_round_idx": int(row["first_seen_round_idx"]),
                    "rollout_count": int(row["rollout_count"] or 0),
                    "average_score": (
                        float(row["average_score"])
                        if row["average_score"] is not None
                        else None
                    ),
                    "created_at": float(row["created_at"]),
                    "updated_at": float(row["updated_at"]),
                }
            )
        return payload

    def query_rollouts(
        self,
        *,
        candidate_id: str | None = None,
        split: str | None = None,
        rollout_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT
                rollout_id,
                candidate_id,
                compare_parent_candidate_id,
                lever_bundle_hash,
                split,
                round_idx,
                task_row_ref,
                seed,
                prompt_id,
                score,
                score_components_json,
                rollout_summary_json,
                trace_payload_json,
                verifier_verdict_json,
                workspace_json,
                created_at
            FROM rollouts
            WHERE run_id = ?
              AND (? IS NULL OR candidate_id = ?)
              AND (? IS NULL OR split = ?)
              AND (? IS NULL OR rollout_id = ?)
            ORDER BY created_at DESC, rollout_id DESC
            LIMIT ?
            """,
            (
                self.run_id,
                candidate_id,
                candidate_id,
                split,
                split,
                rollout_id,
                rollout_id,
                int(limit),
            ),
        ).fetchall()
        return [
            {
                "rollout_id": str(row["rollout_id"]),
                "candidate_id": str(row["candidate_id"]),
                "parent_candidate_id": str(
                    row["compare_parent_candidate_id"] or ""
                ).strip()
                or None,
                "lever_bundle_hash": str(row["lever_bundle_hash"]),
                "split": str(row["split"]),
                "round_idx": int(row["round_idx"]),
                "task_row_ref": str(row["task_row_ref"] or "").strip() or None,
                "seed": int(row["seed"]) if row["seed"] is not None else None,
                "prompt_id": str(row["prompt_id"] or "").strip() or None,
                "score": float(row["score"]),
                "score_components": _json_field(
                    row["score_components_json"], default={}
                ),
                "rollout_summary": _json_field(
                    row["rollout_summary_json"], default={}
                ),
                "trace_payload": _json_field(row["trace_payload_json"], default={}),
                "verifier_verdict": _json_field(
                    row["verifier_verdict_json"], default={}
                ),
                "workspace": _json_field(row["workspace_json"], default={}),
                "created_at": float(row["created_at"]),
            }
            for row in rows
        ]

    def query_evidence_files(
        self,
        *,
        rollout_id: str | None = None,
        candidate_id: str | None = None,
        kind: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT
                evidence_id,
                rollout_id,
                candidate_id,
                split,
                kind,
                path,
                relative_path,
                size_bytes,
                content_hash,
                description,
                created_at
            FROM rollout_evidence_files
            WHERE run_id = ?
              AND (? IS NULL OR rollout_id = ?)
              AND (? IS NULL OR candidate_id = ?)
              AND (? IS NULL OR kind = ?)
            ORDER BY created_at DESC, evidence_id DESC
            LIMIT ?
            """,
            (
                self.run_id,
                rollout_id,
                rollout_id,
                candidate_id,
                candidate_id,
                kind,
                kind,
                int(limit),
            ),
        ).fetchall()
        return [
            {
                "evidence_id": str(row["evidence_id"]),
                "rollout_id": str(row["rollout_id"]),
                "candidate_id": str(row["candidate_id"]),
                "split": str(row["split"]),
                "kind": str(row["kind"]),
                "path": str(row["path"]),
                "relative_path": str(row["relative_path"]),
                "size_bytes": int(row["size_bytes"]),
                "content_hash": str(row["content_hash"]),
                "description": str(row["description"]),
                "created_at": float(row["created_at"]),
            }
            for row in rows
        ]

    def query_candidate_rollout_deltas(
        self,
        *,
        candidate_id: str | None = None,
        compare_to_candidate_id: str | None = None,
        split: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT delta_id, candidate_id, compare_to_candidate_id, comparison_kind, split,
                   summary_json, artifact_path, created_at, updated_at
            FROM candidate_deltas
            WHERE run_id = ?
              AND (? IS NULL OR candidate_id = ?)
              AND (? IS NULL OR compare_to_candidate_id = ?)
              AND (? IS NULL OR split = ?)
            ORDER BY updated_at DESC, delta_id DESC
            LIMIT ?
            """,
            (
                self.run_id,
                candidate_id,
                candidate_id,
                compare_to_candidate_id,
                compare_to_candidate_id,
                split,
                split,
                int(limit),
            ),
        ).fetchall()
        return [
            {
                "delta_id": str(row["delta_id"]),
                "candidate_id": str(row["candidate_id"]),
                "compare_to_candidate_id": str(row["compare_to_candidate_id"]),
                "comparison_kind": str(row["comparison_kind"]),
                "split": str(row["split"]),
                "summary": _json_field(row["summary_json"], default={}),
                "artifact_path": str(row["artifact_path"]),
                "created_at": float(row["created_at"]),
                "updated_at": float(row["updated_at"]),
            }
            for row in rows
        ]

    def query_candidate_verdict_digests(
        self,
        *,
        candidate_id: str | None = None,
        split: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT digest_id, candidate_id, split, summary_json, artifact_path, created_at, updated_at
            FROM candidate_verdict_digests
            WHERE run_id = ?
              AND (? IS NULL OR candidate_id = ?)
              AND (? IS NULL OR split = ?)
            ORDER BY updated_at DESC, digest_id DESC
            LIMIT ?
            """,
            (
                self.run_id,
                candidate_id,
                candidate_id,
                split,
                split,
                int(limit),
            ),
        ).fetchall()
        return [
            {
                "digest_id": str(row["digest_id"]),
                "candidate_id": str(row["candidate_id"]),
                "split": str(row["split"]),
                "summary": _json_field(row["summary_json"], default={}),
                "artifact_path": str(row["artifact_path"]),
                "created_at": float(row["created_at"]),
                "updated_at": float(row["updated_at"]),
            }
            for row in rows
        ]

    def query_heldout_snapshots(self, *, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT seq, round_idx, best_candidate_id, best_score, baseline_score, lift
            FROM heldout_snapshots
            WHERE run_id = ?
            ORDER BY seq DESC
            LIMIT ?
            """,
            (self.run_id, int(limit)),
        ).fetchall()
        return [
            {
                "seq": int(row["seq"]),
                "round_idx": int(row["round_idx"]),
                "best_candidate_id": str(row["best_candidate_id"]),
                "best_score": float(row["best_score"]),
                "baseline_score": (
                    float(row["baseline_score"])
                    if row["baseline_score"] is not None
                    else None
                ),
                "lift": float(row["lift"]) if row["lift"] is not None else None,
            }
            for row in rows
        ]

    def query_proposer_rounds(self, *, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT seq, round_idx, summary_json, diagnostics_json, stop_reason, skipped_tabu_delta, created_at
            FROM proposer_rounds
            WHERE run_id = ?
            ORDER BY seq DESC
            LIMIT ?
            """,
            (self.run_id, int(limit)),
        ).fetchall()
        return [
            {
                "seq": int(row["seq"]),
                "round_idx": int(row["round_idx"]),
                "summary": _json_field(row["summary_json"], default={}),
                "diagnostics": _json_field(row["diagnostics_json"], default={}),
                "stop_reason": str(row["stop_reason"]),
                "skipped_tabu_delta": int(row["skipped_tabu_delta"] or 0),
                "created_at": float(row["created_at"]),
            }
            for row in rows
        ]

    def query_run_state(self) -> dict[str, Any]:
        rows = self._conn.execute(
            """
            SELECT key, value_json
            FROM run_state
            WHERE run_id = ?
            ORDER BY key ASC
            """,
            (self.run_id,),
        ).fetchall()
        return {
            str(row["key"]): _json_field(row["value_json"], default=None) for row in rows
        }

    def query_events(
        self,
        *,
        event_type: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT seq, event_type, payload_json, round_idx, candidate_id, created_at
            FROM run_events
            WHERE run_id = ?
              AND (? IS NULL OR event_type = ?)
            ORDER BY seq DESC
            LIMIT ?
            """,
            (self.run_id, event_type, event_type, int(limit)),
        ).fetchall()
        return [
            {
                "seq": int(row["seq"]),
                "event_type": str(row["event_type"]),
                "payload": _json_field(row["payload_json"], default={}),
                "round_idx": int(row["round_idx"]) if row["round_idx"] is not None else None,
                "candidate_id": str(row["candidate_id"] or "").strip() or None,
                "created_at": float(row["created_at"]),
            }
            for row in rows
        ]

    def query_checkpoints(
        self,
        *,
        stage: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT checkpoint_id, stage, round_idx, path, metadata_json, created_at
            FROM checkpoints
            WHERE run_id = ?
              AND (? IS NULL OR stage = ?)
            ORDER BY round_idx ASC, created_at ASC
            LIMIT ?
            """,
            (self.run_id, stage, stage, int(limit)),
        ).fetchall()
        return [
            {
                "checkpoint_id": str(row["checkpoint_id"]),
                "stage": str(row["stage"]),
                "round_idx": int(row["round_idx"]),
                "path": str(row["path"]),
                "metadata": _json_field(row["metadata_json"], default={}),
                "created_at": float(row["created_at"]),
            }
            for row in rows
        ]

    def build_run_read_model(self, *, limit_per_section: int = 50) -> dict[str, Any]:
        state = self.query_run_state()
        return {
            "run": self.get_run_status(),
            "best_summary": {
                "best_train_score": state.get("best_train_score"),
                "best_train_candidate": state.get("best_train_candidate"),
                "heldout_baseline_score": state.get("heldout_baseline_score"),
                "heldout_best_score": state.get("heldout_best_score"),
                "heldout_lift": state.get("heldout_lift"),
            },
            "run_state": state,
            "candidates": self.query_candidates(limit=limit_per_section),
            "model_transforms": self.query_model_transforms(limit=limit_per_section),
            "transform_failures": self.query_transform_failures(limit=limit_per_section),
            "heldout_snapshots": self.query_heldout_snapshots(limit=limit_per_section),
            "proposer_rounds": self.query_proposer_rounds(limit=limit_per_section),
            "checkpoints": self.query_checkpoints(limit=limit_per_section),
            "events": self.query_events(limit=limit_per_section),
            "workspace": self.workspace_snapshot(limit_per_section=min(limit_per_section, 20)),
        }

    def read_evidence_file(
        self,
        path: str,
        *,
        max_chars: int = 12000,
        offset: int = 0,
    ) -> dict[str, Any]:
        requested = Path(str(path)).expanduser()
        if not requested.is_absolute():
            requested = self.workspace_root / requested
        resolved = requested.resolve()
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError as exc:
            raise ValueError("evidence file must live under workspace_root") from exc
        if not resolved.is_file():
            raise FileNotFoundError(str(resolved))
        text = resolved.read_text(encoding="utf-8", errors="replace")
        start = max(0, int(offset))
        limit = max(1, int(max_chars))
        end = start + limit
        return {
            "path": str(resolved),
            "offset": start,
            "truncated": len(text) > end,
            "content": text[start:end],
        }

    def get_rollout_trace(
        self, *, rollout_id: str, max_chars: int = 12000, offset: int = 0
    ) -> dict[str, Any]:
        trace_files = [
            item
            for item in self.query_evidence_files(rollout_id=rollout_id, limit=50)
            if item["kind"] in {"rollout_trace", "trace_summary"}
        ]
        trace_payload = next(
            (item for item in trace_files if item["kind"] == "rollout_trace"), None
        )
        trace_summary = next(
            (item for item in trace_files if item["kind"] == "trace_summary"), None
        )
        if trace_payload is None:
            return {"ok": False, "error": f"no_rollout_trace_materialized:{rollout_id}"}
        return {
            "ok": True,
            "rollout_id": str(rollout_id),
            "trace_file": trace_payload,
            "trace_preview": self.read_evidence_file(
                trace_payload["path"], max_chars=max_chars, offset=offset
            ),
            "trace_summary_file": trace_summary,
            "trace_summary_preview": (
                self.read_evidence_file(
                    trace_summary["path"], max_chars=max_chars, offset=offset
                )
                if trace_summary is not None
                else None
            ),
        }

    def workspace_snapshot(self, *, limit_per_section: int = 12) -> dict[str, Any]:
        return {
            "workspace_root": str(self.workspace_root),
            "ledger_path": str(self.ledger_path),
            "candidates": self.query_candidates(limit=limit_per_section),
            "model_transforms": self.query_model_transforms(limit=limit_per_section),
            "transform_failures": self.query_transform_failures(limit=limit_per_section),
            "rollouts": self.query_rollouts(limit=limit_per_section),
            "evidence_files": self.query_evidence_files(limit=limit_per_section),
            "candidate_deltas": self.query_candidate_rollout_deltas(
                limit=limit_per_section
            ),
            "verdict_digests": self.query_candidate_verdict_digests(
                limit=limit_per_section
            ),
            "checkpoints": self.query_checkpoints(limit=limit_per_section),
        }

    def load_resume_state(self) -> MiproRunResumeState:
        observations_rows = list(
            self._conn.execute(
                """
                SELECT seq, round_idx, config_json, score, details_json, candidate_id, lever_bundle_hash,
                       latency_ms, cost_proxy, timestamp
                FROM observations
                WHERE run_id = ?
                ORDER BY seq ASC
                """,
                (self.run_id,),
            ).fetchall()
        )
        observations: list[LedgerObservation] = []
        for row in observations_rows:
            trial = MiproTrialResult(
                config=dict(_json_loads(str(row["config_json"]))),
                score=float(row["score"]),
                details=dict(_json_loads(str(row["details_json"]))),
                latency_ms=float(row["latency_ms"]),
                cost_proxy=float(row["cost_proxy"]),
                timestamp=float(row["timestamp"]),
                candidate_id=(str(row["candidate_id"]).strip() or None),
                lever_bundle_hash=(str(row["lever_bundle_hash"]).strip() or None),
            )
            observations.append(
                LedgerObservation(
                    seq=int(row["seq"]),
                    round_idx=int(row["round_idx"]),
                    trial=trial,
                )
            )

        heldout_rows = list(
            self._conn.execute(
                """
                SELECT seq, round_idx, best_candidate_id, best_score, baseline_score, lift
                FROM heldout_snapshots
                WHERE run_id = ?
                ORDER BY seq ASC
                """,
                (self.run_id,),
            ).fetchall()
        )
        heldout_snapshots: list[LedgerHeldoutSnapshot] = []
        for row in heldout_rows:
            heldout_snapshots.append(
                LedgerHeldoutSnapshot(
                    seq=int(row["seq"]),
                    snapshot=RunLedgerHeldoutSnapshot(
                        round_idx=int(row["round_idx"]),
                        best_candidate_id=str(row["best_candidate_id"]),
                        best_score=float(row["best_score"]),
                        baseline_score=(
                            float(row["baseline_score"])
                            if row["baseline_score"] is not None
                            else None
                        ),
                        lift=float(row["lift"]) if row["lift"] is not None else None,
                    ),
                )
            )

        proposer_rows = list(
            self._conn.execute(
                """
                SELECT seq, round_idx, summary_json, diagnostics_json, stop_reason, skipped_tabu_delta
                FROM proposer_rounds
                WHERE run_id = ?
                ORDER BY seq ASC
                """,
                (self.run_id,),
            ).fetchall()
        )
        proposer_rounds: list[LedgerProposerRound] = []
        for row in proposer_rows:
            proposer_rounds.append(
                LedgerProposerRound(
                    seq=int(row["seq"]),
                    round_idx=int(row["round_idx"]),
                    summary=dict(_json_loads(str(row["summary_json"]))),
                    diagnostics=dict(_json_loads(str(row["diagnostics_json"]))),
                    stop_reason=str(row["stop_reason"]),
                    skipped_tabu_delta=int(row["skipped_tabu_delta"] or 0),
                )
            )

        state_rows = list(
            self._conn.execute(
                """
                SELECT key, value_json
                FROM run_state
                WHERE run_id = ?
                ORDER BY key ASC
                """,
                (self.run_id,),
            ).fetchall()
        )
        run_state = {
            str(row["key"]): _json_loads(str(row["value_json"])) for row in state_rows
        }
        return MiproRunResumeState(
            run_id=self.run_id,
            ledger_path=self.ledger_path,
            observations=observations,
            heldout_snapshots=heldout_snapshots,
            proposer_rounds=proposer_rounds,
            run_state=run_state,
            next_observation_seq=1 + (max((item.seq for item in observations), default=0)),
            next_heldout_seq=1
            + (max((item.seq for item in heldout_snapshots), default=0)),
            next_proposer_round_seq=1
            + (max((item.seq for item in proposer_rounds), default=0)),
        )

    @staticmethod
    def candidate_from_state(value: Any) -> MiproProgramCandidate | None:
        if not isinstance(value, dict):
            return None
        try:
            return MiproProgramCandidate.from_dict(value)
        except Exception:
            return None

    def _run_exists(self) -> bool:
        row = self._conn.execute(
            "SELECT run_id FROM runs WHERE run_id = ?",
            (self.run_id,),
        ).fetchone()
        return row is not None

    def _upsert_run(self, *, status: str) -> None:
        now_ts = float(time.time())
        self._conn.execute(
            """
            INSERT INTO runs(run_id, program_id, mode, status, ledger_path, workspace_root, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                program_id=excluded.program_id,
                mode=excluded.mode,
                status=excluded.status,
                ledger_path=excluded.ledger_path,
                workspace_root=excluded.workspace_root,
                updated_at=excluded.updated_at
            """,
            (
                self.run_id,
                self.program_id,
                self.mode,
                status,
                self.ledger_path,
                str(self.workspace_root),
                now_ts,
                now_ts,
            ),
        )
        self._conn.commit()

    def _materialize_rollout_evidence(
        self,
        *,
        rollout_id: str,
        candidate_id: str,
        split: str,
        summary_payload: dict[str, Any],
        trace_payload: Any,
        verifier_verdict: dict[str, Any],
        evidence_artifacts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        rollout_dir = self.workspace_root / "rollouts" / _safe_path_component(rollout_id)
        rollout_dir.mkdir(parents=True, exist_ok=True)
        files: list[dict[str, Any]] = []

        def write_text_file(
            filename: str,
            content: str,
            *,
            kind: str,
            description: str,
        ) -> dict[str, Any]:
            path = rollout_dir / filename
            path.write_text(str(content), encoding="utf-8")
            created_at = float(time.time())
            entry = {
                "evidence_id": f"evidence_{_hash_text(f'{rollout_id}:{path.name}:{kind}')[:16]}",
                "rollout_id": rollout_id,
                "candidate_id": candidate_id,
                "split": split,
                "kind": kind,
                "path": str(path),
                "relative_path": str(path.relative_to(self.workspace_root)),
                "size_bytes": int(path.stat().st_size),
                "content_hash": _hash_text(path.read_text(encoding="utf-8")),
                "description": description,
                "created_at": created_at,
            }
            files.append(entry)
            return entry

        write_text_file(
            "rollout_summary.json",
            json.dumps(summary_payload, indent=2, sort_keys=True, ensure_ascii=True),
            kind="rollout_summary",
            description="Canonical rollout summary for one candidate x row evaluation.",
        )
        trace_json = (
            trace_payload
            if isinstance(trace_payload, (dict, list))
            else {"trace_text": str(trace_payload)}
        )
        write_text_file(
            "rollout_trace.json",
            json.dumps(trace_json, indent=2, sort_keys=True, ensure_ascii=True),
            kind="rollout_trace",
            description="Structured rollout trace payload for proposer inspection.",
        )
        write_text_file(
            "trace_summary.md",
            self._summarize_trace_payload(trace_json),
            kind="trace_summary",
            description="Human-readable summary of the rollout trace payload.",
        )
        write_text_file(
            "verifier_verdict.json",
            json.dumps(verifier_verdict, indent=2, sort_keys=True, ensure_ascii=True),
            kind="verifier_verdict",
            description="Structured verifier verdict for this rollout when available.",
        )
        artifact_index: list[dict[str, Any]] = []
        for idx, artifact in enumerate(evidence_artifacts):
            if not isinstance(artifact, dict):
                continue
            kind = str(artifact.get("kind") or "artifact").strip() or "artifact"
            name = str(artifact.get("name") or f"{kind}_{idx:03d}.txt").strip()
            payload = artifact.get("content")
            content = (
                json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)
                if isinstance(payload, (dict, list))
                else str(payload or "")
            )
            file_entry = write_text_file(
                f"{idx:03d}_{_safe_path_component(name)}",
                content,
                kind=kind,
                description=str(artifact.get("description") or f"Artifact {kind}."),
            )
            artifact_index.append(
                {
                    "kind": file_entry["kind"],
                    "path": file_entry["path"],
                    "relative_path": file_entry["relative_path"],
                    "size_bytes": file_entry["size_bytes"],
                }
            )
        write_text_file(
            "artifact_index.json",
            json.dumps(artifact_index, indent=2, sort_keys=True, ensure_ascii=True),
            kind="artifact_index",
            description="Index of materialized rollout evidence files.",
        )
        return {
            "workspace_dir": str(rollout_dir),
            "files": files,
            "updated_at": float(time.time()),
        }

    @staticmethod
    def _summarize_trace_payload(trace_payload: Any) -> str:
        if isinstance(trace_payload, dict):
            lines: list[str] = []
            for key in (
                "trace_text",
                "prompt_excerpt",
                "assistant_response_excerpt",
                "ideal_excerpt",
                "reasoning_trace",
            ):
                value = str(trace_payload.get(key) or "").strip()
                if value:
                    lines.append(f"{key}: {value}")
            if lines:
                return "\n".join(lines)
            return json.dumps(trace_payload, indent=2, sort_keys=True, ensure_ascii=True)[:4000]
        if isinstance(trace_payload, list):
            return json.dumps(trace_payload, indent=2, sort_keys=True, ensure_ascii=True)[:4000]
        return str(trace_payload or "")

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                program_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                status TEXT NOT NULL,
                ledger_path TEXT NOT NULL,
                workspace_root TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS observations (
                run_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                round_idx INTEGER NOT NULL,
                config_json TEXT NOT NULL,
                score REAL NOT NULL,
                details_json TEXT NOT NULL,
                candidate_id TEXT,
                lever_bundle_hash TEXT,
                latency_ms REAL NOT NULL,
                cost_proxy REAL NOT NULL,
                timestamp REAL NOT NULL,
                PRIMARY KEY (run_id, seq)
            );

            CREATE TABLE IF NOT EXISTS heldout_snapshots (
                run_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                round_idx INTEGER NOT NULL,
                best_candidate_id TEXT NOT NULL,
                best_score REAL NOT NULL,
                baseline_score REAL,
                lift REAL,
                PRIMARY KEY (run_id, seq)
            );

            CREATE TABLE IF NOT EXISTS proposer_rounds (
                run_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                round_idx INTEGER NOT NULL,
                summary_json TEXT NOT NULL,
                diagnostics_json TEXT NOT NULL,
                stop_reason TEXT NOT NULL,
                skipped_tabu_delta INTEGER NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (run_id, seq)
            );

            CREATE TABLE IF NOT EXISTS checkpoints (
                run_id TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                round_idx INTEGER NOT NULL,
                path TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (run_id, checkpoint_id)
            );

            CREATE TABLE IF NOT EXISTS rollout_queues (
                run_id TEXT NOT NULL,
                queue_id TEXT NOT NULL,
                round_idx INTEGER NOT NULL,
                queue_kind TEXT NOT NULL,
                queue_json TEXT NOT NULL,
                artifact_path TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (run_id, queue_id)
            );

            CREATE TABLE IF NOT EXISTS run_state (
                run_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value_json TEXT NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (run_id, key)
            );

            CREATE TABLE IF NOT EXISTS run_events (
                run_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                round_idx INTEGER,
                candidate_id TEXT,
                created_at REAL NOT NULL,
                PRIMARY KEY (run_id, seq)
            );

            CREATE TABLE IF NOT EXISTS candidates (
                run_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                parent_candidate_id TEXT,
                lever_bundle_hash TEXT NOT NULL,
                candidate_json TEXT NOT NULL,
                prompt_text TEXT NOT NULL,
                source_config_json TEXT NOT NULL,
                candidate_metadata_json TEXT NOT NULL,
                first_seen_round_idx INTEGER NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (run_id, candidate_id)
            );

            CREATE TABLE IF NOT EXISTS rollouts (
                run_id TEXT NOT NULL,
                rollout_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                compare_parent_candidate_id TEXT,
                lever_bundle_hash TEXT NOT NULL,
                split TEXT NOT NULL,
                round_idx INTEGER NOT NULL,
                task_row_ref TEXT,
                seed INTEGER,
                prompt_id TEXT,
                score REAL NOT NULL,
                score_components_json TEXT NOT NULL,
                rollout_summary_json TEXT NOT NULL,
                trace_payload_json TEXT NOT NULL,
                verifier_verdict_json TEXT NOT NULL,
                workspace_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (run_id, rollout_id)
            );

            CREATE TABLE IF NOT EXISTS rollout_evidence_files (
                run_id TEXT NOT NULL,
                evidence_id TEXT NOT NULL,
                rollout_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                split TEXT NOT NULL,
                kind TEXT NOT NULL,
                path TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                description TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (run_id, evidence_id)
            );

            CREATE TABLE IF NOT EXISTS candidate_deltas (
                run_id TEXT NOT NULL,
                delta_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                compare_to_candidate_id TEXT NOT NULL,
                comparison_kind TEXT NOT NULL,
                split TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                artifact_path TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (run_id, delta_id)
            );

            CREATE TABLE IF NOT EXISTS candidate_verdict_digests (
                run_id TEXT NOT NULL,
                digest_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                split TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                artifact_path TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (run_id, digest_id)
            );

            CREATE TABLE IF NOT EXISTS candidate_model_transforms (
                run_id TEXT NOT NULL,
                transform_id TEXT NOT NULL,
                transform_type TEXT NOT NULL,
                training_backend TEXT NOT NULL,
                parent_candidate_id TEXT NOT NULL,
                child_candidate_id TEXT,
                finetune_ref TEXT,
                status TEXT NOT NULL,
                training_summary_path TEXT,
                holdout_model_compare_path TEXT,
                contract_path TEXT,
                failure_path TEXT,
                baseline_holdout_score REAL,
                finetuned_holdout_score REAL,
                holdout_delta REAL,
                num_train_samples INTEGER,
                optimizer_steps INTEGER,
                transform_stage TEXT,
                cost_proxy_train REAL,
                cost_proxy_heldout REAL,
                cost_proxy_total REAL,
                estimated_cost_usd REAL,
                estimate_source TEXT,
                inference_target_json TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (run_id, transform_id)
            );

            CREATE TABLE IF NOT EXISTS transform_failures (
                run_id TEXT NOT NULL,
                transform_id TEXT NOT NULL,
                failure_kind TEXT NOT NULL,
                phase TEXT NOT NULL,
                retriable INTEGER NOT NULL,
                message TEXT NOT NULL,
                request_path TEXT,
                result_path TEXT,
                stdout_path TEXT,
                stderr_path TEXT,
                traceback_text TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (run_id, transform_id)
            );
            """
        )
        self._migrate_candidate_model_transforms_schema()
        self._conn.commit()

    def _migrate_candidate_model_transforms_schema(self) -> None:
        existing = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(candidate_model_transforms)").fetchall()
        }
        desired_columns: list[tuple[str, str]] = [
            ("transform_stage", "TEXT"),
            ("cost_proxy_train", "REAL"),
            ("cost_proxy_heldout", "REAL"),
            ("cost_proxy_total", "REAL"),
            ("estimated_cost_usd", "REAL"),
            ("estimate_source", "TEXT"),
        ]
        for column_name, column_ddl in desired_columns:
            if column_name in existing:
                continue
            self._conn.execute(
                f"ALTER TABLE candidate_model_transforms ADD COLUMN {column_name} {column_ddl}"
            )


def open_sqlite_run_ledger(
    *,
    program_id: str,
    mode: str,
    run_id: str | None = None,
    ledger_path: str | None = None,
    resume: bool = False,
) -> SQLiteMiproRunLedger:
    resolved_run_id = str(run_id or "").strip()
    resolved_path = str(ledger_path or "").strip()
    if not resolved_run_id and resolved_path:
        resolved_run_id = Path(resolved_path).stem
    if not resolved_run_id:
        resolved_run_id = f"mipro_{uuid4().hex[:12]}"
    if not resolved_path:
        resolved_path = str((_DEFAULT_RUN_ROOT / f"{resolved_run_id}.sqlite").resolve())
    return SQLiteMiproRunLedger(
        run_id=resolved_run_id,
        ledger_path=resolved_path,
        program_id=program_id,
        mode=mode,
        resume=bool(resume),
    )


def load_resume_state(ledger: SQLiteMiproRunLedger) -> MiproRunResumeState:
    return ledger.load_resume_state()


__all__ = [
    "LedgerObservation",
    "LedgerHeldoutSnapshot",
    "LedgerProposerRound",
    "MiproRunResumeState",
    "RunLedgerHeldoutSnapshot",
    "SQLiteMiproRunLedger",
    "load_resume_state",
    "open_sqlite_run_ledger",
]
