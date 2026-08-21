from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

# Live OpenAI-compatible policy is supplied per rollout through request.policy.
try:
    from openai import AsyncOpenAI
except Exception as _openai_err:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore[assignment]
    _OPENAI_IMPORT_ERROR = _openai_err
else:
    _OPENAI_IMPORT_ERROR = None

# Fixed cap for OpenAI-compatible calls inside this service. Adaptive fan-out is
# owned by the Rust GEPA scheduler so run behavior is checkpointed with the run.
POLICY_CONCURRENCY = int(os.environ.get("BANKING77_POLICY_CONCURRENCY", "30"))
POLICY_TIMEOUT_SECONDS = float(os.environ.get("BANKING77_POLICY_TIMEOUT_SECONDS", "20"))
POLICY_RETRIES = int(os.environ.get("BANKING77_POLICY_RETRIES", "1"))
POLICY_RETRY_BACKOFF_SECONDS = float(
    os.environ.get("BANKING77_POLICY_RETRY_BACKOFF_SECONDS", "1.5")
)
ROLLOUT_TIMEOUT_SECONDS = float(
    os.environ.get("BANKING77_ROLLOUT_TIMEOUT_SECONDS", str(POLICY_TIMEOUT_SECONDS + 5))
)
DEFAULT_POLICY_MAX_TOKENS = 16
# USD / 1M tokens. `cost_usd: 0.0` is not a measurement — OpenAI chat usage has
# tokens, not a billed dollar field, so this container must apply list prices.
_POLICY_USD_PER_MILLION: dict[str, tuple[float, float, float]] = {
    # model: (input, cached_input, output)
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    "openai/gpt-4.1-nano": (0.10, 0.025, 0.40),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
}


def _normalize_policy_model(model: str) -> str:
    text = str(model or "").strip()
    if text.lower().startswith("openai/"):
        return text[len("openai/") :]
    return text


def _price_policy_usage(model: str, usage: dict[str, Any]) -> dict[str, Any]:
    out = dict(usage)
    out["model"] = model
    prompt = int(out.get("prompt_tokens") or out.get("input_tokens") or 0)
    completion = int(out.get("completion_tokens") or out.get("output_tokens") or 0)
    cached = int(out.get("cached_prompt_tokens") or 0)
    billed = out.get("cost_usd")
    try:
        billed_f = float(billed) if billed is not None else None
    except (TypeError, ValueError):
        billed_f = None
    if billed_f is not None and billed_f > 0.0:
        out["cost_source"] = "provider_billed"
        return out
    rates = _POLICY_USD_PER_MILLION.get(model) or _POLICY_USD_PER_MILLION.get(
        _normalize_policy_model(model)
    )
    if rates is None:
        out["cost_usd"] = 0.0
        out["cost_source"] = "unpriced" if (prompt or completion) else "no_tokens"
        return out
    input_rate, cached_rate, output_rate = rates
    billable_prompt = max(0, prompt - min(cached, prompt))
    cost = (
        billable_prompt * input_rate / 1_000_000.0
        + min(cached, prompt) * cached_rate / 1_000_000.0
        + completion * output_rate / 1_000_000.0
    )
    out["cost_usd"] = cost
    out["cost_source"] = f"static_price:{_normalize_policy_model(model).lower()}"
    out["cost_pricing"] = {
        "input_usd_per_million": input_rate,
        "cached_input_usd_per_million": cached_rate,
        "output_usd_per_million": output_rate,
    }
    return out
DESKTOP_EVAL_POLICY_REF = {
    "harness": "desktop_eval",
    "config": "banking77_gpt_4_1_nano",
    "model": "openai/gpt-4.1-nano",
    "auth": "openrouter",
}
_openai_clients: dict[tuple[str, str, str], Any] = {}
_openai_semaphore: asyncio.Semaphore | None = None
_RAW_CREDENTIAL_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "authorization",
    "bearer_token",
    "openai_api_key",
    "openrouter_api_key",
    "secret_key",
}


def _find_raw_credential_key(value: Any) -> str | None:
    if isinstance(value, dict):
        for raw_key, raw_value in value.items():
            normalized = str(raw_key).strip().lower().replace("-", "_")
            if normalized in _RAW_CREDENTIAL_KEYS or normalized.endswith("_api_key"):
                return str(raw_key)
            nested = _find_raw_credential_key(raw_value)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for item in value:
            nested = _find_raw_credential_key(item)
            if nested is not None:
                return nested
    return None


def _normalize_policy_enum(value: Any, default: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    return text or default


def _strip_openai_endpoint_suffix(url: str) -> str:
    normalized = url.strip().rstrip("/")
    for suffix in ("/chat/completions", "/responses"):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def _require_policy(payload: dict[str, Any]) -> dict[str, Any]:
    policy = payload.get("policy")
    if not isinstance(policy, dict):
        policy_ref = payload.get("policy_ref") or payload.get("policyRef")
        if not isinstance(policy_ref, dict) or any(
            policy_ref.get(field) != DESKTOP_EVAL_POLICY_REF[field]
            for field in ("harness", "config")
        ):
            raise HTTPException(
                status_code=422,
                detail=(
                    "rollout.policy is required for GEPA optimizer contract v2; "
                    "prepared Desktop evals must use the advertised policy_ref."
                ),
            )
        workshop_route = os.environ.get("WORKSHOP_OPENAI_ROUTE", "").strip()
        # This cookbook launcher runs the service on the host. Workshop emits a
        # Docker-reachable route for true container workers, so translate only
        # its fixed bridge hostname back to loopback before the host process
        # calls the local proxy. The capability path and port remain unchanged.
        workshop_route = workshop_route.replace(
            "://host.docker.internal:", "://127.0.0.1:"
        )
        policy = {
            "provider": "openrouter",
            "model": DESKTOP_EVAL_POLICY_REF["model"],
            "api_family": "chat_completions",
            "inference_url": workshop_route or None,
            "base_url": None if workshop_route else "https://openrouter.ai/api/v1",
            "credential_mode": "workshop_proxy" if workshop_route else "byok",
            "max_tokens": int(os.environ.get("BANKING77_POLICY_MAX_TOKENS", "64")),
            "disable_reasoning": os.environ.get(
                "BANKING77_POLICY_DISABLE_REASONING", "auto"
            ),
        }
    workshop_route = os.environ.get("WORKSHOP_OPENAI_ROUTE", "").strip()
    if workshop_route:
        workshop_route = workshop_route.replace(
            "://host.docker.internal:", "://127.0.0.1:"
        )
        policy = {
            **policy,
            "provider": "openrouter",
            "inference_url": workshop_route,
            "base_url": None,
            "credential_mode": "workshop_proxy",
        }
    raw_key = _find_raw_credential_key(policy.get("config", {}))
    if raw_key is not None:
        raise HTTPException(
            status_code=422,
            detail=f"rollout.policy.config must not carry raw credential field {raw_key!r}.",
        )
    provider = str(policy.get("provider") or "").strip()
    model = str(policy.get("model") or "").strip()
    if not provider or not model:
        raise HTTPException(
            status_code=422,
            detail="rollout.policy.provider and rollout.policy.model are required.",
        )
    api_family = _normalize_policy_enum(policy.get("api_family"), "chat_completions")
    if api_family not in {"chat_completions", "responses"}:
        raise HTTPException(
            status_code=422,
            detail=f"unsupported rollout.policy.api_family: {api_family!r}",
        )
    credential_mode = _normalize_policy_enum(policy.get("credential_mode"), "byok")
    if credential_mode not in {"byok", "proxy", "workshop_proxy"}:
        raise HTTPException(
            status_code=422,
            detail=f"unsupported rollout.policy.credential_mode: {credential_mode!r}",
        )
    disable_reasoning = _normalize_policy_enum(policy.get("disable_reasoning"), "auto")
    if disable_reasoning not in {"auto", "on", "off", "true", "false", "1", "0"}:
        raise HTTPException(
            status_code=422,
            detail=f"unsupported rollout.policy.disable_reasoning: {disable_reasoning!r}",
        )
    raw_base_url = (
        str(policy.get("inference_url") or "").strip()
        if credential_mode in {"proxy", "workshop_proxy"}
        else str(policy.get("base_url") or "").strip()
    )
    if credential_mode in {"proxy", "workshop_proxy"} and not raw_base_url:
        raise HTTPException(
            status_code=422,
            detail="rollout.policy.inference_url is required when credential_mode is proxied.",
        )
    if (
        provider.lower() == "openrouter"
        and credential_mode == "byok"
        and not raw_base_url
    ):
        raise HTTPException(
            status_code=422,
            detail="rollout.policy.base_url is required for provider=openrouter.",
        )
    max_tokens = policy.get("max_tokens", DEFAULT_POLICY_MAX_TOKENS)
    if max_tokens is None:  # optimizer may send max_tokens: null explicitly
        max_tokens = DEFAULT_POLICY_MAX_TOKENS
    try:
        max_tokens = int(max_tokens)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail="rollout.policy.max_tokens must be an integer when set.",
        ) from exc
    if max_tokens <= 0:
        raise HTTPException(
            status_code=422,
            detail="rollout.policy.max_tokens must be positive.",
        )
    return {
        "provider": provider,
        "model": model,
        "api_family": api_family,
        "base_url": (
            _strip_openai_endpoint_suffix(raw_base_url) if raw_base_url else None
        ),
        "credential_mode": credential_mode,
        "max_tokens": max_tokens,
        "disable_reasoning": disable_reasoning,
    }


def _policy_api_key(policy: dict[str, Any]) -> str:
    if policy["credential_mode"] in {"proxy", "workshop_proxy"}:
        return (
            os.environ.get("OPENAI_API_KEY", "").strip()
            or os.environ.get("OPENROUTER_API_KEY", "").strip()
            or "workshop-proxy"
        )
    env_name = (
        "OPENROUTER_API_KEY"
        if policy["provider"].lower() == "openrouter"
        else "OPENAI_API_KEY"
    )
    api_key = os.environ.get(env_name, "").strip()
    if api_key:
        return api_key
    raise HTTPException(
        status_code=503,
        detail=f"{env_name} is not set; rollout.policy credential_mode=byok requires a container env credential.",
    )


def _get_openai_client(policy: dict[str, Any]) -> Any:
    if AsyncOpenAI is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "openai package not installed; install with `pip install openai>=1.0`. "
                f"Original import error: {_OPENAI_IMPORT_ERROR!r}"
            ),
        )
    base_url = policy.get("base_url")
    key = (policy["provider"].lower(), policy["credential_mode"], str(base_url or ""))
    client = _openai_clients.get(key)
    if client is None:
        client_kwargs = {
            "api_key": _policy_api_key(policy),
            "timeout": POLICY_TIMEOUT_SECONDS,
            # GEPA owns retry policy and attempt accounting. Hidden SDK retries
            # made one declared attempt issue three provider requests.
            "max_retries": 0,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        client = AsyncOpenAI(**client_kwargs)
        _openai_clients[key] = client
    return client


def _get_openai_semaphore() -> asyncio.Semaphore:
    """Lazy semaphore creation so it binds to the running event loop."""
    global _openai_semaphore
    if _openai_semaphore is None:
        _openai_semaphore = asyncio.Semaphore(max(1, POLICY_CONCURRENCY))
    return _openai_semaphore


def _is_policy_timeout(error: Exception) -> bool:
    if isinstance(error, TimeoutError):
        return True
    name = type(error).__name__.lower()
    return "timeout" in name or "timedout" in name


def _safe_policy_error(error: Exception) -> dict[str, Any]:
    """Retain actionable provider diagnostics without leaking proxy handles."""
    detail: dict[str, Any] = {"exception_type": type(error).__name__}
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        detail["status_code"] = status_code
    request_id = getattr(error, "request_id", None)
    if isinstance(request_id, str) and request_id.strip():
        detail["request_id"] = request_id.strip()
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        provider_error = body.get("error") if isinstance(body.get("error"), dict) else body
        for key in ("code", "type", "message"):
            value = provider_error.get(key)
            if isinstance(value, str) and value.strip():
                detail[key] = value.strip()
    return detail


def _policy_prefers_chat(policy: dict[str, Any]) -> bool:
    return policy["api_family"] == "chat_completions"


def _policy_retry_delay(attempt: int) -> float:
    return min(POLICY_RETRY_BACKOFF_SECONDS * (2 ** max(0, attempt - 1)), 8.0)


def _policy_chat_extra_body(policy: dict[str, Any]) -> dict[str, Any] | None:
    setting = policy["disable_reasoning"]
    disable_reasoning = setting in {"1", "true", "yes", "on"}
    if setting == "auto":
        disable_reasoning = "openrouter.ai" in str(policy.get("base_url") or "").lower()
    if not disable_reasoning:
        return None
    return {
        "reasoning": {"effort": "none", "exclude": True},
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


try:
    from synth_containers import GEPA_OPTIMIZER_CONTRACT_VERSION
except Exception:
    GEPA_OPTIMIZER_CONTRACT_VERSION = "synth_optimizers.gepa.v2"


TASK_ID = "banking77.intent_classification"

# Sample sizes: pick a balanced random subset from each split, then expose
# stable 0..N-1 seed indices for the optimizer inside that process.
TRAIN_SAMPLE = int(os.environ.get("BANKING77_TRAIN_SAMPLE", "24"))
TEST_SAMPLE = int(os.environ.get("BANKING77_TEST_SAMPLE", "200"))
TRAIN_SHUFFLE_SEED = int(os.environ.get("BANKING77_TRAIN_SHUFFLE_SEED", "1009"))
TEST_SHUFFLE_SEED = int(os.environ.get("BANKING77_TEST_SHUFFLE_SEED", "2003"))


def _load_banking77_rows() -> tuple[list[str], list[dict[str, Any]]]:
    """Load deterministic mixed PolyAI/banking77 train+test slices."""
    from datasets import load_dataset

    ds = load_dataset("PolyAI/banking77", trust_remote_code=True)
    label_names: list[str] = list(ds["train"].features["label"].names)

    def mixed_rows(
        split_name: str, sample_size: int, shuffle_seed: int
    ) -> list[dict[str, Any]]:
        split = ds[split_name]
        grouped: dict[int, list[int]] = {idx: [] for idx in range(len(label_names))}
        for source_index, ex in enumerate(split):
            grouped[int(ex["label"])].append(source_index)
        rng = random.Random(shuffle_seed)
        for indices in grouped.values():
            rng.shuffle(indices)
        label_order = list(grouped)
        rng.shuffle(label_order)
        selected: list[int] = []
        while len(selected) < min(sample_size, len(split)):
            progressed = False
            for label_idx in label_order:
                if grouped[label_idx]:
                    selected.append(grouped[label_idx].pop())
                    progressed = True
                    if len(selected) >= sample_size:
                        break
            if not progressed:
                break
        rng.shuffle(selected)
        rows: list[dict[str, Any]] = []
        for seed, source_index in enumerate(selected):
            ex = split[source_index]
            rows.append(
                {
                    "seed": seed,
                    "source_index": source_index,
                    "split": split_name,
                    "text": str(ex["text"]),
                    "label": label_names[int(ex["label"])],
                }
            )
        return rows

    rows = []
    rows.extend(mixed_rows("train", TRAIN_SAMPLE, TRAIN_SHUFFLE_SEED))
    rows.extend(mixed_rows("test", TEST_SAMPLE, TEST_SHUFFLE_SEED))
    return label_names, rows


LABELS, ROWS = _load_banking77_rows()
_LABEL_BY_LOWER = {label.lower(): label for label in LABELS}
_LABEL_BY_SIMPLIFIED = {
    "".join(ch for ch in label.lower() if ch.isalnum() or ch == "_"): label
    for label in LABELS
}

DEFAULT_STAGE2_SYSTEM = (
    "Classify the customer banking query into exactly one Banking77 intent. "
    "Return exactly one label from the allowed label list, preserving the label's spelling, "
    "underscores, capitalization, and punctuation. Use the full query, not one keyword. "
    "Prefer the label for the user's concrete banking action, status, or problem: separate "
    "physical-card ordering from delivery timing, virtual-card creation from virtual-card "
    "problems, card payments from cash withdrawals, top-ups from incoming transfers, "
    "pending from failed/declined/reverted, passcodes from card PINs, and phone loss from "
    "card compromise. Return only the label."
)

BANKING77_LABEL_GUIDANCE = {
    "output_contract": [
        "Return one canonical label exactly as it appears in the allowed list.",
        "Some dataset labels intentionally contain capitalization or punctuation; preserve them.",
        "Do not rewrite labels into normalized lowercase if the allowed label differs.",
    ],
    "high_value_boundaries": [
        {
            "boundary": "card_delivery_estimate vs order_physical_card vs get_physical_card",
            "rule": "Delivery timing or choosing a delivery day is card_delivery_estimate; asking to order/request a physical card is order_physical_card; asking how to obtain one generally is get_physical_card.",
        },
        {
            "boundary": "getting_virtual_card vs get_disposable_virtual_card vs virtual_card_not_working",
            "rule": "Getting a normal virtual card is getting_virtual_card; one-time disposable virtual cards are get_disposable_virtual_card; an existing virtual card failing is virtual_card_not_working.",
        },
        {
            "boundary": "pending_* vs failed_* vs declined_* vs reverted_*",
            "rule": "Pending means delayed or not completed yet; failed means the attempted operation did not complete; declined is an explicit refusal; reverted means a completed-looking top-up/payment was reversed.",
        },
        {
            "boundary": "card payment vs cash withdrawal",
            "rule": "Purchases/payments with a card use card_payment labels; ATM/cash-machine withdrawal issues use cash_withdrawal labels.",
        },
        {
            "boundary": "passcode_forgotten vs change_pin vs pin_blocked",
            "rule": "App passcode reset is passcode_forgotten; changing a card PIN is change_pin; too many wrong PIN attempts is pin_blocked.",
        },
        {
            "boundary": "lost_or_stolen_phone vs compromised_card vs lost_or_stolen_card",
            "rule": "Lost phone or app access on a new device is lost_or_stolen_phone; stolen physical card is lost_or_stolen_card; unauthorized card/account use is compromised_card.",
        },
        {
            "boundary": "receiving_money vs transfer_into_account vs topping_up_by_card",
            "rule": "Receiving external money or salary is receiving_money; bank transfer into the account is transfer_into_account; adding money by debit/credit card is topping_up_by_card.",
        },
        {
            "boundary": "supported_cards_and_currencies vs fiat_currency_support",
            "rule": "Supported cards/currencies for adding money is supported_cards_and_currencies; account or holding support for fiat currencies is fiat_currency_support.",
        },
    ],
}

BANKING77_PROPOSER_HINTS = {
    "task_output_space": "finite_intent_label",
    "literal_training_targets": "allow",
    "proposal_goal": (
        "Infer reusable Banking77 label-boundary rules from rollout traces, mistakes, and guard "
        "wins. Concrete query-to-label examples are valid when they teach a reusable distinction."
    ),
    "trace_review": [
        "Compare expected labels, predicted labels, rationales, and trace summaries for losses.",
        "Look for repeated boundary confusions rather than only isolated one-off examples.",
        "Preserve the exact allowed-label output contract.",
    ],
}


app = FastAPI(title="banking77-gepa-container")
_ASYNC_ROLLOUTS: dict[str, dict[str, Any]] = {}
_ASYNC_ROLLOUT_LOCK = asyncio.Lock()
_STREAM_LOCK = asyncio.Lock()
_STREAMS: dict[str, list[dict[str, Any]]] = {}
_TERMINAL_ROLLOUT_STATUSES = {"completed", "failed", "cancelled"}
_STREAM_SCHEMA = "synth.trace-stream-event.v1"
_ROLLOUT_RECORD_SCHEMA = "synth.banking77-rollout-record.v1"


def _stream_id(rollout_id: str) -> str:
    return f"stream:{rollout_id}"


def _stream_descriptor(rollout_id: str) -> dict[str, Any]:
    return {
        "schema": "synth.rollout.stream.v1",
        "id": _stream_id(rollout_id),
        "stream.id": _stream_id(rollout_id),
        "transports": {
            "poll": {"url": f"/rollouts/{rollout_id}/events"},
            "sse": {"url": f"/rollouts/{rollout_id}/events/sse"},
            "websocket": None,
        },
        "cursor": {"kind": "sequence", "producer_kind": None},
        "reward": {"url": f"/reward?rollout_id={rollout_id}"},
        "auth": {"mode": "none"},
        "retention": "run",
    }


def _stream_root() -> Path:
    configured = os.environ.get("BANKING77_STREAM_ROOT", "").strip()
    return Path(configured) if configured else Path.cwd() / ".banking77-streams"


def _stream_path(rollout_id: str) -> Path:
    digest = hashlib.sha256(rollout_id.encode("utf-8")).hexdigest()
    return _stream_root() / f"{digest}.jsonl"


def _rollout_record_path(rollout_id: str) -> Path:
    digest = hashlib.sha256(rollout_id.encode("utf-8")).hexdigest()
    return _stream_root() / "records" / f"{digest}.json"


def _load_rollout_record(rollout_id: str) -> dict[str, Any] | None:
    cached = _ASYNC_ROLLOUTS.get(rollout_id)
    if cached is not None:
        return dict(cached)
    path = _rollout_record_path(rollout_id)
    if not path.is_file():
        return None
    envelope = json.loads(path.read_text(encoding="utf-8"))
    if (
        envelope.get("schema") != _ROLLOUT_RECORD_SCHEMA
        or envelope.get("rollout_id") != rollout_id
        or not isinstance(envelope.get("record"), dict)
    ):
        raise RuntimeError(f"invalid rollout record identity at {path}")
    record = dict(envelope["record"])
    _ASYNC_ROLLOUTS[rollout_id] = record
    return dict(record)


def _persist_rollout_record(record: dict[str, Any]) -> None:
    rollout_id = str(record.get("rollout_id") or "").strip()
    if not rollout_id:
        raise RuntimeError("rollout record is missing rollout_id")
    path = _rollout_record_path(rollout_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema": _ROLLOUT_RECORD_SCHEMA,
        "rollout_id": rollout_id,
        "record": record,
    }
    encoded = json.dumps(envelope, sort_keys=True, separators=(",", ":"), default=str)
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _store_rollout_record(record: dict[str, Any]) -> None:
    rollout_id = str(record.get("rollout_id") or "").strip()
    _persist_rollout_record(record)
    _ASYNC_ROLLOUTS[rollout_id] = dict(record)


def _load_stream(rollout_id: str) -> list[dict[str, Any]]:
    cached = _STREAMS.get(rollout_id)
    if cached is not None:
        return cached
    path = _stream_path(rollout_id)
    items: list[dict[str, Any]] = []
    expected_sequence = 1
    if path.is_file():
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            item = json.loads(line)
            if (
                item.get("schema") != _STREAM_SCHEMA
                or item.get("rollout_id") != rollout_id
            ):
                raise RuntimeError(
                    f"invalid rollout stream identity at {path}:{line_number}"
                )
            sequence = item.get("sequence")
            if sequence is None:
                if (
                    item.get("kind") != "stream.subscribed"
                    or item.get("control") is not True
                ):
                    raise RuntimeError(
                        f"invalid rollout control record at {path}:{line_number}"
                    )
            else:
                if sequence != expected_sequence or item.get("control") is not False:
                    raise RuntimeError(
                        f"rollout stream sequence gap at {path}:{line_number}"
                    )
                expected_sequence += 1
            items.append(item)
    _STREAMS[rollout_id] = items
    return items


def _persist_stream_item(rollout_id: str, item: dict[str, Any]) -> None:
    path = _stream_path(rollout_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(item, sort_keys=True, separators=(",", ":"), default=str)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


async def _ensure_stream(rollout_id: str) -> dict[str, Any]:
    async with _STREAM_LOCK:
        items = _load_stream(rollout_id)
        if not any(item.get("kind") == "stream.subscribed" for item in items):
            subscribed = {
                "schema": _STREAM_SCHEMA,
                "kind": "stream.subscribed",
                "event_id": "stream.subscribed",
                "sequence": None,
                "control": True,
                "slot": "stream",
                "stream_id": _stream_id(rollout_id),
                "rollout_id": rollout_id,
                "ts": _now(),
                "ready": True,
                "payload": {
                    "type": "stream.subscribed",
                    "stream.id": _stream_id(rollout_id),
                    "rollout_id": rollout_id,
                    "next_sequence": 1,
                    "ready": True,
                },
            }
            _persist_stream_item(rollout_id, subscribed)
            items.append(subscribed)
    return _stream_descriptor(rollout_id)


async def _append_stream_event(
    rollout_id: str, kind: str, payload: dict[str, Any]
) -> dict[str, Any]:
    async with _STREAM_LOCK:
        items = _load_stream(rollout_id)
        high_water = max(
            (
                int(item["sequence"])
                for item in items
                if item.get("sequence") is not None
            ),
            default=0,
        )
        sequence = high_water + 1
        item = {
            "schema": _STREAM_SCHEMA,
            "kind": kind,
            "event_id": sequence,
            "sequence": sequence,
            "control": False,
            "slot": "stream",
            "stream_id": _stream_id(rollout_id),
            "rollout_id": rollout_id,
            "run_id": rollout_id,
            "lane": rollout_id,
            "ts": _now(),
            "payload": dict(payload),
        }
        _persist_stream_item(rollout_id, item)
        items.append(item)
        return item


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/metadata")
@app.get("/info")
async def metadata() -> dict[str, Any]:
    return {
        "runtime": {
            "runtime_id": "banking77_gepa_live",
            "name": "Banking77 GEPA (live OpenAI-compatible policy)",
            "description": "Public prompt-optimizer cookbook for Banking77 with a live OpenAI-compatible policy model.",
        },
        "capabilities": {
            "protocol": GEPA_OPTIMIZER_CONTRACT_VERSION,
            "operations": {
                "rollouts.prepare": True,
                "rollouts.start_prepared": True,
                "rollouts.get": True,
                "rollouts.poll": True,
                "reward.get": True,
                "trace_v5.capture": False,
            },
            "policy_refs": [DESKTOP_EVAL_POLICY_REF],
            "contract_version": "container_contract.v1",
            "rollout_modes": ["blocking", "async"],
            "metadata": {"policy_ready": True},
        },
        "metadata": {
            "optimizer_contracts": {
                "gepa": {
                    "version": GEPA_OPTIMIZER_CONTRACT_VERSION,
                    "program_route": "/program",
                    "taskset_route": "/taskset",
                    "taskset_tasks_route": "/taskset/tasks",
                    "dataset_route": "/dataset",
                    "dataset_rows_route": "/dataset/rows",
                    "rollout_route": "/rollout",
                }
            },
            "task_catalog_route": "/task_catalog",
        },
    }


@app.get("/task_catalog")
async def task_catalog() -> dict[str, Any]:
    return {
        "catalog_id": "banking77_public_rows:catalog",
        "tasks": [
            {
                "task_id": TASK_ID,
                "task_name": "Banking77 intent classification",
                "task_family": "banking77",
                "description": "Classify a customer banking question into one Banking77 label.",
                "benchmark": "PolyAI/banking77",
                "metadata": {
                    "primary_metric": "classification_accuracy",
                    "label_count": len(LABELS),
                },
            }
        ],
        "instances": [
            {
                "task_instance_id": f"banking77:{row['split']}:{row['seed']}",
                "task_id": TASK_ID,
                "split": row["split"],
                "tags": ["banking77", row["split"], row["label"]],
                "metadata": {
                    "seed": row["seed"],
                    "source_index": row["source_index"],
                    "input": row["text"],
                    "output_label": row["label"],
                },
            }
            for row in ROWS
        ],
        "metadata": {
            "dataset_id": "banking77_public_rows",
            "instance_count": len(ROWS),
            "filterable_fields": [
                "split",
                "tags",
                "metadata.output_label",
                "metadata.seed",
                "metadata.source_index",
            ],
        },
    }


@app.get("/task_info")
async def task_info() -> dict[str, Any]:
    return {
        "task": {
            "task_id": TASK_ID,
            "name": "Banking77 intent classification",
            "description": "Classify a customer banking question into one Banking77 label.",
        },
        "output_space": {
            "kind": "finite_intent_label",
            "label_count": len(LABELS),
            "labels": LABELS,
            "contract": "Return exactly one canonical label from the allowed list.",
        },
        "dataset": {
            "dataset_id": "banking77_public_rows",
            "visible_splits": ["train", "test"],
            "default_split": "train",
            "row_count": len(ROWS),
            "sampling": {
                "train_sample": TRAIN_SAMPLE,
                "test_sample": TEST_SAMPLE,
                "train_shuffle_seed": TRAIN_SHUFFLE_SEED,
                "test_shuffle_seed": TEST_SHUFFLE_SEED,
                "method": "balanced_random_per_label",
            },
        },
        "proposer_hints": BANKING77_PROPOSER_HINTS,
        "metadata": {
            "primary_metric": "classification_accuracy",
            "labels": LABELS,
            "label_guidance": BANKING77_LABEL_GUIDANCE,
            "proposer_hints": BANKING77_PROPOSER_HINTS,
        },
    }


@app.get("/program")
async def program() -> dict[str, Any]:
    return {
        "version": "prompt_program.v1",
        "program_id": "banking77_single_stage_gepa",
        "modules": [
            {
                "module_id": "stage2_system",
                "role": "system",
                "content": DEFAULT_STAGE2_SYSTEM,
                "mutable": True,
                "candidate_field": "stage2_system",
                "template_variables": [],
            }
        ],
        "target_modules": [
            {
                "module_id": "stage2_system",
                "candidate_field": "stage2_system",
                "objective": "classification_accuracy",
            }
        ],
        "seed_candidate": {"stage2_system": DEFAULT_STAGE2_SYSTEM},
        "rollout_overlay_schema": {"candidate_fields": ["stage2_system"]},
        "metadata": {
            "task_id": TASK_ID,
            "dataset_id": "banking77_public_rows",
            "labels": LABELS,
            "label_guidance": BANKING77_LABEL_GUIDANCE,
            "proposer_hints": BANKING77_PROPOSER_HINTS,
        },
    }


@app.get("/dataset")
async def dataset() -> dict[str, Any]:
    return {
        "dataset_id": "banking77_public_rows",
        "splits": {
            "train": sum(1 for row in ROWS if row["split"] == "train"),
            "test": sum(1 for row in ROWS if row["split"] == "test"),
        },
        "sampling": {
            "train_sample": TRAIN_SAMPLE,
            "test_sample": TEST_SAMPLE,
            "train_shuffle_seed": TRAIN_SHUFFLE_SEED,
            "test_shuffle_seed": TEST_SHUFFLE_SEED,
            "method": "balanced_random_per_label",
        },
        "labels": LABELS,
        "label_guidance": BANKING77_LABEL_GUIDANCE,
    }


@app.post("/dataset/rows")
async def dataset_rows(request: Request) -> dict[str, Any]:
    payload = await request.json()
    split = str(payload.get("split") or "train")
    seeds = [int(seed) for seed in payload.get("seeds") or []]
    selected = []
    for seed in seeds:
        selected.append(_row_for_seed(split=split, seed=seed))
    return {"rows": selected}


@app.get("/taskset")
async def taskset() -> dict[str, Any]:
    """Expose the current GEPA v2 taskset contract over the pinned row sample."""
    return {
        "taskset_id": "banking77_public_rows:v1",
        "splits": {
            "train": sum(1 for row in ROWS if row["split"] == "train"),
            "test": sum(1 for row in ROWS if row["split"] == "test"),
        },
        "labels": LABELS,
        "source": "PolyAI/banking77",
        "metadata": {
            "train_shuffle_seed": TRAIN_SHUFFLE_SEED,
            "test_shuffle_seed": TEST_SHUFFLE_SEED,
        },
    }


@app.post("/taskset/tasks")
async def taskset_tasks(request: Request) -> dict[str, Any]:
    payload = await request.json()
    split = str(payload.get("split") or "").strip()
    if split not in {"train", "test"}:
        raise HTTPException(status_code=422, detail="split must be train or test")
    raw_task_ids = payload.get("task_ids")
    if not isinstance(raw_task_ids, list) or not raw_task_ids:
        raise HTTPException(status_code=422, detail="task_ids must be a non-empty list")
    tasks = []
    for raw_task_id in raw_task_ids:
        task_id = str(raw_task_id).strip()
        prefix = f"{split}:"
        if not task_id.startswith(prefix):
            raise HTTPException(
                status_code=422,
                detail=f"task_id {task_id!r} must start with {prefix!r}",
            )
        try:
            seed = int(task_id[len(prefix) :])
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"task_id {task_id!r} must end in an integer seed",
            ) from exc
        row = _row_for_seed(split=split, seed=seed)
        tasks.append(
            {
                "task_id": task_id,
                "task_instance_id": f"banking77:{split}:{seed}",
                "split": split,
                "seed": seed,
                "text": row["text"],
                "label": row["label"],
                "source_index": row["source_index"],
            }
        )
    return {"tasks": tasks, "metadata": {"taskset_id": "banking77_public_rows:v1"}}


@app.post("/rollout")
@app.post("/rollouts")
async def rollout(request: Request) -> dict[str, Any]:
    payload = await request.json()
    telemetry = (
        payload.get("telemetry") if isinstance(payload.get("telemetry"), dict) else {}
    )
    transport = str(telemetry.get("transport") or "poll").strip().lower()
    if transport == "auto":
        raise HTTPException(
            status_code=422, detail="telemetry.transport=auto is forbidden"
        )
    if transport not in {"", "poll", "sse"}:
        raise HTTPException(
            status_code=422, detail=f"unsupported telemetry transport: {transport}"
        )
    rollout_id = str(
        payload.get("rollout_id")
        or payload.get("trace_correlation_id")
        or f"rollout_{uuid.uuid4().hex[:12]}"
    )
    payload = {**payload, "rollout_id": rollout_id}
    stream = await _ensure_stream(rollout_id)
    submission_mode = str(payload.get("submission_mode") or "sync").strip().lower()
    async with _ASYNC_ROLLOUT_LOCK:
        existing = _load_rollout_record(rollout_id)
    if existing is not None:
        return existing
    if submission_mode == "sync":
        now = _now()
        running = {
            "rollout_id": rollout_id,
            "status": "running",
            "success_status": "running",
            "status_detail": "running",
            "task_id": TASK_ID,
            "seed": int(payload.get("seed") or 0),
            "summary": {},
            "usage": {},
            "metadata": {"submission_mode": "sync"},
            "stream": stream,
            "created_at": now,
            "updated_at": now,
        }
        async with _ASYNC_ROLLOUT_LOCK:
            _store_rollout_record(running)
        try:
            completed = await _execute_rollout_payload_with_timeout(payload)
        except Exception as exc:
            failed = {
                **running,
                "status": "failed",
                "success_status": "failed",
                "status_detail": str(getattr(exc, "detail", exc)),
                "summary": {"status_detail": str(getattr(exc, "detail", exc))},
                "updated_at": _now(),
            }
            async with _ASYNC_ROLLOUT_LOCK:
                _store_rollout_record(failed)
            raise
        completed = {
            **completed,
            "stream": stream,
            "metadata": {
                **dict(completed.get("metadata") or {}),
                "submission_mode": "sync",
            },
        }
        async with _ASYNC_ROLLOUT_LOCK:
            _store_rollout_record(completed)
        return completed
    if submission_mode != "async":
        raise HTTPException(
            status_code=400,
            detail="submission_mode must be one of: sync, async",
        )
    now = _now()
    queued = {
        "rollout_id": rollout_id,
        "status": "queued",
        "success_status": "pending",
        "status_detail": "queued",
        "task_id": TASK_ID,
        "seed": int(payload.get("seed") or 0),
        "summary": {},
        "usage": {},
        "metadata": {"submission_mode": "async"},
        "stream": stream,
        "created_at": now,
        "updated_at": now,
    }
    async with _ASYNC_ROLLOUT_LOCK:
        _store_rollout_record(queued)
    asyncio.create_task(_complete_async_rollout(rollout_id, payload))
    return queued


@app.post("/rollouts/prepare")
async def prepare_rollout(request: Request) -> dict[str, Any]:
    payload = await request.json()
    telemetry = (
        payload.get("telemetry") if isinstance(payload.get("telemetry"), dict) else {}
    )
    transport = str(telemetry.get("transport") or "poll").strip().lower()
    if transport == "auto":
        raise HTTPException(
            status_code=422, detail="telemetry.transport=auto is forbidden"
        )
    if transport not in {"", "poll", "sse"}:
        raise HTTPException(
            status_code=422, detail=f"unsupported telemetry transport: {transport}"
        )
    rollout_id = str(
        payload.get("rollout_id")
        or payload.get("trace_correlation_id")
        or f"rollout_{uuid.uuid4().hex[:12]}"
    )
    return {
        "rollout_id": rollout_id,
        "status": "prepared",
        "stream": await _ensure_stream(rollout_id),
    }


@app.get("/rollouts/{rollout_id}/state")
async def rollout_state(rollout_id: str) -> dict[str, Any]:
    return await _async_rollout_record(rollout_id)


@app.get("/rollouts/{rollout_id}")
async def rollout_record(rollout_id: str) -> dict[str, Any]:
    return await _async_rollout_record(rollout_id)


@app.get("/rollouts/{rollout_id}/events")
async def rollout_events(rollout_id: str, after: int = 0) -> dict[str, Any]:
    if after < 0:
        raise HTTPException(status_code=422, detail="after must be non-negative")
    async with _STREAM_LOCK:
        items = list(_load_stream(rollout_id))
    if not items:
        raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
    high_water = max(
        (int(item["sequence"]) for item in items if item.get("sequence") is not None),
        default=0,
    )
    visible = [
        item
        for item in items
        if (item.get("sequence") is None and after == 0)
        or (item.get("sequence") is not None and int(item["sequence"]) > after)
    ]
    return {
        "rollout_id": rollout_id,
        "stream.id": _stream_id(rollout_id),
        "cursor": {"kind": "sequence", "after": after, "high_water": high_water},
        "events": visible,
    }


@app.get("/rollouts/{rollout_id}/events/sse")
async def rollout_events_sse(rollout_id: str) -> StreamingResponse:
    await _ensure_stream(rollout_id)

    async def event_source():
        delivered: set[str] = set()
        while True:
            async with _STREAM_LOCK:
                items = list(_load_stream(rollout_id))
            for item in items:
                identity = str(item.get("event_id"))
                if identity in delivered:
                    continue
                delivered.add(identity)
                yield (
                    f"event: {item.get('kind', 'message')}\n"
                    f"data: {json.dumps(item, separators=(',', ':'), default=str)}\n\n"
                )
            async with _ASYNC_ROLLOUT_LOCK:
                current = _load_rollout_record(rollout_id)
            if (
                current is not None
                and str(current.get("status") or "") in _TERMINAL_ROLLOUT_STATUSES
            ):
                break
            await asyncio.sleep(0.1)

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/reward")
async def reward(rollout_id: str) -> dict[str, Any]:
    current = await _async_rollout_record(rollout_id)
    value = current.get("summary", {}).get("outcome_reward")
    terminal = str(current.get("status") or "") in _TERMINAL_ROLLOUT_STATUSES
    if not terminal:
        return {
            "execution_id": f"eval_{rollout_id}",
            "rollout_id": rollout_id,
            "status": "running",
            "reward": None,
            "node_results": [],
        }
    return {
        "execution_id": f"eval_{rollout_id}",
        "rollout_id": rollout_id,
        "status": "scored" if value is not None else "absent",
        "reward": value,
        "node_results": [
            {
                "node_id": "classification_accuracy",
                "kind": "env_reward",
                "authority": "environment",
                "status": "scored" if value is not None else "skipped",
                "value": value,
                "evidence_refs": [{"kind": "rollout", "id": rollout_id}],
            }
        ],
    }


@app.post("/reward")
async def reward_post(request: Request) -> dict[str, Any]:
    payload = await request.json()
    rollout_id = str(payload.get("rollout_id") or payload.get("rolloutId") or "").strip()
    if not rollout_id:
        raise HTTPException(status_code=422, detail="reward requires rollout_id")
    return await reward(rollout_id)


@app.post("/rollouts/{rollout_id}/terminate")
async def terminate_rollout(rollout_id: str, request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    reason = str(payload.get("reason") or "terminated")
    async with _ASYNC_ROLLOUT_LOCK:
        current = _load_rollout_record(rollout_id)
        if current is None:
            raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
        if str(current.get("status") or "") not in _TERMINAL_ROLLOUT_STATUSES:
            now = _now()
            current = {
                **current,
                "status": "cancelled",
                "success_status": "cancelled",
                "status_detail": reason,
                "updated_at": now,
                "metadata": {
                    **dict(current.get("metadata") or {}),
                    "termination": {"reason": reason},
                },
            }
            _store_rollout_record(current)
        return dict(current)


async def _execute_rollout_payload(payload: dict[str, Any]) -> dict[str, Any]:
    policy = _require_policy(payload)
    rollout_id = str(payload.get("rollout_id") or f"rollout_{uuid.uuid4().hex[:12]}")
    await _ensure_stream(rollout_id)
    row = (
        payload.get("dataset_row")
        if isinstance(payload.get("dataset_row"), dict)
        else None
    )
    if not row and isinstance(payload.get("task"), dict):
        row = payload["task"]
    if not row:
        row = _row_for_seed(
            split=str(payload.get("split") or "train"),
            seed=int(payload.get("seed") or 0),
        )
    candidate = (
        payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
    )
    system_prompt = str(candidate.get("stage2_system") or DEFAULT_STAGE2_SYSTEM)
    await _append_stream_event(
        rollout_id,
        "trace.opened",
        {"trace_id": f"trace:{rollout_id}", "task_id": TASK_ID},
    )
    await _append_stream_event(
        rollout_id,
        "policy.session.opened",
        {"session_id": f"policy:{rollout_id}", "model": policy["model"]},
    )
    await _append_stream_event(
        rollout_id,
        "span.llm.opened",
        {"span_id": f"llm:{rollout_id}:1", "model": policy["model"]},
    )
    # Direct await on AsyncOpenAI; concurrency capped inside _predict_label
    # via a module-level asyncio.Semaphore(POLICY_CONCURRENCY).
    try:
        prediction, usage = await _predict_label(
            str(row.get("text") or ""),
            policy=policy,
            system_prompt=system_prompt,
        )
    except Exception as exc:
        await _append_stream_event(
            rollout_id,
            "span.llm.closed",
            {
                "span_id": f"llm:{rollout_id}:1",
                "status": "failed",
                "error_type": type(exc).__name__,
            },
        )
        await _append_stream_event(
            rollout_id,
            "policy.session.closed",
            {"session_id": f"policy:{rollout_id}", "status": "failed"},
        )
        await _append_stream_event(
            rollout_id,
            "trace.closed",
            {"trace_id": f"trace:{rollout_id}", "status": "failed"},
        )
        raise
    expected = str(row.get("label") or "")
    reward = 1.0 if prediction == expected else 0.0
    await _append_stream_event(
        rollout_id,
        "data",
        {
            "span_id": f"llm:{rollout_id}:1",
            "kind": "classification.result",
            "prediction": prediction,
            "reward": reward,
        },
    )
    await _append_stream_event(
        rollout_id,
        "span.llm.closed",
        {"span_id": f"llm:{rollout_id}:1", "status": "completed", "usage": usage},
    )
    await _append_stream_event(
        rollout_id,
        "policy.session.closed",
        {"session_id": f"policy:{rollout_id}", "status": "completed"},
    )
    await _append_stream_event(
        rollout_id,
        "trace.sealing",
        {"trace_id": f"trace:{rollout_id}"},
    )
    await _append_stream_event(
        rollout_id,
        "trace.closed",
        {"trace_id": f"trace:{rollout_id}", "status": "completed"},
    )
    now = _now()
    return {
        "rollout_id": rollout_id,
        "status": "completed",
        "success_status": "succeeded" if reward > 0 else "failed",
        "task_id": TASK_ID,
        "seed": int(row.get("seed") or 0),
        "reward_info": {
            "outcome_reward": reward,
            "event_rewards": [reward],
            "details": {
                "prediction": prediction,
                "expected": expected,
                "system_prompt_len": len(system_prompt),
                "policy_model": policy["model"],
            },
        },
        "summary": {
            "outcome_reward": reward,
            "prediction": prediction,
            "expected": expected,
        },
        "usage": _price_policy_usage(policy["model"], usage),
        "trace": {
            "event_history": [
                {"type": "input", "text": row.get("text")},
                {"type": "prediction", "label": prediction},
            ],
            "metadata": {"label": expected},
        },
        "metadata": {"candidate": candidate},
        "stream": _stream_descriptor(rollout_id),
        "created_at": now,
        "updated_at": now,
    }


async def _execute_rollout_payload_with_timeout(
    payload: dict[str, Any],
) -> dict[str, Any]:
    row = (
        payload.get("dataset_row")
        if isinstance(payload.get("dataset_row"), dict)
        else None
    )
    example_id = str(
        (row or {}).get("example_id") or payload.get("trace_correlation_id") or "-"
    )
    policy = payload.get("policy") if isinstance(payload.get("policy"), dict) else {}
    policy_model = str(policy.get("model") or "unknown")
    api_family = str(policy.get("api_family") or "unknown")
    try:
        return await asyncio.wait_for(
            _execute_rollout_payload(payload),
            timeout=ROLLOUT_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=(
                f"rollout request timed out after {ROLLOUT_TIMEOUT_SECONDS:.1f}s "
                f"for example_id={example_id}; policy_model={policy_model} "
                f"api_family={api_family} policy_timeout={POLICY_TIMEOUT_SECONDS:.1f}s "
                f"policy_retries={POLICY_RETRIES} policy_concurrency={POLICY_CONCURRENCY}"
            ),
        ) from exc


async def _async_rollout_record(rollout_id: str) -> dict[str, Any]:
    async with _ASYNC_ROLLOUT_LOCK:
        current = _load_rollout_record(rollout_id)
    if current is None:
        raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
    return dict(current)


async def _complete_async_rollout(rollout_id: str, payload: dict[str, Any]) -> None:
    async with _ASYNC_ROLLOUT_LOCK:
        current = _load_rollout_record(rollout_id)
        if current is None or str(current.get("status") or "") == "cancelled":
            return
        _store_rollout_record(
            {
                **current,
                "status": "running",
                "success_status": "running",
                "status_detail": "running",
                "updated_at": _now(),
            }
        )
    try:
        completed = await _execute_rollout_payload_with_timeout(payload)
    except Exception as exc:
        completed = {
            "rollout_id": rollout_id,
            "status": "failed",
            "success_status": "failed",
            "status_detail": str(exc),
            "task_id": TASK_ID,
            "seed": int(payload.get("seed") or 0),
            "summary": {"status_detail": str(exc)},
            "usage": {},
            "metadata": {"submission_mode": "async"},
            "created_at": _now(),
            "updated_at": _now(),
        }
    async with _ASYNC_ROLLOUT_LOCK:
        current = _load_rollout_record(rollout_id)
        if current is None or str(current.get("status") or "") == "cancelled":
            return
        completed = {
            **completed,
            "metadata": {
                **dict(completed.get("metadata") or {}),
                "submission_mode": "async",
            },
        }
        _store_rollout_record(completed)


def _row_for_seed(*, split: str, seed: int) -> dict[str, Any]:
    rows = [row for row in ROWS if row["split"] == split]
    if not rows:
        rows = list(ROWS)
    match = next((row for row in rows if int(row["seed"]) == int(seed)), None)
    row = match or rows[int(seed) % len(rows)]
    result = dict(row)
    result.setdefault(
        "example_id", f"{result.get('split', split)}:{result.get('seed', seed)}"
    )
    return result


async def _predict_label(
    text: str,
    *,
    policy: dict[str, Any],
    system_prompt: str,
) -> tuple[str, dict[str, int]]:
    """Call the live policy model. Returns (predicted_label, token_usage).

    Uses AsyncOpenAI + a module-level Semaphore so the container only ever
    has `POLICY_CONCURRENCY` OpenAI calls in flight at once. Lets the
    container accept any number of concurrent /rollout requests without
    overrunning OpenAI's per-key connection pool.
    """
    client = _get_openai_client(policy)
    semaphore = _get_openai_semaphore()
    user_content = (
        f"Customer query:\n{text}\n\n"
        f"Complete allowed Banking77 label list ({len(LABELS)} labels total). "
        "Return EXACTLY one label from this list as written, no other text:\n"
        + "\n".join(f"- {label}" for label in LABELS)
    )
    # Deterministic policy: temperature=0 so identical (seed, candidate)
    # pairs produce byte-identical predictions across both stacks.
    async with semaphore:
        if _policy_prefers_chat(policy):
            last_error: Exception | None = None
            for attempt in range(1, POLICY_RETRIES + 1):
                try:
                    extra_body = _policy_chat_extra_body(policy)
                    request_kwargs = {
                        "model": policy["model"],
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content},
                        ],
                        "temperature": 0,
                        "max_tokens": policy["max_tokens"],
                    }
                    if extra_body is not None:
                        request_kwargs["extra_body"] = extra_body
                    resp = await asyncio.wait_for(
                        client.chat.completions.create(**request_kwargs),
                        timeout=POLICY_TIMEOUT_SECONDS,
                    )
                    break
                except Exception as chat_error:
                    last_error = chat_error
                    if attempt >= POLICY_RETRIES or not _is_policy_timeout(chat_error):
                        status_code = 504 if _is_policy_timeout(chat_error) else 502
                        provider_error = _safe_policy_error(chat_error)
                        raise HTTPException(
                            status_code=status_code,
                            detail=(
                                f"Policy model {policy['model']!r} failed through Chat Completions API "
                                f"after {attempt}/{POLICY_RETRIES} attempts, "
                                f"timeout={POLICY_TIMEOUT_SECONDS:.1f}s; "
                                f"provider_error={json.dumps(provider_error, sort_keys=True)}"
                            ),
                        ) from chat_error
                    await asyncio.sleep(_policy_retry_delay(attempt))
            else:
                raise HTTPException(
                    status_code=504,
                    detail=f"Policy model {policy['model']!r} failed: {last_error!r}",
                )
            raw = (resp.choices[0].message.content or "").strip()
            usage = {
                "prompt_tokens": int(getattr(resp.usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(
                    getattr(resp.usage, "completion_tokens", 0) or 0
                ),
                "total_tokens": int(getattr(resp.usage, "total_tokens", 0) or 0),
            }
            details = getattr(resp.usage, "prompt_tokens_details", None)
            if details is not None:
                usage["cached_prompt_tokens"] = int(
                    getattr(details, "cached_tokens", 0) or 0
                )
            return _normalize_policy_label(raw), _price_policy_usage(policy["model"], usage)
        responses_error: Exception | None = None
        resp = None
        for attempt in range(1, POLICY_RETRIES + 1):
            try:
                # Parity with the chat branch. Responses spells the output cap
                # `max_output_tokens`, not `max_tokens`; omitting it does not mean
                # "use the chat value", it means uncapped. It also budgets reasoning
                # tokens out of the same allowance, so a small cap can be consumed
                # entirely by a reasoning item, returning status="incomplete" with no
                # message at all (observed against nvidia/nemotron-3.5-lightning at
                # max_output_tokens=16). Reasoning suppression uses the same payload
                # the chat branch sends.
                request_kwargs: dict[str, Any] = {
                    "model": policy["model"],
                    "input": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0,
                    "max_output_tokens": policy["max_tokens"],
                }
                extra_body = _policy_chat_extra_body(policy)
                if extra_body is not None:
                    request_kwargs["extra_body"] = extra_body
                resp = await asyncio.wait_for(
                    client.responses.create(**request_kwargs),
                    timeout=POLICY_TIMEOUT_SECONDS,
                )
                break
            except Exception as error:
                responses_error = error
                if attempt >= POLICY_RETRIES or not _is_policy_timeout(error):
                    break
                await asyncio.sleep(_policy_retry_delay(attempt))
        if resp is not None:
            responses_error = None
            # An exhausted or truncated response is an infra failure, not a wrong
            # prediction. Letting it fall through to _normalize_policy_label would
            # score an empty string as a bad label and quietly depress the reward.
            status = str(getattr(resp, "status", "") or "")
            if status and status != "completed":
                reason = getattr(getattr(resp, "incomplete_details", None), "reason", None)
                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"Policy model {policy['model']!r} returned Responses status "
                        f"{status!r} (reason={reason!r}) with no completed message; "
                        f"max_output_tokens={policy['max_tokens']}."
                    ),
                )
            raw = (resp.output_text or "").strip()
            if not raw:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"Policy model {policy['model']!r} returned an empty Responses "
                        f"output_text; max_output_tokens={policy['max_tokens']}."
                    ),
                )
            usage = {
                "prompt_tokens": int(getattr(resp.usage, "input_tokens", 0) or 0),
                "completion_tokens": int(getattr(resp.usage, "output_tokens", 0) or 0),
                "total_tokens": int(getattr(resp.usage, "total_tokens", 0) or 0),
            }
            details = getattr(resp.usage, "input_tokens_details", None)
            if details is not None:
                usage["cached_prompt_tokens"] = int(
                    getattr(details, "cached_tokens", 0) or 0
                )
            return _normalize_policy_label(raw), _price_policy_usage(policy["model"], usage)
        if responses_error is not None:
            if _is_policy_timeout(responses_error):
                raise HTTPException(
                    status_code=504,
                    detail=(
                        f"Policy model {policy['model']!r} timed out after "
                        f"{POLICY_TIMEOUT_SECONDS:.1f}s through Responses API."
                    ),
                ) from responses_error
            # Fallback to Chat Completions for endpoints that don't support Responses API.
            try:
                extra_body = _policy_chat_extra_body(policy)
                request_kwargs = {
                    "model": policy["model"],
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0,
                    "max_tokens": policy["max_tokens"],
                }
                if extra_body is not None:
                    request_kwargs["extra_body"] = extra_body
                resp = await asyncio.wait_for(
                    client.chat.completions.create(**request_kwargs),
                    timeout=POLICY_TIMEOUT_SECONDS,
                )
            except Exception as chat_error:
                status_code = 504 if _is_policy_timeout(chat_error) else 502
                raise HTTPException(
                    status_code=status_code,
                    detail=(
                        f"Policy model {policy['model']!r} failed through OpenAI-compatible API. "
                        f"Responses error: {responses_error!r}; chat completions error: {chat_error!r}"
                    ),
                ) from chat_error
            raw = (resp.choices[0].message.content or "").strip()
            usage = {
                "prompt_tokens": int(getattr(resp.usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(
                    getattr(resp.usage, "completion_tokens", 0) or 0
                ),
                "total_tokens": int(getattr(resp.usage, "total_tokens", 0) or 0),
            }
    return _normalize_policy_label(raw), _price_policy_usage(policy["model"], usage)


def _normalize_policy_label(raw: str) -> str:
    # Normalize: map common exact/case/punctuation variants back to the canonical
    # dataset label so the scorer honors labels like Refund_not_showing_up.
    candidate = raw.strip().strip("`'\"").splitlines()[0].strip()
    if candidate in LABELS:
        return candidate
    lowered = candidate.lower()
    if lowered in _LABEL_BY_LOWER:
        return _LABEL_BY_LOWER[lowered]
    simplified = "".join(ch for ch in lowered if ch.isalnum() or ch == "_")
    if simplified in _LABEL_BY_SIMPLIFIED:
        return _LABEL_BY_SIMPLIFIED[simplified]
    for label in LABELS:
        if label.lower() in lowered:
            return label
        label_simplified = "".join(
            ch for ch in label.lower() if ch.isalnum() or ch == "_"
        )
        if label_simplified and label_simplified in simplified:
            return label
    # Last-resort: no recognized label in response — return the raw first-line so
    # the scorer marks it incorrect. Optimizer sees a real "wrong" signal.
    return candidate or "<no_label>"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--storage-root", type=Path)
    args = parser.parse_args()
    if args.storage_root is not None:
        args.storage_root.mkdir(parents=True, exist_ok=True)
        os.environ["BANKING77_STREAM_ROOT"] = str(args.storage_root.resolve())
    uvicorn.run(
        app, host=args.host, port=args.port, log_level="warning", access_log=False
    )


if __name__ == "__main__":
    main()
