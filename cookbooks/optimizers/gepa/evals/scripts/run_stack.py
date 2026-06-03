"""Run one stack (synth_gepa or gepa_ai) against a benchmark under parity conditions.

Both stacks run against the same Banking77 container, using parameters defined in
configs/<benchmark>.toml. Outputs go to runs/<stack>/<benchmark>/<run_id>/. A
commands.jsonl row is appended to evidence/ after each run.

Usage (from evals/):
    uv run python scripts/run_stack.py --benchmark banking77 --stack synth_gepa
    uv run python scripts/run_stack.py --benchmark banking77 --stack gepa_ai
    uv run python scripts/run_stack.py --benchmark banking77 --stack gepa_ai --dry-run
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import queue
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import tomllib
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = EVALS_DIR.parents[3]  # synth-cookbooks-public/
EVIDENCE_DIR = EVALS_DIR / "evidence"
SYNTH_AI_ENV = REPO_ROOT.parent / "synth-ai" / ".env"


def load_env() -> None:
    if SYNTH_AI_ENV.is_file():
        for line in SYNTH_AI_ENV.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_config(benchmark: str) -> dict:
    cfg_path = EVALS_DIR / "configs" / f"{benchmark}.toml"
    if not cfg_path.exists():
        raise SystemExit(f"config not found: {cfg_path}")
    with open(cfg_path, "rb") as f:
        return tomllib.load(f)


def pick_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def benchmark_container_dir(cfg: dict) -> Path:
    bench = cfg["benchmark"]
    dirname = bench.get("container_dir") or f"{bench['name']}_container"
    return EVALS_DIR.parent / dirname


def benchmark_env_prefix(cfg: dict) -> str:
    bench = cfg["benchmark"]
    return bench.get("env_prefix") or bench["name"].upper().replace("-", "_")


def start_container(port: int, cfg: dict) -> subprocess.Popen:
    bench = cfg["benchmark"]
    parity = cfg["parity_controls"]
    train_seeds = cfg["dataset"]["train_seeds"]
    heldout_seeds = cfg["dataset"]["heldout_seeds"]
    prefix = benchmark_env_prefix(cfg)
    container_dir = benchmark_container_dir(cfg)
    env = {
        **os.environ,
        f"{prefix}_POLICY_CONCURRENCY": str(parity["rollout_concurrency"]),
        f"{prefix}_POLICY_MODEL": parity["policy_model"],
        f"{prefix}_POLICY_API_KEY_ENV": parity["policy_api_key_env"],
        f"{prefix}_POLICY_MAX_TOKENS": str(parity.get("policy_max_tokens", 16)),
        f"{prefix}_POLICY_RETRIES": str(parity.get("policy_retries", 1)),
        f"{prefix}_ROLLOUT_TIMEOUT_SECONDS": str(parity.get("rollout_timeout_seconds", 30)),
        f"{prefix}_POLICY_TIMEOUT_SECONDS": str(parity.get("policy_timeout_seconds", 25)),
    }
    if parity.get("judge_model"):
        env[f"{prefix}_JUDGE_MODEL"] = parity["judge_model"]
    if parity.get("judge_max_tokens"):
        env[f"{prefix}_JUDGE_MAX_TOKENS"] = str(parity["judge_max_tokens"])
    if parity.get("policy_base_url"):
        env[f"{prefix}_POLICY_BASE_URL"] = parity["policy_base_url"]
        env["OPENAI_BASE_URL"] = parity["policy_base_url"]
    if bench["name"] == "tau2_retail":
        env["TAU2_RETAIL_AGENT_MODEL"] = parity["policy_model"]
    if bench.get("synthetic_train_cap"):
        env[f"{prefix}_TRAIN_CAP"] = str(bench["synthetic_train_cap"])
    if bench["name"] == "banking77":
        env.update({
            "BANKING77_TRAIN_SAMPLE": str(max(train_seeds) + 1),
            "BANKING77_TEST_SAMPLE": str(max(heldout_seeds) + 1),
            "BANKING77_TRAIN_SHUFFLE_SEED": str(bench["train_shuffle_seed"]),
            "BANKING77_TEST_SHUFFLE_SEED": str(bench["test_shuffle_seed"]),
            "BANKING77_POLICY_DISABLE_REASONING": "auto",
            "BANKING77_POLICY_API_MODE": "auto",
        })
    cmd = [
        "uv", "run", "--project", str(container_dir),
        "python", str(container_dir / "synth_service_app.py"),
        "--host", "127.0.0.1", "--port", str(port),
    ]
    return subprocess.Popen(
        cmd,
        cwd=str(container_dir.parent),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait_for_health(port: int, timeout: float = 90.0) -> None:
    import httpx
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=3.0)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1.0)
    raise RuntimeError(f"container did not become healthy within {timeout}s")


def append_command(row: dict) -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE_DIR / "commands.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


# ── Synth GEPA ──────────────────────────────────────────────────────────────

def _fail(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    """Raise a loud, clearly-delimited configuration error."""
    banner = "\n" + "=" * 70
    raise SystemExit(f"{banner}\n✗ PROPOSER CONFIG ERROR\n{msg}{banner}")


# Valid proposer backends and the auth modes each supports.
VALID_PROPOSER_BACKENDS = ("codex_app_server", "deepseek_chat")
VALID_CODEX_AUTH_MODES = ("chatgpt", "api_key")


class _CodexAppServerReflectionClient:
    def __init__(self, *, workspace_dir: Path, settings: dict, model: str):
        self.workspace_dir = workspace_dir
        self.settings = settings
        self.model = model
        self.next_id = 1
        self.sent_messages: list[dict] = []
        self.received_messages: list[dict] = []
        self._buffer: list[dict] = []
        self._stderr_tail: list[str] = []
        self._initialized = False

        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        env = self._build_env()
        self.proc = subprocess.Popen(
            ["codex", "app-server"],
            cwd=str(self.workspace_dir),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._stdout_queue: queue.Queue[dict | Exception | None] = queue.Queue()
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def run_turn(self, prompt: str) -> tuple[str, dict]:
        if not self._initialized:
            initialize_id = self._send_request(
                "initialize",
                {
                    "clientInfo": {
                        "name": "gepa-ai-codex-reflection",
                        "title": "gepa-ai Codex Reflection",
                        "version": "1",
                    }
                },
            )
            self._wait_for_response(initialize_id, timeout=60)
            self._send_notification("initialized", None)
            self._initialized = True

        thread_id = self._send_request(
            "thread/start",
            {
                "model": self.model,
                "developerInstructions": "You are the reflection LM for an upstream GEPA optimizer. Return only the requested optimized prompt text.",
                "approvalPolicy": self.settings.get("proposer_approval_policy", "never"),
                "sandbox": self.settings.get("proposer_sandbox_mode", "workspace-write"),
            },
        )
        thread_response = self._wait_for_response(thread_id, timeout=60)
        thread = self._extract_id(thread_response, ("thread", "threadId"))
        if not thread:
            raise RuntimeError(f"codex app-server thread/start response missing thread id: {thread_response}")

        turn_id = self._send_request(
            "turn/start",
            {
                "threadId": thread,
                "model": self.model,
                "input": [{"type": "text", "text": prompt, "text_elements": []}],
                "effort": self.settings.get("proposer_reasoning_effort", "medium"),
                "approvalPolicy": self.settings.get("proposer_approval_policy", "never"),
                "sandboxPolicy": self._sandbox_policy(),
            },
        )
        turn = self._wait_for_turn_started(turn_id, timeout=60)
        final_turn = self._wait_for_turn(turn, timeout=int(self.settings.get("proposer_timeout_seconds", 900)))
        if final_turn.get("method") != "turn/completed":
            raise RuntimeError(f"codex app-server turn did not complete: {final_turn}")
        text = self._extract_assistant_text(final_turn)
        if not text:
            text = self._extract_assistant_text_from_messages()
        if not text:
            text = self._extract_assistant_text_from_session(thread)
        if not text:
            raise RuntimeError(f"codex app-server completed without extractable assistant text: {final_turn}")
        return text.strip(), self._latest_usage()

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

    def _build_env(self) -> dict:
        env = dict(os.environ)
        auth_mode = self.settings.get("proposer_auth_mode")
        if auth_mode != "chatgpt":
            raise SystemExit("gepa_ai reflection via Codex currently requires proposer_auth_mode='chatgpt'")
        source = Path(os.path.expanduser(self.settings.get("proposer_codex_home") or "~/.codex"))
        auth_json = source / "auth.json"
        if not auth_json.is_file():
            raise SystemExit(f"ChatGPT Codex auth missing at {auth_json}; run `codex login` first")
        target = self.workspace_dir / ".codex_home"
        target.mkdir(parents=True, exist_ok=True)
        copied_auth = False
        for name in ("auth.json", "installation_id", "version.json", "models_cache.json"):
            src = source / name
            if src.is_file():
                shutil.copy2(src, target / name)
                copied_auth = copied_auth or name == "auth.json"
        if not copied_auth:
            raise SystemExit(f"ChatGPT Codex auth missing at {auth_json}; run `codex login` first")
        env["CODEX_HOME"] = str(target)
        return env

    def _sandbox_policy(self):
        mode = self.settings.get("proposer_sandbox_mode", "workspace-write")
        if mode == "danger-full-access":
            return {"type": "dangerFullAccess"}
        if mode == "read-only":
            return {"type": "readOnly", "access": {"type": "fullAccess"}, "networkAccess": True}
        if mode == "workspace-write":
            return {"type": "workspaceWrite", "readOnlyAccess": {"type": "fullAccess"}, "networkAccess": True}
        return mode

    def _read_stdout(self) -> None:
        assert self.proc.stdout is not None
        try:
            for line in self.proc.stdout:
                line = line.strip()
                if not line:
                    continue
                self._stdout_queue.put(json.loads(line))
        except Exception as exc:
            self._stdout_queue.put(exc)
        finally:
            self._stdout_queue.put(None)

    def _drain_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            self._stderr_tail.append(line.rstrip())
            self._stderr_tail = self._stderr_tail[-20:]

    def _send_request(self, method: str, params: dict) -> int:
        msg_id = self.next_id
        self.next_id += 1
        self._send({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params})
        return msg_id

    def _send_notification(self, method: str, params) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _send(self, payload: dict) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("codex app-server stdin unavailable")
        self.sent_messages.append(payload)
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()

    def _read_next(self, timeout: int) -> dict:
        if self._buffer:
            return self._buffer.pop(0)
        item = self._stdout_queue.get(timeout=timeout)
        if item is None:
            tail = "\n".join(self._stderr_tail[-5:])
            raise RuntimeError(f"codex app-server stdout closed; stderr_tail={tail}")
        if isinstance(item, Exception):
            raise RuntimeError(f"codex app-server stdout read failed: {item}")
        self.received_messages.append(item)
        return item

    def _wait_for_response(self, msg_id: int, timeout: int) -> dict:
        deferred = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self._read_next(max(1, int(deadline - time.time())))
            if msg.get("id") == msg_id and not msg.get("method"):
                self._buffer = deferred + self._buffer
                if msg.get("error"):
                    raise RuntimeError(f"codex app-server request {msg_id} failed: {msg['error']}")
                return msg
            deferred.append(msg)
        tail = "\n".join(self._stderr_tail[-5:])
        raise TimeoutError(f"timed out waiting for codex app-server response {msg_id}; stderr_tail={tail}")

    def _wait_for_turn_started(self, msg_id: int, timeout: int) -> str:
        deferred = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self._read_next(max(1, int(deadline - time.time())))
            if msg.get("id") == msg_id and not msg.get("method"):
                self._buffer = deferred + self._buffer
                if msg.get("error"):
                    raise RuntimeError(f"codex app-server turn/start failed: {msg['error']}")
                turn = self._extract_id(msg, ("turn", "turnId"))
                if turn:
                    return turn
            if msg.get("method") == "turn/started":
                turn = self._extract_id(msg, ("turn", "turnId"))
                if turn:
                    self._buffer = deferred + self._buffer
                    return turn
            deferred.append(msg)
        tail = "\n".join(self._stderr_tail[-5:])
        raise TimeoutError(f"timed out waiting for codex turn start; stderr_tail={tail}")

    def _wait_for_turn(self, turn_id: str, timeout: int) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self._read_next(max(1, int(deadline - time.time())))
            method = msg.get("method")
            observed_turn = self._extract_id(msg, ("turn", "turnId"))
            if method in ("turn/completed", "turn/failed", "turn/interrupted") and (
                not observed_turn or observed_turn == turn_id
            ):
                return msg
        tail = "\n".join(self._stderr_tail[-5:])
        raise TimeoutError(f"timed out waiting for codex turn {turn_id}; stderr_tail={tail}")

    def _extract_id(self, msg: dict, names: tuple[str, str]) -> str | None:
        object_name, flat_name = names
        for root in ("result", "params"):
            value = msg.get(root) or {}
            nested = value.get(object_name)
            if isinstance(nested, dict) and nested.get("id"):
                return str(nested["id"])
            if value.get(flat_name):
                return str(value[flat_name])
        return None

    def _extract_assistant_text(self, msg: dict) -> str | None:
        candidates = [
            ("params", "turn", "outputText"),
            ("params", "turn", "output_text"),
            ("params", "turn", "lastAssistantMessage"),
            ("result", "turn", "outputText"),
            ("result", "turn", "output_text"),
        ]
        for path in candidates:
            value = msg
            for part in path:
                value = value.get(part) if isinstance(value, dict) else None
            if isinstance(value, str) and value.strip():
                return value
        return None

    def _extract_assistant_text_from_messages(self) -> str | None:
        for msg in reversed(self.received_messages):
            text = self._find_assistant_text(msg)
            if text:
                return text
        return None

    def _extract_assistant_text_from_session(self, thread_id: str) -> str | None:
        db_path = self.workspace_dir / ".codex_home" / "state_5.sqlite"
        if not db_path.is_file():
            return None
        try:
            with sqlite3.connect(db_path) as con:
                row = con.execute("select rollout_path from threads where id = ?", (thread_id,)).fetchone()
        except sqlite3.Error:
            return None
        if not row:
            return None
        path = Path(row[0])
        if not path.is_file():
            return None
        found = None
        try:
            for line in path.read_text().splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = self._find_assistant_text(event)
                if text:
                    found = text
        except OSError:
            return None
        return found

    def _find_assistant_text(self, value) -> str | None:
        if isinstance(value, dict):
            role = value.get("role") or value.get("type")
            for key in ("content", "text", "message", "outputText", "output_text"):
                item = value.get(key)
                if isinstance(item, str) and item.strip() and role in ("assistant", "agent", "message"):
                    return item
                if isinstance(item, list) and role in ("assistant", "agent", "message"):
                    parts = []
                    for part in item:
                        if isinstance(part, dict):
                            text = part.get("text")
                            if isinstance(text, str) and text.strip():
                                parts.append(text)
                    if parts:
                        return "\n".join(parts)
            for item in value.values():
                found = self._find_assistant_text(item)
                if found:
                    return found
        elif isinstance(value, list):
            for item in reversed(value):
                found = self._find_assistant_text(item)
                if found:
                    return found
        return None

    def _latest_usage(self) -> dict:
        latest = {}
        for msg in self.received_messages:
            if msg.get("method") != "thread/tokenUsage/updated":
                continue
            usage = ((msg.get("params") or {}).get("tokenUsage") or {}).get("total") or {}
            if usage:
                latest = usage
        return latest


def _build_proposer_section(sg: dict, pc: dict) -> str:
    """Construct the [proposer] TOML block with explicit per-backend/auth handling.

    Validates the configuration up front and fails LOUDLY with an actionable
    message, so a misconfig surfaces here in Python rather than as a cryptic
    Rust error after the container has already started.
    """
    backend = sg.get("proposer_backend")
    model = pc["proposer_model"]
    timeout = sg["proposer_timeout_seconds"]

    if backend not in VALID_PROPOSER_BACKENDS:
        _fail(f"proposer_backend={backend!r} is not supported.\n"
              f"Valid backends: {', '.join(VALID_PROPOSER_BACKENDS)}.")

    # ── codex_app_server (agentic codex proposer) ────────────────────────────
    if backend == "codex_app_server":
        auth_mode = sg.get("proposer_auth_mode")
        if auth_mode not in VALID_CODEX_AUTH_MODES:
            _fail(f"proposer_backend='codex_app_server' has proposer_auth_mode={auth_mode!r}.\n"
                  f"Valid auth modes: {', '.join(VALID_CODEX_AUTH_MODES)}.")

        if auth_mode == "chatgpt":
            # ChatGPT-subscription auth: copy the host codex login (codex_home);
            # api_key_env is forbidden by the optimizer in this mode.
            codex_home = os.path.expanduser(sg.get("proposer_codex_home") or "~/.codex")
            auth_json = os.path.join(codex_home, "auth.json")
            if not os.path.isdir(codex_home):
                _fail(f"proposer_auth_mode='chatgpt' needs a codex login dir, but\n"
                      f"  proposer_codex_home -> {codex_home}\ndoes not exist. "
                      f"Run `codex login` (ChatGPT), or set proposer_codex_home.")
            if not os.path.isfile(auth_json):
                _fail(f"proposer_auth_mode='chatgpt' requires a ChatGPT login at\n"
                      f"  {auth_json}\nbut it is missing. Run `codex login` with a ChatGPT account.")
            if sg.get("proposer_api_key_env"):
                _fail("proposer_auth_mode='chatgpt' cannot be combined with "
                      "proposer_api_key_env (the optimizer rejects this). "
                      "Remove proposer_api_key_env for ChatGPT auth.")
            auth_lines = (
                f"copy_host_auth = {str(sg.get('proposer_copy_host_auth', True)).lower()}\n"
                f'codex_home = "{codex_home}"'
            )
        else:  # api_key
            key_env = sg.get("proposer_api_key_env")
            if not key_env:
                _fail("proposer_auth_mode='api_key' requires proposer_api_key_env.")
            if not os.environ.get(key_env):
                _fail(f"proposer_auth_mode='api_key' needs env var {key_env}, which is NOT set.")
            auth_lines = f'api_key_env = "{key_env}"'

        return (
            "[proposer]\n"
            f'backend = "codex_app_server"\n'
            "execution_mode = \"local_process\"\n"
            f"timeout_seconds = {timeout}\n"
            f'model = "{model}"\n'
            f'reasoning_effort = "{sg["proposer_reasoning_effort"]}"\n'
            f'auth_mode = "{auth_mode}"\n'
            f"{auth_lines}\n"
            f'sandbox_mode = "{sg["proposer_sandbox_mode"]}"\n'
            f'approval_policy = "{sg["proposer_approval_policy"]}"\n'
        )

    # ── deepseek_chat (direct chat-completions proposer) ─────────────────────
    key_env = sg.get("proposer_api_key_env")
    if not key_env:
        _fail("proposer_backend='deepseek_chat' requires proposer_api_key_env (e.g. DEEPSEEK_API_KEY).")
    if not os.environ.get(key_env):
        _fail(f"proposer_backend='deepseek_chat' needs env var {key_env}, which is NOT set.")
    base_url = sg.get("proposer_base_url", "https://api.deepseek.com")
    return (
        "[proposer]\n"
        'backend = "deepseek_chat"\n'
        'provider = "deepseek"\n'
        'runtime_substrate = "local"\n'
        'execution_mode = "local_process"\n'
        f"timeout_seconds = {timeout}\n"
        f'model = "{model}"\n'
        'auth_mode = "api_key"\n'
        f'api_key_env = "{key_env}"\n'
        f'base_url = "{base_url}"\n'
    )


def _build_synth_gepa_toml(cfg: dict, port: int, run_id: str, out_dir: Path) -> str:
    ds = cfg["dataset"]
    sc = cfg["seed_candidate"]
    pc = cfg["parity_controls"]
    sg = cfg["synth_gepa"]
    bench = cfg["benchmark"]
    budget = int(cfg["limits"]["search_rollout_budget"])

    train_seeds_str = ", ".join(str(s) for s in ds["train_seeds"])
    heldout_seeds_str = ", ".join(str(s) for s in ds["heldout_seeds"])
    # v2 taskset ids are "<split>:<seed>" strings; train/heldout must be disjoint.
    train_ids_str = ", ".join(f'"{ds["train_split"]}:{s}"' for s in ds["train_seeds"])
    heldout_ids_str = ", ".join(f'"{ds["heldout_split"]}:{s}"' for s in ds["heldout_seeds"])

    proposer_section = _build_proposer_section(sg, pc)

    return f"""\
[run]
run_id = "{run_id}"
output_dir = "{out_dir}"
seed = {pc["random_seed"]}

[container]
url = "http://127.0.0.1:{port}"
startup_timeout_seconds = 5

[taskset]
train_split = "{ds['train_split']}"
heldout_split = "{ds['heldout_split']}"
train_ids = [{train_ids_str}]
heldout_ids = [{heldout_ids_str}]

[candidate]
target_modules = ["{bench['mutable_field']}"]

[seed_candidate]
{bench['mutable_field']} = {json.dumps(sc[bench['mutable_field']])}

[policy]
provider = "openrouter"
model = "{pc['policy_model']}"
base_url = "{pc.get('policy_base_url', 'https://openrouter.ai/api/v1')}"
api_key_env = "{pc['policy_api_key_env']}"

{proposer_section}
[gepa]
max_generations = {sg['max_generations']}
proposals_per_generation = {sg['proposals_per_generation']}
minibatch_size = {sg['minibatch_size']}
rollout_submission_mode = "{sg.get('rollout_submission_mode', 'sync')}"
rollout_chunk_size = {sg.get('rollout_chunk_size', 100)}
# Search-only rollout budget (apples-to-apples with gepa-ai max_metric_calls).
# Search is capped by max_train_rollouts = budget. Internal heldout is pinned to
# the minimum (1) — it can't be 0 — and its result is ignored; heldout scoring is
# done uniformly afterwards by evaluate_heldout.py. max_total_rollouts carries
# headroom so the train cap is the binding constraint on search.
max_train_rollouts = {budget}
max_heldout_rollouts = 1
max_total_rollouts = {budget + 200}
max_cost_usd = 0.0

[gepa.pipeline]
mode = "{sg.get('pipeline_mode', 'sync_serial')}"

[cache]
mode = "{pc['cache_mode']}"
path = ""
namespace = ""
"""


def run_synth_gepa(cfg: dict, dry_run: bool) -> Path:
    try:
        import synth_optimizers  # noqa: F401
    except ImportError:
        raise SystemExit(
            "synth_optimizers not installed. "
            "Run: uv add 'synth-optimizers @ git+https://github.com/synth-laboratories/optimizers.git'"
        )
    from synth_optimizers import GepaRun

    bench_name = cfg["benchmark"]["name"]
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_id = f"synth_gepa_{bench_name}_{ts}"
    # GepaRun creates output_dir/run_id/ internally, so pass the parent.
    runs_base = EVALS_DIR / "runs" / "synth_gepa" / bench_name
    runs_base.mkdir(parents=True, exist_ok=True)
    out_dir = runs_base / run_id  # the actual run dir GepaRun will create
    out_dir.mkdir(parents=True, exist_ok=True)

    port = pick_free_port()
    proc = start_container(port, cfg)
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        wait_for_health(port)
        print(f"[synth_gepa] container healthy on port {port}", flush=True)

        toml_text = _build_synth_gepa_toml(cfg, port, run_id, runs_base)
        toml_path = out_dir / "gepa_run.toml"
        toml_path.write_text(toml_text)
        print(f"[synth_gepa] config: {toml_path}", flush=True)

        if dry_run:
            print("[synth_gepa] --dry-run: skipping GepaRun.execute()", flush=True)
            return out_dir

        try:
            result = GepaRun.from_toml(str(toml_path)).execute()
        except Exception as exc:
            bar = "=" * 70
            print(f"\n{bar}\n✗ SYNTH_GEPA RUN FAILED\n  {type(exc).__name__}: {exc}\n"
                  f"  config: {toml_path}\n{bar}", flush=True)
            raise
        print(f"[synth_gepa] manifest: {result.manifest_path}", flush=True)
    finally:
        finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    append_command({
        "command_id": str(uuid.uuid4()),
        "phase": "run",
        "stack": "synth_gepa",
        "benchmark": bench_name,
        "cwd": str(REPO_ROOT),
        "argv": [
            "uv", "run", "--project", str(EVALS_DIR),
            "python", "scripts/run_stack.py",
            "--benchmark", bench_name, "--stack", "synth_gepa",
        ],
        "env_keys": ["OPENAI_API_KEY"],
        "inputs": [str(toml_path)],
        "outputs": [str(out_dir)],
        "started_at": started_at,
        "finished_at": finished_at,
        "exit_code": 0,
    })

    return out_dir


# ── gepa-ai ─────────────────────────────────────────────────────────────────

class _InstrumentedAdapter:
    """Container adapter for gepa-ai that records per-eval timing."""

    propose_new_texts = None  # gepa Protocol attribute — must be present

    def __init__(self, port: int, concurrency: int = 40):
        self.port = port
        self.concurrency = concurrency
        self.eval_calls: list[dict] = []
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.rollout_calls = 0

    def _post_rollout(self, seed: int, split: str, candidate: dict) -> dict:
        import httpx
        body = {
            "seed": seed,
            "split": split,
            "candidate": candidate,
            "submission_mode": "sync",
            "rollout_id": f"gepaai_{split}_{seed}_{int(time.time()*1000)%1_000_000}",
        }
        r = httpx.post(
            f"http://127.0.0.1:{self.port}/rollout",
            json=body,
            timeout=60,
        )
        r.raise_for_status()
        return r.json()

    def _fetch_rows(self, split: str, seeds: list[int]) -> list[dict]:
        import httpx
        r = httpx.post(
            f"http://127.0.0.1:{self.port}/dataset/rows",
            json={"split": split, "seeds": seeds},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["rows"]

    def fetch_train_rows(self, seeds: list[int]) -> list[dict]:
        return self._fetch_rows("train", seeds)

    def fetch_heldout_rows(self, seeds: list[int]) -> list[dict]:
        return self._fetch_rows("test", seeds)

    def evaluate(self, batch: list[dict], candidate: dict, capture_traces: bool = False):
        from gepa.core.adapter import EvaluationBatch

        started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        scores: list[float] = [0.0] * len(batch)
        outputs: list[dict] = [{}] * len(batch)
        trajectories: list[dict] | None = ([{}] * len(batch)) if capture_traces else None
        call_prompt_tokens = 0
        call_completion_tokens = 0

        def _one(i: int, data: dict):
            try:
                result = self._post_rollout(
                    seed=int(data["seed"]),
                    split=str(data["split"]),
                    candidate=candidate,
                )
                reward = float((result.get("reward_info") or {}).get("outcome_reward", 0.0))
                reward_info = result.get("reward_info") or {}
                details = reward_info.get("details") or {}
                summary = result.get("summary") or {}
                metadata = result.get("metadata") or {}
                pred = (
                    details.get("prediction")
                    or summary.get("prediction")
                    or metadata.get("response")
                    or ""
                )
                expected = (
                    details.get("expected")
                    or summary.get("expected")
                    or f"rubric_score={reward:.4f}"
                )
                usage = result.get("usage") or {}
                pin = int(usage.get("prompt_tokens", 0))
                pout = int(usage.get("completion_tokens", 0))
                return i, reward, pred, expected, pin, pout, details, summary, metadata
            except Exception as exc:
                return i, 0.0, "", f"<error: {exc!r}>", 0, 0, {}, {}, {}

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = [pool.submit(_one, i, d) for i, d in enumerate(batch)]
            for fut in as_completed(futures):
                i, reward, pred, expected, pin, pout, details, summary, metadata = fut.result()
                scores[i] = reward
                outputs[i] = {
                    "prediction": pred,
                    "expected": expected,
                    "summary": summary,
                    "details": details,
                }
                self.prompt_tokens += pin
                self.completion_tokens += pout
                call_prompt_tokens += pin
                call_completion_tokens += pout
                self.rollout_calls += 1
                if trajectories is not None:
                    criteria_count = int(details.get("criteria_count", 0) or summary.get("criteria_count", 0) or 0)
                    if criteria_count:
                        criteria_met = int(details.get("criteria_met", 0) or summary.get("criteria_met", 0) or 0)
                        missed = [
                            str(c.get("criterion") or "")
                            for c in (metadata.get("per_criterion") or [])
                            if not c.get("met")
                        ][:3]
                        fb = (
                            f"Rubric reward {reward:.4f}; met {criteria_met}/{criteria_count} criteria. "
                            f"Missed criteria examples: {json.dumps(missed)}"
                        )
                    else:
                        fb = (
                            f"Correct. Predicted '{pred}'."
                            if reward >= 1.0
                            else f"Wrong. Predicted '{pred}', expected '{expected}'."
                        )
                    trajectories[i] = {
                        "data": batch[i],
                        "full_assistant_response": pred,
                        "feedback": fb,
                    }

        finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        mean_score = sum(scores) / len(scores) if scores else 0.0
        self.eval_calls.append({
            "batch_size": len(batch),
            "candidate_payload": candidate,
            "started_at": started_at,
            "finished_at": finished_at,
            "mean_score": mean_score,
            "prompt_tokens": call_prompt_tokens,
            "completion_tokens": call_completion_tokens,
            "rollout_count": len(batch),
        })

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        comp = components_to_update[0]
        items = []
        trajs = eval_batch.trajectories or []
        for traj in trajs:
            data = traj.get("data", {})
            items.append({
                "Inputs": _reflection_input_text(data),
                "Generated Outputs": traj.get("full_assistant_response", ""),
                "Feedback": traj.get("feedback", ""),
            })
        if not items:
            raise Exception("No valid predictions found for any module.")
        return {comp: items}


def _reflection_input_text(row: dict) -> str:
    if row.get("text"):
        return str(row["text"])
    if row.get("question"):
        parts = [f"Question:\n{row.get('question')}"]
        if row.get("context"):
            parts.append(f"Passages:\n{row.get('context')}")
        return "\n\n".join(parts)
    hidden = {"answer", "label"}
    visible = {k: v for k, v in row.items() if k not in hidden}
    return json.dumps(visible, sort_keys=True)


def _normalize_train_row(row: dict) -> dict:
    out = dict(row)
    out.setdefault("text", _reflection_input_text(row))
    if "label" not in out and row.get("answer") is not None:
        out["label"] = row.get("answer")
    return out


def run_gepa_ai(cfg: dict, dry_run: bool) -> Path:
    try:
        import gepa
        from gepa import optimize
    except ImportError:
        raise SystemExit("gepa not installed. Run: uv add gepa")

    bench_name = cfg["benchmark"]["name"]
    mutable = cfg["benchmark"]["mutable_field"]
    ds = cfg["dataset"]
    sc = cfg["seed_candidate"]
    pc = cfg["parity_controls"]
    ga = cfg["gepa_ai"]

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_id = f"gepa_ai_{bench_name}_{ts}"
    out_dir = EVALS_DIR / "runs" / "gepa_ai" / bench_name / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    port = pick_free_port()
    proc = start_container(port, cfg)
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        wait_for_health(port)
        print(f"[gepa_ai] container healthy on port {port}", flush=True)

        adapter = _InstrumentedAdapter(port=port, concurrency=pc["rollout_concurrency"])

        train_rows = adapter.fetch_train_rows(ds["train_seeds"])
        heldout_rows = adapter.fetch_heldout_rows(ds["heldout_seeds"])
        print(f"[gepa_ai] fetched train={len(train_rows)} heldout={len(heldout_rows)}", flush=True)

        trainset = [_normalize_train_row(r) for r in train_rows]
        valset = trainset[:max(1, len(trainset) // 2)]

        seed_candidate = {mutable: sc[mutable]}

        class _ReflectionLM:
            def __init__(self):
                self.prompt_tokens = 0
                self.completion_tokens = 0
                self.cached_prompt_tokens = 0
                self.calls = 0
                # Per-call records so proposer cost can be attributed over time.
                self.call_log: list[dict] = []
                self.client = _CodexAppServerReflectionClient(
                    workspace_dir=out_dir / "reflection_workspace",
                    settings=cfg["synth_gepa"],
                    model=pc["reflection_model"],
                )

            def close(self) -> None:
                self.client.close()

            def __call__(self, prompt) -> str:
                if isinstance(prompt, str):
                    prompt_text = prompt
                else:
                    prompt_text = json.dumps(list(prompt), indent=2)
                started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
                text, usage = self.client.run_turn(prompt_text)
                finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
                pin = int(usage.get("inputTokens", 0) or 0)
                pout = int(usage.get("outputTokens", 0) or 0)
                cached = int(usage.get("cachedInputTokens", 0) or 0)
                self.prompt_tokens += pin
                self.completion_tokens += pout
                self.cached_prompt_tokens += cached
                self.calls += 1
                self.call_log.append({
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "prompt_tokens": pin,
                    "cached_prompt_tokens": cached,
                    "completion_tokens": pout,
                    "total_tokens": int(usage.get("totalTokens", 0) or 0),
                    "backend": "codex_app_server",
                    "auth_mode": "chatgpt",
                })
                return text

        reflection_lm = _ReflectionLM()
        run_dir = str(out_dir / "gepa_run")

        if dry_run:
            print("[gepa_ai] --dry-run: skipping optimize()", flush=True)
            (out_dir / "gepa_ai_run.json").write_text(json.dumps({"dry_run": True}))
            reflection_lm.close()
            return out_dir

        budget = int(cfg["limits"]["search_rollout_budget"])
        t0 = time.time()
        try:
            result = optimize(
                seed_candidate=seed_candidate,
                trainset=trainset,
                valset=valset,
                adapter=adapter,
                reflection_lm=reflection_lm,
                candidate_selection_strategy=ga["candidate_selection_strategy"],
                reflection_minibatch_size=ga["reflection_minibatch_size"],
                max_metric_calls=budget,
                display_progress_bar=False,
                seed=pc["random_seed"],
                run_dir=run_dir,
            )
        except Exception as exc:
            bar = "=" * 70
            print(f"\n{bar}\n✗ GEPA_AI RUN FAILED\n  {type(exc).__name__}: {exc}\n{bar}", flush=True)
            raise
        finally:
            reflection_lm.close()
        elapsed = time.time() - t0
        print(f"[gepa_ai] optimize done in {elapsed:.1f}s", flush=True)

        # Extract per-candidate data from result.
        candidates_out = []
        for i, cand in enumerate(result.candidates or []):
            val_score = None
            try:
                val_score = result.val_aggregate_scores[i]
            except (IndexError, TypeError, AttributeError):
                pass
            parent_idxs = None
            try:
                parent_idxs = list(result.parents[i]) if result.parents else None
            except (IndexError, TypeError, AttributeError):
                pass
            candidates_out.append({
                "idx": i,
                "candidate_payload": cand,
                "val_aggregate_score": val_score,
                "parent_idxs": parent_idxs,
            })

        run_data = {
            "run_id": run_id,
            "stack": "gepa_ai",
            "benchmark": bench_name,
            "started_at": started_at,
            "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "search_rollout_budget": budget,
            "max_metric_calls": budget,
            "total_metric_calls": getattr(result, "total_metric_calls", None),
            "num_candidates": getattr(result, "num_candidates", len(candidates_out)),
            "best_idx": getattr(result, "best_idx", None),
            "best_candidate": getattr(result, "best_candidate", None),
            "seed_candidate": seed_candidate,
            "train_seeds": ds["train_seeds"],
            "heldout_seeds": ds["heldout_seeds"],
            "reflection_model": pc["reflection_model"],
            "policy_model": pc["policy_model"],
            "rollout_prompt_tokens": adapter.prompt_tokens,
            "rollout_completion_tokens": adapter.completion_tokens,
            "rollout_calls": adapter.rollout_calls,
            "reflection_prompt_tokens": reflection_lm.prompt_tokens,
            "reflection_cached_prompt_tokens": reflection_lm.cached_prompt_tokens,
            "reflection_completion_tokens": reflection_lm.completion_tokens,
            "reflection_calls": reflection_lm.calls,
            "reflection_call_log": reflection_lm.call_log,
            "candidates": candidates_out,
            "eval_calls": adapter.eval_calls,
        }
        (out_dir / "gepa_ai_run.json").write_text(json.dumps(run_data, indent=2))
        print(f"[gepa_ai] run data: {out_dir / 'gepa_ai_run.json'}", flush=True)

    finally:
        finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    append_command({
        "command_id": str(uuid.uuid4()),
        "phase": "run",
        "stack": "gepa_ai",
        "benchmark": bench_name,
        "cwd": str(EVALS_DIR),
        "argv": [
            "uv", "run", "--project", ".",
            "python", "scripts/run_stack.py",
            "--benchmark", bench_name, "--stack", "gepa_ai",
        ],
        "env_keys": ["OPENAI_API_KEY"],
        "inputs": [str(EVALS_DIR / "configs" / f"{bench_name}.toml")],
        "outputs": [str(out_dir)],
        "started_at": started_at,
        "finished_at": finished_at,
        "exit_code": 0,
    })

    return out_dir


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one stack against a benchmark.")
    p.add_argument("--benchmark", default="banking77")
    p.add_argument("--stack", required=True, choices=["synth_gepa", "gepa_ai"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Slash budgets to minimum for a fast pipeline smoke test.",
    )
    return p.parse_args()


def _apply_smoke(cfg: dict) -> dict:
    """Return a copy of cfg with minimal budgets for smoke testing.

    Keeps the SAME search-rollout budget for both stacks so the smoke run still
    exercises the apples-to-apples path, just smaller.
    """
    import copy
    cfg = copy.deepcopy(cfg)
    cfg["limits"]["search_rollout_budget"] = 240
    cfg["gepa_ai"]["reflection_minibatch_size"] = 4
    cfg["synth_gepa"]["max_generations"] = 1
    cfg["synth_gepa"]["proposals_per_generation"] = 1
    cfg["synth_gepa"]["minibatch_size"] = 8
    return cfg


def main() -> int:
    load_env()
    args = parse_args()
    cfg = load_config(args.benchmark)
    if args.smoke:
        cfg = _apply_smoke(cfg)
        print("[smoke] Budget cut to minimum for pipeline validation.", flush=True)

    if args.stack == "synth_gepa":
        run_dir = run_synth_gepa(cfg, args.dry_run)
    else:
        run_dir = run_gepa_ai(cfg, args.dry_run)

    print(f"Run dir: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
