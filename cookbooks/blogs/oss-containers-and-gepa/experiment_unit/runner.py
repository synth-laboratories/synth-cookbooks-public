"""Parallel, time-boxed executor for blog experiments.

Hard rule: no run exceeds `time_limit` (default 30 min). On timeout we kill the
run's whole process group — which takes its container child with it — and still
extract + heldout whatever candidates were persisted. synth_gepa checkpoints
`candidate_registry.json` + `workspace.sqlite` incrementally, so an aborted run
is still scoreable. (gepa-ai loses in-memory candidates on a kill until an
incremental checkpoint is added to its adapter in scripts/run_stack.py — until
then an aborted gepa-ai arm recovers nothing, which this runner reports honestly.)

Parallelism is safe because every arm — head-to-head AND proposer-sweep — boots
its OWN container on a free port (its own uvicorn loop). The legacy sweep ran
arms sequentially because they shared one container's event loop; here each arm
is isolated. (The OpenRouter rpm cap is still shared across containers, so keep
--max-parallel modest unless the arms route to a direct provider.)

Phases:
  head-to-head:  run_stack.py --stack {synth_gepa,gepa_ai}  ->  extract  ->  heldout+build
  proposer-sweep: boot container -> templated arm config -> `synth-optimizers gepa run` (capped)
"""

from __future__ import annotations

import os
import re
import signal
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable
from urllib.request import urlopen

from blog_paths import CHARTS_DIR, EVALS_DIR, REPO_ROOT

from .model import Experiment, ExperimentKind

DEFAULT_TIME_LIMIT = 1800   # 30 min hard cap per arm
EXTRACT_LIMIT = 600
SCORE_LIMIT = 1800
CONTAINER_HEALTH_TIMEOUT = 120
LOG_DIR = EVALS_DIR / "runs" / "_runner_logs"
GEPA_DIR = EVALS_DIR.parent  # cookbooks/optimizers/gepa
SWEEP_CONFIG_DIR = CHARTS_DIR / "chart-d-proposer-scaling" / "configs" / "proposer_sweep"

# .env files the sweep scripts / eval harness source, loaded before launching so
# subprocesses inherit provider keys even from a bare shell.
ENV_FILES = (
    Path.home() / "Documents" / "GitHub" / "optimizers" / ".env",
    REPO_ROOT.parent / "synth-ai" / ".env",
)

# Provider keys each container needs (the runner fails fast if any are missing).
CONTAINER_ENV_REQUIRED: dict[str, tuple[str, ...]] = {
    "tau2_retail": ("GEMINI_API_KEY", "OPENAI_API_KEY"),   # agent=Gemini, user-sim=OpenAI
    "healthbench": ("OPENROUTER_API_KEY", "OPENAI_API_KEY"),
    "banking77": ("OPENROUTER_API_KEY", "OPENAI_API_KEY"),
    "hotpotqa": ("OPENROUTER_API_KEY", "OPENAI_API_KEY"),
}


def load_env_files() -> None:
    """Source known .env files into os.environ without overriding what's set."""
    for path in ENV_FILES:
        if not path.exists():
            continue
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip().removeprefix("export ").strip()
            if key and key not in os.environ:
                os.environ[key] = val.strip().strip('"').strip("'")


def missing_keys(experiments: list[Experiment]) -> list[str]:
    needed: set[str] = set()
    for exp in experiments:
        needed.update(CONTAINER_ENV_REQUIRED.get(exp.container, ()))
    return sorted(k for k in needed if not os.environ.get(k))


class RunStatus(str, Enum):
    COMPLETED = "completed"
    ABORTED = "aborted"    # hit the hard time limit -> killed
    FAILED = "failed"      # exited non-zero, or could not start, before the limit


@dataclass(frozen=True)
class Job:
    label: str
    argv: list[str]
    time_limit: int


@dataclass
class JobResult:
    label: str
    status: RunStatus
    returncode: int | None
    elapsed: float
    log_path: str


@dataclass(frozen=True)
class ContainerSpec:
    """How to boot a benchmark's container (mirrors the chart-d sweep scripts)."""
    project: Path
    app: Path
    env: dict[str, str]


CONTAINER_SPECS: dict[str, ContainerSpec] = {
    "tau2_retail": ContainerSpec(
        GEPA_DIR / "tau2_retail_container",
        GEPA_DIR / "tau2_retail_container" / "synth_service_app.py",
        # Gemini-direct: LiteLLM native `gemini/` provider routes the agent to
        # Google via GEMINI_API_KEY (no shared OpenRouter rpm cap). We deliberately
        # leave OPENAI_BASE_URL unset so the gpt-4.1-nano user-simulator still
        # routes to OpenAI via OPENAI_API_KEY. Both keys come from os.environ.
        {
            "TAU2_RETAIL_AGENT_MODEL": "gemini/gemini-3.1-flash-lite",
            "TAU2_RETAIL_USER_MODEL": "gpt-4.1-nano",
        },
    ),
    "healthbench": ContainerSpec(
        GEPA_DIR / "healthbench_container",
        GEPA_DIR / "healthbench_container" / "synth_service_app.py",
        {
            "HEALTHBENCH_POLICY_MODEL": "google/gemini-2.5-flash-lite",
            "HEALTHBENCH_JUDGE_MODEL": "google/gemini-2.5-flash-lite",
            "HEALTHBENCH_POLICY_BASE_URL": "https://openrouter.ai/api/v1",
            "HEALTHBENCH_POLICY_API_KEY_ENV": "OPENROUTER_API_KEY",
        },
    ),
}


def _uv(*script_args: str) -> list[str]:
    return ["uv", "run", "--project", str(EVALS_DIR), "python", *script_args]


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGTERM then SIGKILL the process group (container/child included)."""
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            continue


def _wait_health(port: int, timeout: int = CONTAINER_HEALTH_TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=3) as r:  # noqa: S310 (localhost only)
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


# --- plain subprocess job (head-to-head arms, extract, heldout, build) --------
def run_job(job: Job) -> JobResult:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{job.label}.log"
    start = time.monotonic()
    with log_path.open("w") as fh:
        fh.write(f"$ {' '.join(job.argv)}\n\n")
        fh.flush()
        proc = subprocess.Popen(
            job.argv, cwd=str(EVALS_DIR), stdout=fh, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            rc = proc.wait(timeout=job.time_limit)
            status = RunStatus.COMPLETED if rc == 0 else RunStatus.FAILED
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            rc = None
            status = RunStatus.ABORTED
    return JobResult(job.label, status, rc, time.monotonic() - start, str(log_path))


# --- managed proposer-sweep arm (own container + templated config, capped) ----
def _templated_config(src: Path, port: int) -> Path:
    """Copy the arm config beside the original (so its relative output_dir still
    resolves) with the container URL repointed to this arm's private port."""
    text = re.sub(r"http://127\.0\.0\.1:\d+", f"http://127.0.0.1:{port}", src.read_text())
    text = _with_task_pools(text)
    text = _with_fresh_run_tag(text)
    tmp = src.with_name(f"{src.stem}.runner_{port}.toml")
    tmp.write_text(text)
    return tmp


def _with_fresh_run_tag(text: str) -> str:
    tag = os.environ.get("EXPERIMENT_UNIT_FRESH_RUN_TAG", "").strip()
    if not tag:
        return text

    def replace_run_id(match: re.Match[str]) -> str:
        return f'{match.group(1)}{match.group(2)}_{tag}{match.group(3)}'

    def replace_output_dir(match: re.Match[str]) -> str:
        output_dir = match.group(2).rstrip("/")
        base, _, leaf = output_dir.rpartition("/")
        tagged_leaf = f"{leaf}_{tag}"
        return f'{match.group(1)}{base}/{tagged_leaf}{match.group(3)}'

    def replace_cache_path(match: re.Match[str]) -> str:
        path = match.group(2)
        if path.endswith(".sqlite"):
            path = f"{path[:-7]}_{tag}.sqlite"
        return f"{match.group(1)}{path}{match.group(3)}"

    text = re.sub(r'^(run_id\s*=\s*")([^"]+)(")$', replace_run_id, text, flags=re.MULTILINE)
    text = re.sub(r'^(output_dir\s*=\s*")([^"]+)(")$', replace_output_dir, text, flags=re.MULTILINE)
    text = re.sub(r'^(namespace\s*=\s*")([^"]+)(")$', replace_run_id, text, flags=re.MULTILINE)
    return re.sub(r'^(path\s*=\s*")([^"]+\.sqlite)(")$', replace_cache_path, text, flags=re.MULTILINE)


def _with_task_pools(text: str) -> str:
    if "[gepa.task_pools]" in text:
        return text
    train = re.search(r"^train_ids = \[(?P<ids>[^\n]*)\]$", text, re.MULTILINE)
    heldout = re.search(r"^heldout_ids = \[(?P<ids>[^\n]*)\]$", text, re.MULTILINE)
    if not train or not heldout:
        return text
    block = (
        "\n[gepa.task_pools]\n"
        f"pareto = [{train.group('ids')}]\n"
        f"minibatch = [{train.group('ids')}]\n"
        f"reflection = [{train.group('ids')}]\n"
        f"heldout = [{heldout.group('ids')}]\n"
    )
    if "\n[cache]\n" in text:
        return text.replace("\n[cache]\n", f"{block}\n[cache]\n", 1)
    return f"{text.rstrip()}\n{block}"


def run_sweep_arm(exp: Experiment, arm: str, time_limit: int) -> JobResult:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    label = f"{exp.container}.D.{arm}"
    log_path = LOG_DIR / f"{label}.log"
    spec = CONTAINER_SPECS.get(exp.container)
    src_cfg = SWEEP_CONFIG_DIR / f"{exp.container}_{arm}.toml"
    start = time.monotonic()

    def fail(msg: str) -> JobResult:
        log_path.write_text(f"{msg}\n")
        return JobResult(label, RunStatus.FAILED, None, time.monotonic() - start, str(log_path))

    if spec is None:
        return fail(f"no ContainerSpec for {exp.container}")
    if not src_cfg.exists():
        return fail(f"no arm config {src_cfg}")

    port = _free_port()
    log = log_path.open("w")
    log.write(f"# proposer-sweep arm {label} on private container :{port}\n")
    log.flush()
    env = {**os.environ, **spec.env}
    container = subprocess.Popen(
        ["uv", "run", "--project", str(spec.project), "python", str(spec.app),
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    tmp_cfg: Path | None = None
    try:
        if not _wait_health(port):
            _kill_group(container)
            log.close()
            return JobResult(label, RunStatus.FAILED, None, time.monotonic() - start, str(log_path))
        tmp_cfg = _templated_config(src_cfg, port)
        argv = ["uv", "run", "--project", str(EVALS_DIR),
                "synth-optimizers", "gepa", "run", "--config", str(tmp_cfg)]
        log.write(f"\n$ {' '.join(argv)}\n\n")
        log.flush()
        run = subprocess.Popen(argv, cwd=str(EVALS_DIR), stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = run.wait(timeout=time_limit)
            status = RunStatus.COMPLETED if rc == 0 else RunStatus.FAILED
        except subprocess.TimeoutExpired:
            _kill_group(run)
            rc = None
            status = RunStatus.ABORTED
    finally:
        _kill_group(container)
        if tmp_cfg and tmp_cfg.exists():
            tmp_cfg.unlink()
        log.close()
    return JobResult(label, status, rc, time.monotonic() - start, str(log_path))


# --- pool ---------------------------------------------------------------------
def run_parallel(thunks: list[Callable[[], JobResult]], max_parallel: int) -> list[JobResult]:
    results: list[JobResult] = []
    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        for fut in as_completed([ex.submit(t) for t in thunks]):
            r = fut.result()
            results.append(r)
            cap = "  (HIT CAP)" if r.status is RunStatus.ABORTED else ""
            print(f"  [{r.status.value:9}] {r.label:34} {r.elapsed:5.0f}s{cap}", flush=True)
    return results


def latest_run_dir(stack: str, container: str) -> Path | None:
    base = EVALS_DIR / "runs" / stack / container
    if not base.exists():
        return None
    dirs = [d for d in base.iterdir() if d.is_dir()]
    return max(dirs, key=lambda d: d.stat().st_mtime, default=None)


def arm_thunks(experiments: list[Experiment], time_limit: int) -> list[Callable[[], JobResult]]:
    """One capped thunk per arm (the unit that consumes rollouts)."""
    thunks: list[Callable[[], JobResult]] = []
    for exp in experiments:
        if exp.kind is ExperimentKind.HEAD_TO_HEAD:
            for stack in exp.arms:
                job = Job(f"{exp.container}.{stack}",
                          _uv("scripts/run_stack.py", "--benchmark", exp.container, "--stack", stack),
                          time_limit)
                thunks.append(lambda j=job: run_job(j))
        else:  # proposer sweep: own container per arm, capped
            for arm in exp.arms:
                thunks.append(lambda e=exp, a=arm: run_sweep_arm(e, a, time_limit))
    return thunks


def score_jobs(exp: Experiment) -> list[Job]:
    jobs: list[Job] = []
    for stack in exp.arms:
        rd = latest_run_dir(stack, exp.container)
        if rd is None:
            continue
        jobs.append(Job(
            f"extract.{exp.container}.{stack}",
            _uv("scripts/extract_candidates.py", "--benchmark", exp.container,
                "--stack", stack, "--run-dir", str(rd)),
            EXTRACT_LIMIT))
    return jobs


def heldout_jobs(exp: Experiment) -> list[Job]:
    return [
        Job(f"heldout.{exp.container}",
            _uv("scripts/evaluate_heldout.py", "--benchmark", exp.container), SCORE_LIMIT),
        Job(f"build.{exp.container}",
            _uv("scripts/build_evidence.py", "--benchmark", exp.container), EXTRACT_LIMIT),
    ]


def run_experiments(experiments: list[Experiment], time_limit: int, max_parallel: int,
                    dry_run: bool) -> int:
    head = [e for e in experiments if e.kind is ExperimentKind.HEAD_TO_HEAD]
    sweeps = [e for e in experiments if e.kind is ExperimentKind.PROPOSER_SWEEP]
    n_arms = sum(len(e.arms) for e in experiments)
    print(f"plan: {len(experiments)} experiments -> {n_arms} capped arms "
          f"(<= {time_limit}s each), max_parallel={max_parallel}")
    print(f"      head-to-head: {len(head)}  |  proposer-sweep: {len(sweeps)} "
          f"(each arm gets its own container)")

    load_env_files()
    missing = missing_keys(experiments)
    if missing:
        srcs = ", ".join(str(p) for p in ENV_FILES if p.exists()) or "(no .env found)"
        print(f"\nPREFLIGHT FAILED — missing provider keys: {', '.join(missing)}\n"
              f"  loaded from: {srcs}\n"
              f"  export them or add to a sourced .env. tau2 needs GEMINI_API_KEY "
              f"(agent) + OPENAI_API_KEY (user-sim); others need OPENROUTER_API_KEY "
              f"+ OPENAI_API_KEY.", flush=True)
        return 2
    print(f"      preflight ok: provider keys present "
          f"({', '.join(sorted({k for e in experiments for k in CONTAINER_ENV_REQUIRED.get(e.container, ())}))})")

    thunks = arm_thunks(experiments, time_limit)
    if dry_run:
        for exp in head:
            for stack in exp.arms:
                print(f"  would run [{time_limit}s] {exp.container}.{stack}: "
                      f"run_stack.py --benchmark {exp.container} --stack {stack}")
        for exp in sweeps:
            for arm in exp.arms:
                cfg = SWEEP_CONFIG_DIR / f"{exp.container}_{arm}.toml"
                print(f"  would run [{time_limit}s] {exp.container}.D.{arm}: own container + "
                      f"synth-optimizers gepa run --config {cfg.name}"
                      f"{'  [MISSING CONFIG]' if not cfg.exists() else ''}")
        return 0

    print("\n=== phase 1: arms (hard-capped, parallel) ===")
    arm_results = run_parallel(thunks, max_parallel)
    aborted = [r for r in arm_results if r.status is RunStatus.ABORTED]
    if aborted:
        print(f"  {len(aborted)} arm(s) hit the cap; recovering partial candidates "
              "(synth_gepa only; gepa-ai aborts recover nothing until the adapter checkpoints).")

    if head:
        print("\n=== phase 2: extract surviving candidates (head-to-head) ===")
        run_parallel([lambda j=j: run_job(j) for e in head for j in score_jobs(e)], max_parallel)
        print("\n=== phase 3: heldout re-eval + compact evidence (head-to-head) ===")
        for e in head:  # sequential per experiment: build reads what heldout writes
            run_parallel([lambda j=j: run_job(j) for j in heldout_jobs(e)], 1)

    if sweeps:
        print("\npost-launch proposer sweeps done -> rebuild draft Chart D:")
        print("  cd charts/chart-d-proposer-scaling && uv run python build_chart.py")
    print("\nre-check verdicts:  uv run python -m experiment_unit status")
    return 0
