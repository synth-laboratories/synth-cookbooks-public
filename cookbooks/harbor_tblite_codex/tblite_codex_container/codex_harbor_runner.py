"""Generic Codex rollout runner for Harbor task images."""

from __future__ import annotations

import argparse
import base64
import json
import os
import mimetypes
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

DEFAULT_MODEL_NAME = "openai/gpt-5.4-mini"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_CODEX_TIMEOUT_SECONDS = 180
_CODEX_MODEL_FALLBACKS: dict[str, str] = {
    "gpt-5.4-nano": "gpt-5.4-mini",
}


class CodexRunResult:
    def __init__(
        self,
        *,
        returncode: int,
        stdout: str,
        stderr: str,
        stdout_log_path: str | None,
        stderr_log_path: str | None,
        status_path: str | None,
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.stdout_log_path = stdout_log_path
        self.stderr_log_path = stderr_log_path
        self.status_path = status_path


def _candidate_paths(*paths: Path) -> list[Path]:
    return [path for path in paths if path is not None]


def load_rollout(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_instruction(task_root: Path, rollout: dict[str, Any]) -> str:
    instruction = (task_root / "instruction.md").read_text(encoding="utf-8").strip()
    policy = rollout.get("policy") if isinstance(rollout.get("policy"), dict) else {}
    policy_config = policy.get("config") if isinstance(policy.get("config"), dict) else {}
    system_prompt = str(policy_config.get("system_prompt") or "").strip()
    system_prompt_suffix = str(policy_config.get("system_prompt_suffix") or "").strip()
    messages = rollout.get("messages") if isinstance(rollout.get("messages"), list) else []
    user_parts = [
        str(message.get("content") or "").strip()
        for message in messages
        if isinstance(message, dict) and str(message.get("role") or "").strip() == "user"
    ]
    sections: list[str] = []
    if system_prompt:
        sections.append(f"System guidance:\n{system_prompt}")
    sections.append(instruction)
    if system_prompt_suffix:
        sections.append(f"Additional policy guidance:\n{system_prompt_suffix}")
    if user_parts:
        sections.append("Additional rollout request:\n" + "\n\n".join(user_parts))
    return "\n\n".join(section for section in sections if section.strip())


def load_harbor_agent(rollout: dict[str, Any]) -> dict[str, Any]:
    raw = rollout.get("harbor_agent")
    if not isinstance(raw, dict):
        raise RuntimeError("rollout payload is missing harbor_agent")
    name = str(raw.get("name") or "").strip().lower()
    if name != "codex":
        raise RuntimeError(f"unsupported harbor_agent.name={name!r}")
    kwargs = dict(raw.get("kwargs") or {})
    kwargs.setdefault("reasoning_effort", DEFAULT_REASONING_EFFORT)
    return {
        "name": name,
        "model_name": str(raw.get("model_name") or DEFAULT_MODEL_NAME).strip(),
        "kwargs": kwargs,
        "env": {str(key): str(value) for key, value in dict(raw.get("env") or {}).items()},
    }


def resolve_workspace_dir(task_root: Path, rollout: dict[str, Any]) -> Path:
    task_metadata = rollout.get("task_metadata")
    if isinstance(task_metadata, dict):
        candidate = str(task_metadata.get("workspace_dir") or "").strip()
        if candidate:
            workspace_dir = Path(candidate)
            if workspace_dir.is_absolute():
                return workspace_dir
            return (task_root / workspace_dir).resolve()
    return task_root


def rollout_env_vars(rollout: dict[str, Any]) -> dict[str, str]:
    raw = rollout.get("env")
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items()}


def _rollout_direct_or_metadata(rollout: dict[str, Any], field: str) -> Any:
    """Resolve a rollout field from the top level or from ``metadata`` (pool rollouts)."""

    if field in rollout:
        return rollout.get(field)
    meta = rollout.get("metadata")
    if isinstance(meta, dict):
        return meta.get(field)
    return None


def _toml_escape_double_quoted(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def build_managed_research_mcp_toml(
    *,
    server_table_name: str,
    synth_api_key: str,
    synth_backend_url: str | None,
) -> str:
    """Stdio MCP block for ``uv run --with synth-managed-research`` (Codex config.toml)."""
    name = str(server_table_name or "synth_managed_research").strip() or "synth_managed_research"
    key = str(synth_api_key or "").strip()
    if not key:
        return ""
    lines = [
        f"[mcp_servers.{name}]",
        "enabled = true",
        'command = "uv"',
        'args = ["run", "--with", "synth-managed-research", "python", "-m", "managed_research.mcp"]',
        f"[mcp_servers.{name}.env]",
        f'SYNTH_API_KEY = "{_toml_escape_double_quoted(key)}"',
    ]
    backend = str(synth_backend_url or "").strip()
    if backend:
        lines.append(f'SYNTH_BACKEND_URL = "{_toml_escape_double_quoted(backend)}"')
    return "\n".join(lines) + "\n"


def _append_managed_research_mcp_from_rollout(rollout: dict[str, Any]) -> str:
    raw = _rollout_direct_or_metadata(rollout, "managed_research_mcp")
    if isinstance(raw, dict):
        merged: dict[str, Any] = dict(raw)
    elif _rollout_direct_or_metadata(rollout, "managed_research_mcp_enabled"):
        merged = {}
    else:
        return ""
    if merged.get("enabled") is False:
        return ""
    key = str(
        merged.get("synth_api_key")
        or (_rollout_direct_or_metadata(rollout, "synth_api_key") or "")
        or os.environ.get("SYNTH_API_KEY")
        or ""
    ).strip()
    if not key:
        return ""
    backend = str(
        merged.get("synth_backend_url")
        or merged.get("backend_base")
        or (_rollout_direct_or_metadata(rollout, "synth_backend_url") or "")
        or os.environ.get("SYNTH_BACKEND_URL")
        or ""
    ).strip() or None
    return build_managed_research_mcp_toml(
        server_table_name=str(merged.get("server_table_name") or "synth_managed_research"),
        synth_api_key=key,
        synth_backend_url=backend,
    )


def materialize_codex_bundled_skills(
    task_root: Path,
    codex_home: Path,
    rollout: dict[str, Any],
) -> None:
    """Copy ``task_root/bundled_skills`` into ``codex_home/skills`` when enabled."""
    if not _rollout_direct_or_metadata(rollout, "codex_enable_bundled_skills"):
        return
    src = task_root / "bundled_skills"
    if not src.is_dir():
        return
    dest = codex_home / "skills"
    shutil.copytree(src, dest, dirs_exist_ok=True)


def _runtime_overrides(rollout: dict[str, Any]) -> dict[str, Any]:
    task_payload = rollout.get("task_payload")
    if not isinstance(task_payload, dict):
        return {}
    overrides = task_payload.get("runtime_overrides")
    return dict(overrides) if isinstance(overrides, dict) else {}


def _safe_runtime_path(task_root: Path, relative_path: str) -> Path:
    candidate = Path(str(relative_path or "").strip())
    if not str(candidate):
        raise RuntimeError("runtime override file path is required")
    if candidate.is_absolute():
        raise RuntimeError("runtime override file path must be relative to task_root")
    resolved = (task_root / candidate).resolve()
    try:
        resolved.relative_to(task_root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"runtime override path escapes task_root: {relative_path}") from exc
    return resolved


def _derived_skill_name(source_path: Path) -> str:
    if source_path.name.lower() == "skill.md" and source_path.parent.name:
        raw = source_path.parent.name
    else:
        raw = source_path.stem or source_path.name
    cleaned = "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in raw).strip("-_.")
    return cleaned or "skill"


def _clear_goex_managed_skills(skills_root: Path) -> None:
    if not skills_root.exists():
        return
    for child in skills_root.iterdir():
        marker = child / ".goex_managed"
        if marker.exists() and child.is_dir():
            shutil.rmtree(child)


def materialize_codex_runtime_overrides(
    task_root: Path,
    codex_home: Path,
    rollout: dict[str, Any],
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    overrides = _runtime_overrides(rollout)
    files = overrides.get("files")
    if isinstance(files, list):
        for item in files:
            if not isinstance(item, dict):
                raise RuntimeError("runtime override files entries must be objects")
            relative_path = str(item.get("path") or "").strip()
            mode = str(item.get("mode") or "append").strip().lower() or "append"
            content = str(item.get("content") or "")
            if mode not in {"append", "replace"}:
                raise RuntimeError(f"unsupported runtime override file mode: {mode}")
            target_path = _safe_runtime_path(task_root, relative_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if mode == "replace":
                target_path.write_text(content, encoding="utf-8")
            else:
                existing = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
                if content.strip() and content.strip() not in existing:
                    if existing.strip():
                        merged = existing.rstrip() + "\n\n" + content.strip() + "\n"
                    else:
                        merged = content.strip() + "\n"
                    target_path.write_text(merged, encoding="utf-8")
                elif not target_path.exists():
                    target_path.write_text(existing, encoding="utf-8")
            refs.append({"kind": "runtime_override_file", "path": str(target_path), "mode": mode})
    skills = overrides.get("skills")
    if isinstance(skills, list):
        skills_root = codex_home / "skills"
        skills_root.mkdir(parents=True, exist_ok=True)
        _clear_goex_managed_skills(skills_root)
        for item in skills:
            if not isinstance(item, dict):
                raise RuntimeError("runtime override skills entries must be objects")
            source_text = str(item.get("source_path") or "").strip()
            if not source_text:
                raise RuntimeError("runtime override skill source_path is required")
            source_path = Path(source_text).expanduser().resolve()
            if not source_path.is_absolute() or not source_path.is_file():
                raise RuntimeError(f"runtime override skill path must reference an existing absolute file: {source_text}")
            skill_name = str(item.get("skill_name") or "").strip() or _derived_skill_name(source_path)
            dest_dir = skills_root / skill_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / "SKILL.md"
            shutil.copy2(source_path, dest_path)
            (dest_dir / ".goex_managed").write_text("1\n", encoding="utf-8")
            refs.append(
                {
                    "kind": "runtime_override_skill",
                    "path": str(dest_path),
                    "skill_name": skill_name,
                    "source_path": str(source_path),
                }
            )
    return refs


def materialize_codex_auth(task_root: Path, rollout: dict[str, Any]) -> tuple[Path, str]:
    codex_home = task_root / ".codex"
    codex_home.mkdir(parents=True, exist_ok=True)

    encoded_auth = str(rollout.get("codex_auth_json_b64") or "").strip()
    auth_source = str(rollout.get("codex_auth_source") or "").strip()
    agent = rollout.get("harbor_agent")
    agent_env = dict(agent.get("env") or {}) if isinstance(agent, dict) else {}
    rollout_env = rollout_env_vars(rollout)
    explicit_api_key = str(
        rollout.get("openai_api_key")
        or agent_env.get("OPENAI_API_KEY")
        or rollout_env.get("OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ).strip()

    auth_path = codex_home / "auth.json"
    if explicit_api_key:
        auth_path.write_text(
            json.dumps({"OPENAI_API_KEY": explicit_api_key}),
            encoding="utf-8",
        )
        auth_source = "openai_api_key"
    elif encoded_auth:
        auth_json = base64.b64decode(encoded_auth.encode("ascii"), validate=True).decode("utf-8")
        auth_path.write_text(auth_json, encoding="utf-8")
        if not auth_source:
            auth_source = "chatgpt_credential"
    else:
        raise RuntimeError("Codex auth is unavailable for this rollout.")

    config_lines = ['approval_policy = "never"']
    openai_base_url = str(
        agent_env.get("OPENAI_BASE_URL")
        or rollout_env.get("OPENAI_BASE_URL")
        or ""
    ).strip()
    if openai_base_url:
        config_lines.append(
            f'openai_base_url = "{_toml_escape_double_quoted(openai_base_url)}"'
        )
    config_body = "\n".join(config_lines) + "\n"
    config_body += _append_managed_research_mcp_from_rollout(rollout)
    append_raw = str(_rollout_direct_or_metadata(rollout, "codex_config_toml_append") or "").strip()
    if append_raw:
        config_body += append_raw
        if not append_raw.endswith("\n"):
            config_body += "\n"
    (codex_home / "config.toml").write_text(config_body, encoding="utf-8")
    materialize_codex_bundled_skills(task_root, codex_home, rollout)
    return codex_home, auth_source


def build_codex_command(agent: dict[str, Any], instruction: str) -> list[str]:
    model_id = resolve_codex_model_id(agent)
    kwargs = dict(agent.get("kwargs") or {})
    command = [
        "codex",
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "--json",
        "--model",
        model_id,
    ]
    reasoning_effort = str(kwargs.get("reasoning_effort") or "").strip()
    if reasoning_effort:
        command.extend(["-c", f"model_reasoning_effort={reasoning_effort}"])
    reasoning_summary = str(kwargs.get("reasoning_summary") or "").strip()
    if reasoning_summary:
        command.extend(["-c", f"model_reasoning_summary={reasoning_summary}"])
    command.extend(["--", instruction])
    return command


def resolve_codex_model_id(agent: dict[str, Any]) -> str:
    requested_model_name = str(agent.get("model_name") or DEFAULT_MODEL_NAME).strip()
    requested_model_id = requested_model_name.split("/", 1)[-1]
    return _CODEX_MODEL_FALLBACKS.get(requested_model_id, requested_model_id)


def resolve_codex_timeout_seconds(
    rollout: dict[str, Any],
    agent: dict[str, Any],
) -> int:
    raw_candidates = [
        (agent.get("kwargs") or {}).get("codex_timeout_seconds"),
        rollout.get("codex_timeout_seconds"),
        _rollout_direct_or_metadata(rollout, "codex_timeout_seconds"),
        os.environ.get("HARBOR_CODEX_TIMEOUT_SECONDS"),
    ]
    for raw_value in raw_candidates:
        if raw_value is None:
            continue
        try:
            parsed = int(str(raw_value).strip())
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return DEFAULT_CODEX_TIMEOUT_SECONDS


def resolve_agent_log_dir(task_root: Path, workspace_dir: Path) -> Path:
    candidates = [
        Path("/logs/agent"),
        workspace_dir / "logs" / "agent",
        task_root / "logs" / "agent",
    ]
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except Exception:
            continue
    raise RuntimeError("unable to create Codex progress log directory")


def _write_status(
    status_path: Path,
    *,
    state: str,
    command: list[str],
    cwd: Path,
    stdout_log_path: Path,
    stderr_log_path: Path,
    returncode: int | None = None,
) -> None:
    payload: dict[str, Any] = {
        "state": state,
        "command": command,
        "cwd": str(cwd),
        "stdout_log_path": str(stdout_log_path),
        "stderr_log_path": str(stderr_log_path),
    }
    if returncode is not None:
        payload["returncode"] = returncode
    status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _stream_pipe(
    pipe: Any,
    *,
    log_path: Path,
    buffer: list[str],
) -> None:
    with log_path.open("a", encoding="utf-8") as sink:
        while True:
            chunk = pipe.readline()
            if not chunk:
                break
            buffer.append(chunk)
            sink.write(chunk)
            sink.flush()
    pipe.close()


def run_codex(
    task_root: Path,
    workspace_dir: Path,
    rollout: dict[str, Any],
    agent: dict[str, Any],
    *,
    codex_home: Path,
    auth_source: str,
) -> CodexRunResult:
    instruction = merge_instruction(task_root, rollout)
    env = os.environ.copy()
    env["HOME"] = str(task_root)
    env["CODEX_HOME"] = str(codex_home)
    env["CODEX_AUTH_SOURCE"] = auth_source
    for key, value in agent.get("env", {}).items():
        env[str(key)] = str(value)
    env.pop("OPENAI_BASE_URL", None)
    if not str(env.get("OPENAI_API_KEY") or "").strip():
        env.pop("OPENAI_API_KEY", None)

    command = build_codex_command(agent, instruction)
    log_dir = resolve_agent_log_dir(task_root, workspace_dir)
    stdout_log_path = log_dir / "codex_stdout.log"
    stderr_log_path = log_dir / "codex_stderr.log"
    status_path = log_dir / "codex_status.json"
    timeout_seconds = resolve_codex_timeout_seconds(rollout, agent)
    _write_status(
        status_path,
        state="running",
        command=command,
        cwd=workspace_dir,
        stdout_log_path=stdout_log_path,
        stderr_log_path=stderr_log_path,
    )

    process = subprocess.Popen(
        command,
        cwd=workspace_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    assert process.stderr is not None

    stdout_buffer: list[str] = []
    stderr_buffer: list[str] = []
    stdout_thread = threading.Thread(
        target=_stream_pipe,
        args=(process.stdout,),
        kwargs={"log_path": stdout_log_path, "buffer": stdout_buffer},
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_stream_pipe,
        args=(process.stderr,),
        kwargs={"log_path": stderr_log_path, "buffer": stderr_buffer},
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    timed_out = False
    deadline = time.monotonic() + timeout_seconds
    while True:
        returncode = process.poll()
        if returncode is not None:
            break
        if time.monotonic() >= deadline:
            timed_out = True
            process.terminate()
            try:
                returncode = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait()
            stderr_buffer.append(
                f"Codex timed out after {timeout_seconds}s and was terminated.\n"
            )
            with stderr_log_path.open("a", encoding="utf-8") as sink:
                sink.write(stderr_buffer[-1])
                sink.flush()
            break
        time.sleep(0.25)
    stdout_thread.join()
    stderr_thread.join()

    final_returncode = 124 if timed_out else returncode
    _write_status(
        status_path,
        state="timed_out" if timed_out else "completed",
        command=command,
        cwd=workspace_dir,
        stdout_log_path=stdout_log_path,
        stderr_log_path=stderr_log_path,
        returncode=final_returncode,
    )
    return CodexRunResult(
        returncode=final_returncode,
        stdout="".join(stdout_buffer),
        stderr="".join(stderr_buffer),
        stdout_log_path=str(stdout_log_path),
        stderr_log_path=str(stderr_log_path),
        status_path=str(status_path),
    )


def run_verifier(
    workspace_dir: Path,
    task_root: Path,
    *,
    rollout_env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    test_path = Path("/tests/test.sh")
    if not test_path.exists():
        test_path = task_root / "tests" / "test.sh"
    if not test_path.exists():
        raise RuntimeError(f"missing verifier script: {test_path}")
    env = os.environ.copy()
    env.update(rollout_env)
    local_tests_dir = task_root / "tests"
    local_logs_dir = workspace_dir / "logs"
    local_logs_dir.mkdir(parents=True, exist_ok=True)
    script_path = test_path
    if not Path("/tests").exists():
        staged_tests_dir = workspace_dir / ".goex_verifier_tests"
        if staged_tests_dir.exists():
            shutil.rmtree(staged_tests_dir)
        shutil.copytree(local_tests_dir, staged_tests_dir)
        tests_prefix = f"{staged_tests_dir.as_posix()}/"
        logs_prefix = f"{local_logs_dir.as_posix()}/"
        workdir_prefix = f"{workspace_dir.as_posix()}/"
        for candidate in staged_tests_dir.rglob("*"):
            if not candidate.is_file():
                continue
            try:
                text = candidate.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            rewritten = (
                text.replace("/tests/", "__GOEX_TESTS__/")
                .replace("/logs/", "__GOEX_LOGS__/")
                .replace("/workdir/", "__GOEX_WORKDIR__/")
                .replace("__GOEX_TESTS__/", tests_prefix)
                .replace("__GOEX_LOGS__/", logs_prefix)
                .replace("__GOEX_WORKDIR__/", workdir_prefix)
            )
            if rewritten != text:
                candidate.write_text(rewritten, encoding="utf-8")
        script_path = staged_tests_dir / "test.sh"
    return subprocess.run(
        ["bash", str(script_path)],
        cwd=workspace_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def load_verifier_score(*, workspace_dir: Path) -> tuple[float | None, str | None]:
    candidates = _candidate_paths(
        Path("/logs/verifier/reward.txt"),
        workspace_dir / "logs" / "verifier" / "reward.txt",
    )
    for path in candidates:
        if not path.exists():
            continue
        try:
            return float(path.read_text(encoding="utf-8").strip()), str(path)
        except Exception:
            continue
    return None, None


def _load_optional_text(*, candidates: list[Path]) -> tuple[str | None, str | None]:
    for path in candidates:
        if not path.exists():
            continue
        try:
            return path.read_text(encoding="utf-8"), str(path)
        except Exception:
            continue
    return None, None


def _read_text_if_exists(path: str | None) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    try:
        return candidate.read_text(encoding="utf-8")
    except Exception:
        return None


def _load_optional_json(*, candidates: list[Path]) -> tuple[dict[str, Any] | None, str | None]:
    for path in candidates:
        if not path.exists():
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8")), str(path)
        except Exception:
            continue
    return None, None


def load_verifier_outputs(
    *,
    workspace_dir: Path,
) -> dict[str, Any]:
    review, review_path = _load_optional_json(
        candidates=_candidate_paths(
            Path("/logs/verifier/review.json"),
            workspace_dir / "logs" / "verifier" / "review.json",
        )
    )
    rubric, rubric_path = _load_optional_json(
        candidates=_candidate_paths(
            Path("/logs/verifier/rubric.json"),
            workspace_dir / "logs" / "verifier" / "rubric.json",
        )
    )
    reward_text, reward_path = _load_optional_text(
        candidates=_candidate_paths(
            Path("/logs/verifier/reward.txt"),
            workspace_dir / "logs" / "verifier" / "reward.txt",
        )
    )
    return {
        "review": review,
        "review_path": review_path,
        "rubric": rubric,
        "rubric_path": rubric_path,
        "reward_text": reward_text,
        "reward_path": reward_path,
    }


def _artifact_entry(
    *,
    name: str,
    content: str,
    content_type: str | None = None,
    artifact_type: str = "custom",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_content_type = content_type or mimetypes.guess_type(name)[0] or "text/plain"
    return {
        "name": name,
        "artifact_type": artifact_type,
        "content_type": resolved_content_type,
        "content": content,
        "metadata": metadata or {},
    }


def build_rollout_result(
    *,
    rollout: dict[str, Any],
    deployment_name: str,
    agent: dict[str, Any],
    codex_result: CodexRunResult,
    verifier_result: subprocess.CompletedProcess[str],
    auth_source: str,
    verifier_score: float | None,
    verifier_score_path: str | None,
    verifier_outputs: dict[str, Any],
) -> dict[str, Any]:
    def _select_failure_error() -> str:
        verifier_stdout = verifier_result.stdout.strip()
        verifier_stderr = verifier_result.stderr.strip()
        codex_stderr = codex_result.stderr.strip()
        codex_stdout = codex_result.stdout.strip()

        # Some verifier wrappers exit 0 and write the real pytest failure
        # to stdout while emitting benign install chatter on stderr.
        if verifier_result.returncode == 0 and verifier_score is not None and verifier_score <= 0.0:
            return verifier_stdout or verifier_stderr or codex_stderr or codex_stdout or "Harbor rollout failed"
        return verifier_stderr or verifier_stdout or codex_stderr or codex_stdout or "Harbor rollout failed"

    score = verifier_score
    if score is None:
        score = 1.0 if codex_result.returncode == 0 and verifier_result.returncode == 0 else 0.0
    success = codex_result.returncode == 0 and verifier_result.returncode == 0 and score > 0.0
    metadata = {
        "deployment_name": deployment_name,
        "agent_name": str(agent.get("name") or ""),
        "model_name": str(agent.get("model_name") or ""),
        "effective_model_name": resolve_codex_model_id(agent),
        "auth_source": auth_source,
        "codex_returncode": codex_result.returncode,
        "verifier_returncode": verifier_result.returncode,
        "codex_stdout": codex_result.stdout[-2000:],
        "codex_stderr": codex_result.stderr[-2000:],
        "codex_stdout_log_path": codex_result.stdout_log_path,
        "codex_stderr_log_path": codex_result.stderr_log_path,
        "codex_status_path": codex_result.status_path,
        "verifier_stdout": verifier_result.stdout[-2000:],
        "verifier_stderr": verifier_result.stderr[-2000:],
        "verifier_score_path": verifier_score_path,
        "verifier_review_path": verifier_outputs.get("review_path"),
        "verifier_rubric_path": verifier_outputs.get("rubric_path"),
    }
    metrics = {
        "outcome_reward": score,
        "reward_mean": score,
        "rubric_score": score,
    }
    review = verifier_outputs.get("review")
    rubric = verifier_outputs.get("rubric")
    reward_text = verifier_outputs.get("reward_text")
    artifacts: list[dict[str, Any]] = []
    codex_trace_text = _read_text_if_exists(codex_result.stdout_log_path)
    codex_stderr_text = _read_text_if_exists(codex_result.stderr_log_path)
    if codex_trace_text:
        artifacts.append(
            _artifact_entry(
                name="codex_trace.jsonl",
                content=codex_trace_text,
                artifact_type="agent_output",
                metadata={"kind": "codex_event_trace"},
            )
        )
    if codex_stderr_text:
        artifacts.append(
            _artifact_entry(
                name="codex_stderr.log",
                content=codex_stderr_text,
                artifact_type="agent_output",
                metadata={"kind": "codex_stderr"},
            )
        )
    if isinstance(review, dict):
        artifacts.append(
            _artifact_entry(
                name="verifier_review.json",
                content=json.dumps(review, indent=2, sort_keys=True),
                artifact_type="custom",
                metadata={"kind": "paperbench_verifier_review"},
            )
        )
    if isinstance(rubric, dict):
        artifacts.append(
            _artifact_entry(
                name="rubric.json",
                content=json.dumps(rubric, indent=2, sort_keys=True),
                artifact_type="custom",
                metadata={"kind": "paperbench_rubric_definition"},
            )
        )
    if reward_text is not None:
        artifacts.append(
            _artifact_entry(
                name="rubric_score.txt",
                content=reward_text,
                artifact_type="custom",
                metadata={"kind": "paperbench_rubric_score"},
            )
        )
    return {
        "success": success,
        "trace_correlation_id": str(rollout.get("trace_correlation_id") or ""),
        "metrics": metrics,
        "score": score,
        "evaluation": {
            "rubric_score": score,
            "review": review,
            "rubric": rubric,
        },
        "artifacts": artifacts,
        "error": None if success else _select_failure_error(),
        "metadata": metadata,
    }


def run_rollout(input_path: Path, output_path: Path, task_root: Path) -> dict[str, Any]:
    rollout = load_rollout(input_path)
    agent = load_harbor_agent(rollout)
    deployment_name = str(rollout.get("deployment_name") or "unknown")
    codex_home, auth_source = materialize_codex_auth(task_root, rollout)
    workspace_dir = resolve_workspace_dir(task_root, rollout)
    rollout_env = rollout_env_vars(rollout)
    codex_result = run_codex(
        task_root,
        workspace_dir,
        rollout,
        agent,
        codex_home=codex_home,
        auth_source=auth_source,
    )
    verifier_result = run_verifier(
        workspace_dir,
        task_root,
        rollout_env=rollout_env,
    )
    verifier_score, verifier_score_path = load_verifier_score(workspace_dir=workspace_dir)
    verifier_outputs = load_verifier_outputs(workspace_dir=workspace_dir)
    result = build_rollout_result(
        rollout=rollout,
        deployment_name=deployment_name,
        agent=agent,
        codex_result=codex_result,
        verifier_result=verifier_result,
        auth_source=auth_source,
        verifier_score=verifier_score,
        verifier_score_path=verifier_score_path,
        verifier_outputs=verifier_outputs,
    )
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _failure_result(rollout: dict[str, Any], *, deployment_name: str, error: str) -> dict[str, Any]:
    return {
        "success": False,
        "trace_correlation_id": str(rollout.get("trace_correlation_id") or ""),
        "metrics": {"outcome_reward": 0.0},
        "score": 0.0,
        "error": error,
        "metadata": {
            "deployment_name": deployment_name,
            "agent_name": "codex",
            "model_name": str(
                ((rollout.get("harbor_agent") or {}).get("model_name")) or DEFAULT_MODEL_NAME
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-root", default="/app/task")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    task_root = Path(args.task_root)
    deployment_name = "unknown"
    try:
        rollout = load_rollout(input_path)
        deployment_name = str(rollout.get("deployment_name") or "unknown")
        result = run_rollout(input_path, output_path, task_root)
        return 0 if result.get("success") else 1
    except Exception as exc:
        rollout = {}
        if input_path.exists():
            try:
                rollout = load_rollout(input_path)
            except Exception:
                rollout = {}
        failure = _failure_result(rollout, deployment_name=deployment_name, error=str(exc))
        output_path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
