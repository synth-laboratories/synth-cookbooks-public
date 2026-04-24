"""Helpers for packaging real OpenThoughts TBLite tasks for Harbor."""

from __future__ import annotations

import base64
import io
import json
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen

OPEN_THOUGHTS_TBLITE_DATASET = "open-thoughts/OpenThoughts-TBLite"
HF_DATASET_TREE_URL = (
    "https://huggingface.co/api/datasets/open-thoughts/OpenThoughts-TBLite/tree/main/{path}"
    "?recursive=true&expand=true"
)
HF_DATASET_FILE_URL = (
    "https://huggingface.co/datasets/open-thoughts/OpenThoughts-TBLite/resolve/main/{path}"
)
_WORKDIR_RE = re.compile(r"^\s*WORKDIR\s+(.+?)\s*$", re.MULTILINE)
_UV_BOOTSTRAP_RE = re.compile(
    r"""# Install uv\s+curl -LsSf https://astral\.sh/uv/0\.9\.5/install\.sh \| sh\s+source \$HOME/\.local/bin/env""",
    re.MULTILINE,
)
_AGENT_VISIBLE_GOLDEN_FILENAMES = {
    "answer.json",
    "answers.json",
    "expected.json",
    "gold.json",
    "golden.json",
    "reference.json",
    "solution.json",
}
_AGENT_VISIBLE_GOLDEN_DIRS = {"__pycache__", "solution", "solutions"}


@dataclass(slots=True)
class PackagedTBLiteTask:
    task_id: str
    dockerfile: str
    context_tar_base64: str
    metadata: dict[str, Any]


def _download_json(url: str) -> list[dict[str, Any]]:
    with urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def _download_bytes(url: str) -> bytes:
    with urlopen(url) as response:
        return response.read()


def _list_task_paths(task_id: str) -> list[str]:
    payload = _download_json(HF_DATASET_TREE_URL.format(path=task_id))
    return [
        str(item.get("path") or "").strip()
        for item in payload
        if item.get("path") and str(item.get("type") or "").strip() == "file"
    ]


def _task_file_bytes(path: str) -> bytes:
    return _download_bytes(HF_DATASET_FILE_URL.format(path=quote(path, safe="/")))


def _extract_workspace_dir(dockerfile: str) -> str:
    matches = _WORKDIR_RE.findall(dockerfile)
    if not matches:
        return "/workspace"
    return matches[-1].strip()


def _append_file(tar: tarfile.TarFile, *, tar_path: str, content: bytes) -> None:
    info = tarfile.TarInfo(name=tar_path)
    info.size = len(content)
    tar.addfile(info, io.BytesIO(content))


def _is_agent_visible_golden_path(relative_path: str) -> bool:
    parts = [part.strip().lower() for part in Path(relative_path).parts if part.strip()]
    if not parts:
        return False
    if any(part in _AGENT_VISIBLE_GOLDEN_DIRS for part in parts):
        return True
    filename = parts[-1]
    return filename in _AGENT_VISIBLE_GOLDEN_FILENAMES or filename.endswith(".pyc")


def _patch_test_script(content: str) -> str:
    """Make verifier bootstrap deterministic when uv is already present.

    The upstream TBLite verifier always hits the network to install uv, which
    makes local Harbor runs flaky. If uvx already exists, skip that bootstrap.
    """

    replacement = """# Install uv only when the image does not already provide it
if command -v uvx >/dev/null 2>&1; then
    export PATH=\"$HOME/.local/bin:$PATH\"
elif [ -x \"$HOME/.local/bin/uvx\" ]; then
    export PATH=\"$HOME/.local/bin:$PATH\"
else
    curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh
    if [ -f \"$HOME/.local/bin/env\" ]; then
        source \"$HOME/.local/bin/env\"
    else
        export PATH=\"$HOME/.local/bin:$PATH\"
    fi
fi"""
    patched = _UV_BOOTSTRAP_RE.sub(replacement, content, count=1)
    return patched


def _build_wrapped_dockerfile(original_dockerfile: str) -> str:
    suffix = """

# Harbor Codex runner additions.
COPY --from=node:22-bookworm /usr/local/ /usr/local/
RUN if ! command -v python >/dev/null 2>&1; then \\
      apt-get update && apt-get install -y --no-install-recommends python3 python-is-python3 && rm -rf /var/lib/apt/lists/*; \\
    fi
RUN npm install -g @openai/codex@latest
RUN mkdir -p /app/task /tests /logs/verifier
COPY ./tb_task/instruction.md /app/task/instruction.md
COPY ./tb_task/task.toml /app/task/task.toml
COPY ./tb_task/tests /tests
COPY ./tb_task/run_codex_harbor_rollout.py /app/run_codex_harbor_rollout.py
"""
    return original_dockerfile.rstrip() + "\n" + suffix.lstrip("\n")


def package_open_thoughts_task(
    task_id: str,
    *,
    runner_path: Path,
) -> PackagedTBLiteTask:
    task_paths = _list_task_paths(task_id)
    if not task_paths:
        raise ValueError(f"No files found for OpenThoughts TBLite task {task_id!r}.")

    environment_paths = [path for path in task_paths if path.startswith(f"{task_id}/environment/")]
    if not environment_paths:
        raise ValueError(f"Task {task_id!r} is missing an environment/ directory.")

    original_dockerfile_path = f"{task_id}/environment/Dockerfile"
    if original_dockerfile_path not in task_paths:
        raise ValueError(f"Task {task_id!r} is missing environment/Dockerfile.")

    instruction_path = f"{task_id}/instruction.md"
    task_toml_path = f"{task_id}/task.toml"
    tests_paths = [path for path in task_paths if path.startswith(f"{task_id}/tests/")]

    original_dockerfile = _task_file_bytes(original_dockerfile_path).decode("utf-8")
    workspace_dir = _extract_workspace_dir(original_dockerfile)
    wrapped_dockerfile = _build_wrapped_dockerfile(original_dockerfile)

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in environment_paths:
            relative = path[len(f"{task_id}/environment/") :]
            if not relative or relative == "Dockerfile":
                continue
            if _is_agent_visible_golden_path(relative):
                continue
            _append_file(tar, tar_path=relative, content=_task_file_bytes(path))

        _append_file(
            tar,
            tar_path="tb_task/instruction.md",
            content=_task_file_bytes(instruction_path),
        )
        _append_file(
            tar,
            tar_path="tb_task/task.toml",
            content=_task_file_bytes(task_toml_path),
        )
        for path in tests_paths:
            relative = path[len(f"{task_id}/tests/") :]
            if not relative:
                continue
            content = _task_file_bytes(path)
            if relative == "test.sh":
                content = _patch_test_script(content.decode("utf-8")).encode("utf-8")
            _append_file(
                tar,
                tar_path=f"tb_task/tests/{relative}",
                content=content,
            )
        _append_file(
            tar,
            tar_path="tb_task/run_codex_harbor_rollout.py",
            content=runner_path.read_bytes(),
        )

    metadata = {
        "suite": "tb_lite",
        "benchmark_name": "terminal_bench_lite",
        "env_name": "terminal_bench_lite",
        "source_dataset": OPEN_THOUGHTS_TBLITE_DATASET,
        "source_task_id": task_id,
        "workspace_dir": workspace_dir,
        "agent_visible_golden_filters": {
            "filenames": sorted(_AGENT_VISIBLE_GOLDEN_FILENAMES),
            "directories": sorted(_AGENT_VISIBLE_GOLDEN_DIRS),
        },
    }
    return PackagedTBLiteTask(
        task_id=task_id,
        dockerfile=wrapped_dockerfile,
        context_tar_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
        metadata=metadata,
    )
