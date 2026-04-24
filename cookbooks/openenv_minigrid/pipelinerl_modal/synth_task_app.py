from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

try:
    import modal
except ImportError:  # standalone task app mode
    modal = None

if modal is not None:
    from modal_common import task_app_image

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TASKS_PATH = BASE_DIR / "example_tasks" / "agent_tasks.jsonl"
ENVIRONMENT_API_KEY = os.getenv("ENVIRONMENT_API_KEY")
PORT = int(os.getenv("PORT", "8001"))

app = FastAPI(title="NanoLong Synth Task App", version="0.1.0")
if modal is not None:
    modal_app = modal.App("nanolong-synth-task-app")
    image = task_app_image()


class EnvSpec(BaseModel):
    seed: int = Field(default=0)
    task_id: Optional[str] = None
    container_url: Optional[str] = None
    max_steps: int = Field(default=8)
    timeout_s: int = Field(default=120)
    mode: str = Field(default="candidate_output")


class PolicyConfig(BaseModel):
    prompt_template: str = Field(default="{{prompt}}")
    inference_url: Optional[str] = None
    model: Optional[str] = None
    candidate_output: Optional[str] = None
    messages: Optional[list[dict[str, Any]]] = None
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=1024)
    top_p: float = Field(default=1.0)
    logprobs: Optional[bool] = None


class PolicySpec(BaseModel):
    config: PolicyConfig


class RolloutRequest(BaseModel):
    env: EnvSpec
    inputs: dict[str, Any] = Field(default_factory=dict)
    policy: PolicySpec


class RolloutStep(BaseModel):
    reward: float
    info: dict[str, Any] = Field(default_factory=dict)


class RolloutTrajectory(BaseModel):
    steps: list[RolloutStep]


class RolloutResponse(BaseModel):
    metrics: dict[str, Any]
    trajectories: list[RolloutTrajectory]
    info: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def single_reward(reward: float, **extra: Any) -> "RolloutResponse":
        return RolloutResponse(
            metrics={"mean_return": reward},
            trajectories=[RolloutTrajectory(steps=[RolloutStep(reward=reward, info=extra.get("step_info", {}))])],
            info=extra.get("info", {}),
            artifacts=extra.get("artifacts", {}),
        )


# ------------ helpers ------------

def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if ENVIRONMENT_API_KEY and x_api_key != ENVIRONMENT_API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="unauthorized")


TASKS_CACHE: list[dict[str, Any]] | None = None


def load_tasks() -> list[dict[str, Any]]:
    global TASKS_CACHE
    if TASKS_CACHE is not None:
        return TASKS_CACHE
    path = Path(os.getenv("TASKS_PATH", str(DEFAULT_TASKS_PATH)))
    if not path.exists():
        raise FileNotFoundError(f"TASKS_PATH does not exist: {path}")
    tasks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
    if not tasks:
        raise ValueError(f"No tasks loaded from {path}")
    TASKS_CACHE = tasks
    return tasks



def resolve_task(req: RolloutRequest) -> dict[str, Any]:
    tasks = load_tasks()
    if req.env.task_id:
        for row in tasks:
            if row.get("task_id") == req.env.task_id:
                return row
        raise HTTPException(status_code=404, detail=f"unknown task_id={req.env.task_id!r}")
    idx = int(req.env.seed) % len(tasks)
    return tasks[idx]



def build_openai_chat_url(base: str) -> str:
    parsed = urlparse(base)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid inference_url: {base!r}")
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        final_path = path
    elif path.endswith("/v1"):
        final_path = f"{path}/chat/completions"
    else:
        final_path = f"{path}/v1/chat/completions"
    final_url = f"{parsed.scheme}://{parsed.netloc}{final_path}"
    if parsed.query:
        final_url = f"{final_url}?{parsed.query}"
    return final_url



def render_prompt(template: str, task: dict[str, Any], inputs: dict[str, Any]) -> str:
    text = template
    merged = {**task, **inputs}
    for key, value in merged.items():
        text = text.replace(f"{{{{{key}}}}}", str(value))
    if "{{prompt}}" in text and task.get("prompt"):
        text = text.replace("{{prompt}}", str(task["prompt"]))
    return text


async def call_openai_compatible(policy_cfg: PolicyConfig, task: dict[str, Any], inputs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if not policy_cfg.inference_url:
        raise HTTPException(status_code=400, detail="policy.config.inference_url is required for model-executed rollouts")
    url = build_openai_chat_url(policy_cfg.inference_url)
    messages = policy_cfg.messages or [
        {"role": "user", "content": render_prompt(policy_cfg.prompt_template, task, inputs)}
    ]
    body = {
        "model": policy_cfg.model or task.get("model") or "default",
        "messages": messages,
        "temperature": policy_cfg.temperature,
        "top_p": policy_cfg.top_p,
        "max_tokens": policy_cfg.max_tokens,
    }
    if policy_cfg.logprobs is not None:
        body["logprobs"] = policy_cfg.logprobs

    headers = {"Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=task.get("timeout_s", 120)) as client:
        resp = await client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
    message = ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")
    return str(message).strip(), data



def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())



def exact_match_reward(candidate: str, task: dict[str, Any]) -> float:
    target = task.get("answer") or task.get("expected_output") or task.get("target")
    if target is None:
        raise HTTPException(status_code=400, detail="task is missing answer/expected_output/target")
    candidate_n = normalize_text(candidate)
    target_n = normalize_text(str(target))
    if candidate_n == target_n:
        return 1.0
    if target_n and target_n in candidate_n:
        return 1.0
    return 0.0


async def score_via_container(candidate: str, task: dict[str, Any], req: RolloutRequest) -> tuple[float, dict[str, Any]]:
    container_url = req.env.container_url or task.get("container_url")
    if not container_url:
        raise HTTPException(status_code=400, detail="container_url is required for container scoring")
    url = f"{container_url.rstrip('/')}/score_candidate"
    payload = {
        "task": task,
        "inputs": req.inputs,
        "candidate_output": candidate,
        "seed": req.env.seed,
        "max_steps": req.env.max_steps,
        "timeout_s": req.env.timeout_s,
    }
    async with httpx.AsyncClient(timeout=req.env.timeout_s) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    reward = data.get("reward")
    if reward is None:
        metrics = data.get("metrics", {})
        reward = metrics.get("mean_return", metrics.get("reward_mean"))
    if reward is None:
        raise HTTPException(status_code=500, detail=f"Container scorer returned no reward: {data}")
    return float(reward), data


async def run_container_episode(policy_cfg: PolicyConfig, task: dict[str, Any], req: RolloutRequest) -> tuple[float, dict[str, Any]]:
    container_url = req.env.container_url or task.get("container_url")
    if not container_url:
        raise HTTPException(status_code=400, detail="container_url is required for long-horizon rollout mode")
    url = f"{container_url.rstrip('/')}/run_episode"
    payload = {
        "task": task,
        "inputs": req.inputs,
        "seed": req.env.seed,
        "max_steps": req.env.max_steps,
        "timeout_s": req.env.timeout_s,
        "policy": {
            "inference_url": policy_cfg.inference_url,
            "model": policy_cfg.model or task.get("model"),
            "prompt_template": policy_cfg.prompt_template,
            "temperature": policy_cfg.temperature,
            "top_p": policy_cfg.top_p,
            "max_tokens": policy_cfg.max_tokens,
            "candidate_output": policy_cfg.candidate_output,
        },
    }
    async with httpx.AsyncClient(timeout=req.env.timeout_s) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    reward = data.get("reward")
    if reward is None:
        metrics = data.get("metrics", {})
        reward = metrics.get("mean_return", metrics.get("reward_mean"))
    if reward is None:
        raise HTTPException(status_code=500, detail=f"Container environment returned no reward: {data}")
    return float(reward), data


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "tasks": len(load_tasks())}


@app.get("/task_info", dependencies=[Depends(require_api_key)])
def task_info() -> dict[str, Any]:
    tasks = load_tasks()
    sample = tasks[0]
    return {
        "id": "nanolong-agent-tasks",
        "name": "NanoLong agent tasks",
        "description": "Minimal task app for Qwen3.5 continual learning and RL experiments",
        "splits": ["train"],
        "input_schema": {"prompt": "string"},
        "output_schema": {"candidate_output": "string", "reward": "float"},
        "reward_range": [0.0, 1.0],
        "count": len(tasks),
        "example_task": sample,
        "tasks": [
            {
                "task_id": row.get("task_id"),
                "mode": row.get("mode", "candidate_output"),
                "reward_type": row.get("reward_type", "exact_match"),
            }
            for row in tasks[: min(16, len(tasks))]
        ],
    }


@app.post("/rollout", response_model=RolloutResponse, dependencies=[Depends(require_api_key)])
async def rollout(req: RolloutRequest) -> RolloutResponse:
    task = resolve_task(req)
    mode = req.env.mode or task.get("mode", "candidate_output")
    reward_type = task.get("reward_type", "exact_match")

    if mode == "candidate_output":
        candidate = req.policy.config.candidate_output
        if candidate is None:
            candidate, trace = await call_openai_compatible(req.policy.config, task, req.inputs)
        else:
            trace = {"mode": "candidate_output"}

        if reward_type == "container":
            reward, extra = await score_via_container(candidate, task, req)
        else:
            reward = exact_match_reward(candidate, task)
            extra = {"backend": "exact_match"}

        return RolloutResponse.single_reward(
            reward,
            info={"task_id": task.get("task_id"), "mode": mode, "reward_type": reward_type},
            artifacts={"candidate_output": candidate, "llm_trace": trace, "reward_trace": extra},
        )

    if mode == "container_episode":
        reward, trace = await run_container_episode(req.policy.config, task, req)
        trajectories = trace.get("trajectories")
        if trajectories:
            return RolloutResponse(
                metrics={"mean_return": reward},
                trajectories=trajectories,
                info={"task_id": task.get("task_id"), "mode": mode, "reward_type": reward_type},
                artifacts={"episode_trace": trace},
            )
        return RolloutResponse.single_reward(
            reward,
            info={"task_id": task.get("task_id"), "mode": mode, "reward_type": reward_type},
            artifacts={"episode_trace": trace},
        )

    raise HTTPException(status_code=400, detail=f"Unsupported mode={mode!r}")


if modal is not None:
    @modal_app.function(image=image)
    @modal.asgi_app()
    def fastapi_app():
        return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("synth_task_app:app", host="0.0.0.0", port=PORT, reload=False)
