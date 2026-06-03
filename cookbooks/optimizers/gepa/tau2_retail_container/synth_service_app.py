"""
tau2-bench retail GEPA cookbook container.

Speaks the public synth-optimizers GEPA contract:
  GET  /health
  GET  /metadata   (also /info)
  GET  /task_info
  GET  /program
  GET  /dataset
  POST /dataset/rows
  POST /rollout    (also /rollouts)

Each rollout runs a real tau2 retail episode: an LLM customer-service agent,
the built-in tau2 user simulator, the retail tool environment over db.json, and
tau2's native evaluator. The mutable module is the retail domain policy.

If OPENAI_BASE_URL points at OpenRouter and OPENROUTER_API_KEY is set, the
container copies that key into OPENAI_API_KEY for tau2's LiteLLM/OpenAI calls.
This keeps rollout credentials separate from the GEPA proposer credential.
"""
from __future__ import annotations

import argparse
import os
import time
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Request

try:
    from synth_containers import GEPA_OPTIMIZER_CONTRACT_VERSION
except Exception:
    GEPA_OPTIMIZER_CONTRACT_VERSION = "synth_optimizers.gepa.v2"


TASK_ID = "tau2.retail_customer_service"
DATASET_ID = "tau2_retail"
DOMAIN = "retail"

AGENT_MODEL = os.environ.get("TAU2_RETAIL_AGENT_MODEL", "gpt-4.1-nano")
USER_MODEL = os.environ.get("TAU2_RETAIL_USER_MODEL", "gpt-4.1-nano")
MAX_STEPS = int(os.environ.get("TAU2_RETAIL_MAX_STEPS", "40"))
TIMEOUT_SECONDS = float(os.environ.get("TAU2_RETAIL_TIMEOUT_SECONDS", "180"))
DATA_DIR = Path(__file__).resolve().parent / "data"
os.environ.setdefault("TAU2_DATA_DIR", str(DATA_DIR))
if (
    os.environ.get("OPENAI_BASE_URL", "").find("openrouter.ai") >= 0
    and os.environ.get("OPENROUTER_API_KEY")
):
    os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]


def _tau2_imports() -> dict[str, Any]:
    try:
        from tau2.agent.llm_agent import LLMAgent
        from tau2.domains.retail.environment import get_environment, get_tasks
        from tau2.orchestrator.orchestrator import Orchestrator
        from tau2.runner.simulation import run_simulation
        from tau2.user.user_simulator import UserSimulator
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"tau2 dependency is unavailable or failed to import: {exc!r}",
        ) from exc
    return {
        "LLMAgent": LLMAgent,
        "UserSimulator": UserSimulator,
        "Orchestrator": Orchestrator,
        "run_simulation": run_simulation,
        "get_environment": get_environment,
        "get_tasks": get_tasks,
    }


def _get_policy() -> str:
    imports = _tau2_imports()
    env = imports["get_environment"]()
    return str(env.get_policy())


DEFAULT_DOMAIN_POLICY = _get_policy()


def _tasks_for_split(split: str) -> list[Any]:
    imports = _tau2_imports()
    split_name = "base" if split in {"train", "base"} else split
    try:
        tasks = imports["get_tasks"](split_name)
    except Exception:
        tasks = imports["get_tasks"](None)
    return list(tasks)


def _split_tasks(split: str) -> list[Any]:
    tasks = _tasks_for_split("base")
    if split in {"heldout", "test", "validation", "val"}:
        return tasks[20:] or tasks
    return tasks[:20] or tasks


def _task_for_seed(*, split: str, seed: int) -> Any:
    tasks = _split_tasks(split)
    return tasks[int(seed) % len(tasks)]


def _run_episode(split: str, seed: int, domain_policy: str) -> dict[str, Any]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not set.")

    imports = _tau2_imports()
    env = imports["get_environment"]()
    env.policy = domain_policy
    task = _task_for_seed(split=split, seed=seed)

    agent = imports["LLMAgent"](
        tools=env.get_tools(),
        domain_policy=domain_policy,
        llm=AGENT_MODEL,
    )
    user = imports["UserSimulator"](
        llm=USER_MODEL,
        instructions=str(task.user_scenario),
    )
    orchestrator = imports["Orchestrator"](
        domain=DOMAIN,
        agent=agent,
        user=user,
        environment=env,
        task=task,
        max_steps=MAX_STEPS,
        seed=seed,
        timeout=TIMEOUT_SECONDS,
    )
    simulation = imports["run_simulation"](orchestrator)
    reward_info = simulation.reward_info
    reward = float(getattr(reward_info, "reward", 0.0) or 0.0)
    messages = getattr(simulation, "messages", []) or []
    cost_usd = float(sum(float(getattr(m, "cost", 0.0) or 0.0) for m in messages))
    return {
        "example_id": str(task.id),
        "reward": reward,
        "reward_info": reward_info.model_dump() if reward_info is not None else {},
        "termination_reason": str(getattr(simulation, "termination_reason", "")),
        "n_messages": len(messages),
        "cost_usd": cost_usd,
    }


app = FastAPI(title="tau2-retail-gepa-container")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/metadata")
@app.get("/info")
async def metadata() -> dict[str, Any]:
    return {
        "runtime": {
            "runtime_id": "tau2_retail_gepa_live",
            "name": "tau2-bench retail GEPA",
            "description": "Runs real tau2 retail customer-service episodes with tau2's user simulator, tools, and evaluator.",
        },
        "capabilities": {
            "contract_version": "container_contract.v1",
            "rollout_modes": ["blocking"],
            "metadata": {"policy_ready": True, "trace_schema": "tau2.simulation.v1"},
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
            }
        },
    }


@app.get("/task_info")
async def task_info() -> dict[str, Any]:
    return {
        "task": {
            "task_id": TASK_ID,
            "name": "tau2-bench retail customer service",
            "description": "Optimize the retail domain policy for a customer-service LLM agent interacting with tau2's built-in user simulator and retail tools.",
            "objective": "Maximize tau2's native task reward.",
            "domain": "multi-turn tool-using retail customer support",
        },
        "dataset": {
            "dataset_id": DATASET_ID,
            "visible_splits": ["train", "test"],
            "default_split": "train",
            "seed_semantics": "Seed indexes deterministic task rows from tau2 retail's base task set.",
        },
        "prompt_program": {
            "mutable_modules": ["domain_policy"],
            "candidate_field": "domain_policy",
            "output_contract": "Policy text passed to tau2 LLMAgent as the retail domain policy.",
        },
        "evaluation": {
            "primary_metric": "tau2_reward",
            "success_status": "succeeded when reward is positive",
            "rollout_trace_contains": ["tau2_task_id", "termination_reason", "reward_info"],
        },
        "proposal_guidance": {
            "premises": [
                "The agent must follow retail policy while using tools across a multi-turn user simulation.",
                "Rewards come from tau2's evaluator over environment, action, and communication criteria.",
            ],
            "constraints": [
                "Do not encode task-specific customer details.",
                "Keep the policy concise enough for repeated multi-turn rollouts.",
            ],
            "high_leverage_heuristics": [
                "Tell the agent to collect required identifiers before tool calls.",
                "Make irreversible action checks explicit.",
                "State when the agent should ask clarifying questions instead of guessing.",
            ],
        },
        "metadata": {"agent_model": AGENT_MODEL, "user_model": USER_MODEL},
    }


@app.get("/program")
async def program() -> dict[str, Any]:
    return {
        "version": "prompt_program.v1",
        "program_id": "tau2_retail_policy_gepa",
        "modules": [
            {
                "module_id": "domain_policy",
                "role": "system",
                "content": DEFAULT_DOMAIN_POLICY,
                "mutable": True,
                "candidate_field": "domain_policy",
                "template_variables": [],
                "metadata": {"surface": "retail_policy_md"},
            }
        ],
        "target_modules": [
            {
                "module_id": "domain_policy",
                "candidate_field": "domain_policy",
                "objective": "tau2_reward",
            }
        ],
        "seed_candidate": {"domain_policy": DEFAULT_DOMAIN_POLICY},
        "rollout_overlay_schema": {"candidate_fields": ["domain_policy"]},
        "metadata": {"task_id": TASK_ID, "dataset_id": DATASET_ID},
    }


@app.get("/dataset")
async def dataset() -> dict[str, Any]:
    return {
        "dataset_id": DATASET_ID,
        "splits": {
            "train": len(_split_tasks("train")),
            "test": len(_split_tasks("test")),
        },
        "source": "Sierra tau2-bench retail domain (MIT)",
    }


@app.post("/dataset/rows")
async def dataset_rows(request: Request) -> dict[str, Any]:
    payload = await request.json()
    split = str(payload.get("split") or "train")
    seeds = [int(seed) for seed in payload.get("seeds") or []]
    rows = []
    for seed in seeds:
        task = _task_for_seed(split=split, seed=seed)
        rows.append({"seed": seed, "split": split, "example_id": str(task.id)})
    return {"rows": rows}


@app.get("/taskset")
async def taskset() -> dict[str, Any]:
    return {
        "taskset_id": DATASET_ID,
        "name": "tau2-bench retail",
        "splits": {
            "train": {"num_tasks": len(_split_tasks("train"))},
            "test": {"num_tasks": len(_split_tasks("test"))},
        },
    }


def _seed_from_task_id(task_id: Any) -> int:
    return int(str(task_id).rsplit(":", 1)[-1])


def _task_for_id(split: str, task_id: Any) -> dict[str, Any]:
    seed = _seed_from_task_id(task_id)
    task = _task_for_seed(split=split, seed=seed)
    return {
        "task_id": str(task_id),
        "seed": seed,
        "split": split,
        "example_id": str(task.id),
    }


@app.post("/taskset/tasks")
async def taskset_tasks(request: Request) -> dict[str, Any]:
    payload = await request.json()
    split = str(payload.get("split") or "train")
    task_ids = payload.get("task_ids") or []
    tasks = [_task_for_id(split, task_id) for task_id in task_ids]
    return {"tasks": tasks, "metadata": {"split": split, "count": len(tasks)}}


@app.post("/rollout")
@app.post("/rollouts")
def rollout(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    payload = payload or {}
    split = str(payload.get("split") or "train")
    seed = int(payload.get("seed") or 0)
    if isinstance(payload.get("dataset_row"), dict):
        row = payload["dataset_row"]
        split = str(row.get("split") or split)
        seed = int(row.get("seed") or seed)
    if isinstance(payload.get("task"), dict):
        task = payload["task"]
        split = str(task.get("split") or split)
        seed = int(task.get("seed") or seed)
    candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
    domain_policy = str(candidate.get("domain_policy") or DEFAULT_DOMAIN_POLICY)

    result = _run_episode(split, seed, domain_policy)
    reward = float(result["reward"])
    rollout_id = str(payload.get("rollout_id") or f"rollout_{uuid.uuid4().hex[:12]}")
    now = _now()
    return {
        "rollout_id": rollout_id,
        "status": "completed",
        "success_status": "succeeded" if reward > 0 else "failed",
        "task_id": TASK_ID,
        "seed": seed,
        "reward_info": {
            "outcome_reward": reward,
            "event_rewards": [reward],
            "details": {
                "example_id": result["example_id"],
                "tau2_reward_info": result["reward_info"],
                "termination_reason": result["termination_reason"],
                "agent_model": AGENT_MODEL,
                "user_model": USER_MODEL,
            },
        },
        "summary": {
            "outcome_reward": reward,
            "example_id": result["example_id"],
            "n_messages": result["n_messages"],
        },
        "usage": {"cost_usd": result["cost_usd"]},
        "trace": {
            "event_history": [
                {
                    "type": "tau2_episode_complete",
                    "example_id": result["example_id"],
                    "reward": reward,
                    "termination_reason": result["termination_reason"],
                }
            ],
            "metadata": {"example_id": result["example_id"], "call_site_id": TASK_ID},
        },
        "metadata": {"candidate": candidate},
        "created_at": now,
        "updated_at": now,
    }


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8774)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
