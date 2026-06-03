#!/usr/bin/env python3
"""Generate the Chart D proposer-scaling sweep configs.

One source of truth for the sweep matrix: TASKS x PROPOSERS. Proposer SIZE is the
only variable (all gpt-5.4 class — emphasizes size, not model generation); policy,
container, splits, and budget are held fixed per task.

Scope: HealthBench (rubric medical QA) and tau2-bench retail (tool-using
customer-service episodes).

  python gen_configs.py    # writes configs/proposer_sweep/<task>_<slug>.toml
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent / "configs" / "proposer_sweep"
CHATGPT_CODEX_HOME = "/Users/joshpurtell/.codex_chatgpt_jmvpurtell"
RUN_GROUP = "final_20260603"

# `slug` names the config file (full model); `short` names the run dir that
# build_chart.py scans (runs/<task>_<short>/result_manifest.json).
PROPOSERS = [
    {"slug": "gpt-5.4-nano", "short": "nano", "model": "gpt-5.4-nano", "effort": "low"},
    {"slug": "gpt-5.4-mini", "short": "mini", "model": "gpt-5.4-mini", "effort": "medium"},
    {"slug": "gpt-5.4", "short": "gpt54", "model": "gpt-5.4", "effort": "high"},
]

# Per-task fixed config (everything except the proposer).
TASKS = {
    "healthbench": {
        "container": "healthbench_container",
        "port": 8815,
        "module": "stage1_system",
        "seed_candidate": (
            "You are an expert physician. Answer the clinical question directly and "
            "accurately. Provide a clear, evidence-based response that addresses all "
            "aspects of the question. Include relevant considerations for diagnosis, "
            "treatment, or clinical management as appropriate. Be concise and precise."
        ),
        "policy_model": "google/gemini-2.5-flash-lite",
        "policy_base_url": "https://openrouter.ai/api/v1",
        "policy_api_key_env": "OPENROUTER_API_KEY",
        "train_seeds": list(range(10)),
        "heldout_seeds": list(range(25)),
        "max_generations": 3,
        "proposals_per_generation": 2,
        "minibatch_size": 5,
        "max_total_rollouts": 200,
    },
    "tau2_retail": {
        "container": "tau2_retail_container",
        "port": 8775,
        "module": "domain_policy",
        "seed_candidate": (
            "Use the retail policy exactly. Be concise, collect required information "
            "before taking actions, use tools when needed, and never promise an action "
            "unless it is allowed by policy."
        ),
        "policy_model": "openrouter/google/gemini-3.1-flash-lite",
        "policy_base_url": "https://openrouter.ai/api/v1",
        "policy_api_key_env": "OPENROUTER_API_KEY",
        "train_seeds": [0, 1, 2, 3],
        "heldout_seeds": [0, 1, 2, 3, 4],
        "max_generations": 3,
        "proposals_per_generation": 2,
        "minibatch_size": 2,
        "max_total_rollouts": 60,
    },
}


def _ids_block(split: str, seeds: list[int]) -> str:
    return "[" + ", ".join(f'"{split}:{seed}"' for seed in seeds) + "]"


def _proposer_auth_block(prop: dict) -> str:
    if prop["model"] == "gpt-5.4-nano":
        return """auth_mode = "api_key"
copy_host_auth = false
api_key_env = "OPENAI_API_KEY"
"""
    return f"""auth_mode = "chatgpt"
codex_home = "{CHATGPT_CODEX_HOME}"
copy_host_auth = false"""


def render(task: str, cfg: dict, prop: dict) -> str:
    slug = prop["slug"]
    short = prop["short"]
    run_dir = f"{RUN_GROUP}/{task}_{short}"
    proposer_auth = _proposer_auth_block(prop)
    return f"""# Chart D — Proposer scaling sweep: {task} x {slug} proposer
#
# Boot the {task} container on :{cfg['port']} first, then run from repo root:
#   uv run synth-optimizers gepa run --config \\
#     cookbooks/blogs/oss-containers-and-gepa/chart-d-proposer-scaling/configs/proposer_sweep/{task}_{slug}.toml

[run]
run_id = "{task}_proposer_{short}"
output_dir = "../../runs/{run_dir}"
seed = 0

[container]
url = "http://127.0.0.1:{cfg['port']}"
startup_timeout_seconds = 5

[taskset]
train_split = "train"
heldout_split = "test"
train_ids = {_ids_block("train", cfg['train_seeds'])}
heldout_ids = {_ids_block("test", cfg['heldout_seeds'])}

[candidate]
target_modules = ["{cfg['module']}"]

[seed_candidate]
{cfg['module']} = "{cfg['seed_candidate']}"

[policy]
provider = "openai"
model = "{cfg['policy_model']}"
base_url = "{cfg['policy_base_url']}"
api_key_env = "{cfg['policy_api_key_env']}"

[proposer]
backend = "codex_app_server"
execution_mode = "local_process"
timeout_seconds = 900
model = "{prop['model']}"
reasoning_effort = "{prop['effort']}"
{proposer_auth}
sandbox_mode = "workspace-write"
approval_policy = "never"

[gepa]
max_generations = {cfg['max_generations']}
proposals_per_generation = {cfg['proposals_per_generation']}
minibatch_size = {cfg['minibatch_size']}
max_total_rollouts = {cfg['max_total_rollouts']}
max_cost_usd = 0.0

[cache]
mode = "off"
path = "../../runs/{run_dir}/cache.sqlite"
namespace = "{task}_proposer_{short}"
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for task, cfg in TASKS.items():
        for prop in PROPOSERS:
            path = OUT / f"{task}_{prop['slug']}.toml"
            path.write_text(render(task, cfg, prop))
            written.append(path.name)
    print(f"wrote {len(written)} configs: {', '.join(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
