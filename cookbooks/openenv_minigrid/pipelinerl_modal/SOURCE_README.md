# NanoLong Qwen3.5 Modal pipeline

This starter repo is the practical path I would use for **B200 + `Qwen/Qwen3.5-0.8B-Base`** today:

- **mid-train + SFT** on current Hugging Face Transformers + Accelerate/Trainer
- **OPD** via **TRL GKDTrainer** configured in fully on-policy mode (`lmbda=1.0`, `beta=1.0`)
- **RL** via a lightweight **CISPO-style** trainer that clips sequence importance weights and uses Synth task-app rewards
- **serving / teacher / reward-time inference** through a separate **vLLM OpenAI-compatible** server on Modal

That split is intentional: Qwen3.5 is supported by recent Transformers and vLLM, but NeMo’s public recipes/support matrix still center on Qwen3 rather than `Qwen3.5-0.8B-Base`, and TRL’s documented vLLM integration range is narrower than the current Blackwell/Qwen3.5 guidance. So the repo keeps **training** and **serving** images separate.

## Files

- `train_midtrain_modal.py` — continual pretraining / mid-training
- `train_sft_modal.py` — supervised fine-tuning
- `train_opd_modal.py` — on-policy distillation through TRL GKD
- `train_rl_cispo_modal.py` — CISPO-style RL loop using exact-match or Synth task-app rewards
- `serve_vllm_modal.py` — Modal-deployed vLLM OpenAI-compatible server for Qwen3.5
- `synth_task_app.py` — Synth-compatible task app (`/health`, `/task_info`, `/rollout`)
- `smoke_test_qwen35_modal.py` — stage-0 smoke test for load + generate (+ optional vLLM compare)
- `compare_hf_vllm_outputs.py` — greedy-output parity check between HF and deployed vLLM
- `Dockerfile.synth-task-app` — containerize the task app for Synth managed deployment
- `example_tasks/agent_tasks.jsonl` — starter tasks
- `long_horizon_env_stub.py` — toy long-horizon container environment with `/score_candidate` and `/run_episode`
- `Dockerfile.long-horizon-env` — containerize the long-horizon environment stub
- `example_tasks/*_sample.jsonl` — starter datasets for each stage

## Recommended operating mode

### 1) Serve Qwen3.5 with vLLM on Modal

```bash
modal deploy serve_vllm_modal.py
```

Useful runtime settings (set `VLLM_GPU_CONFIG=B200:8` before deploy if you want tensor-parallel serving on all 8 GPUs):

```bash
modal run serve_vllm_modal.py \
  --model Qwen/Qwen3.5-0.8B-Base \
  --served-model-name qwen35-08b \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --enable-prefix-caching \
  --language-model-only
```

For Qwen3.5 text-first agent training, keep `--language-model-only` on at first. It avoids dragging the vision path into the train/RL stack.

### 1.5) Smoke-test the exact stack before training

```bash
modal run smoke_test_qwen35_modal.py --cmd '
  --model-id Qwen/Qwen3.5-0.8B-Base \
  --prompt "Say hello in one sentence." \
  --attn-implementation sdpa \
  --vllm-url https://YOUR-VLLM-ENDPOINT.modal.run \
  --served-model-name qwen35-08b
'
```

You can also run a small output-parity check against a prompt file:

```bash
python compare_hf_vllm_outputs.py \
  --model-id Qwen/Qwen3.5-0.8B-Base \
  --prompts-file /vol/data/smoke/prompts.jsonl \
  --vllm-url https://YOUR-VLLM-ENDPOINT.modal.run \
  --served-model-name qwen35-08b
```

### 2) Mid-train on agent traces / tool data

```bash
modal run train_midtrain_modal.py --cmd '
  --model-id Qwen/Qwen3.5-0.8B-Base \
  --train-files /vol/data/midtrain/*.jsonl \
  --output-dir /vol/checkpoints/midtrain \
  --block-size 8192 \
  --per-device-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-5 \
  --num-train-epochs 1 \
  --attn-implementation sdpa
'
```

Expected JSONL row shapes:

```json
{"text": "raw continual-pretraining text"}
```

or

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### 3) SFT on curated demonstrations

```bash
modal run train_sft_modal.py --cmd '
  --model-id /vol/checkpoints/midtrain/latest \
  --train-files /vol/data/sft/*.jsonl \
  --output-dir /vol/checkpoints/sft \
  --max-length 8192 \
  --per-device-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-5 \
  --num-train-epochs 1 \
  --attn-implementation sdpa
'
```

SFT JSONL rows can be:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

or

```json
{"prompt": "...", "response": "..."}
```

### 4) OPD with a stronger Qwen teacher

```bash
modal run train_opd_modal.py --cmd '
  --model-id /vol/checkpoints/sft/latest \
  --teacher-model-id Qwen/Qwen3.5-4B \
  --train-files /vol/data/opd/*.jsonl \
  --output-dir /vol/checkpoints/opd \
  --max-length 8192 \
  --max-new-tokens 1024 \
  --lmbda 1.0 \
  --beta 1.0 \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-5 \
  --attn-implementation sdpa
'
```

Prompt-only rows are allowed. The script will append an empty assistant stub so the trainer has a generation scaffold.

### 5) Run the Synth task app

Local dev:

```bash
python synth_task_app.py
```

Modal ASGI deployment:

```bash
modal deploy synth_task_app.py
```

Docker / managed Synth deployment:

```bash
docker build -f Dockerfile.synth-task-app -t nanolong-synth-task-app .
```

Then either:
- expose the task app locally behind a tunnel, or
- hand the Dockerfile + build context to the Synth managed deployment flow.

Before using container-backed tasks, you can run the toy environment stub locally:

```bash
python long_horizon_env_stub.py
```

or build it as a container:

```bash
docker build -f Dockerfile.long-horizon-env -t nanolong-long-horizon-env .
```

The task app supports two rollout modes:

- `candidate_output`: score a model completion already produced by your trainer
- `container_episode`: forward a long-horizon episode to your container environment (`/run_episode`)

For container-scored tasks, expose one or both of:

- `POST /score_candidate`
- `POST /run_episode`

on your environment container. The included `long_horizon_env_stub.py` implements both endpoints and gives you a minimal long-horizon contract to start from.

### 6) RL with CISPO-style updates and Synth rewards

```bash
modal run train_rl_cispo_modal.py --cmd '
  --model-id /vol/checkpoints/opd/latest \
  --train-files /vol/data/rl/*.jsonl \
  --output-dir /vol/checkpoints/rl \
  --reward-backend synth_task_app \
  --task-app-url https://YOUR-TASK-APP.modal.run \
  --environment-api-key YOUR_ENV_KEY \
  --prompts-per-device-batch-size 1 \
  --group-size 4 \
  --offpolicy-updates-per-rollout 4 \
  --max-prompt-length 4096 \
  --max-new-tokens 1024 \
  --learning-rate 5e-6 \
  --attn-implementation sdpa
'
```

A minimal RL JSONL row can be:

```json
{
  "task_id": "exact_math_0001",
  "prompt": "Solve exactly and respond with only the final numeric answer: 17 * 19",
  "answer": "323",
  "mode": "candidate_output"
}
```

A long-horizon container row can be:

```json
{
  "task_id": "container_episode_demo_0001",
  "prompt": "Use the container environment to inspect the repository and tell me which Python file defines the FastAPI app.",
  "container_url": "https://your-env.example.com",
  "mode": "container_episode",
  "max_steps": 12,
  "timeout_s": 180
}
```

## Notes on the CISPO trainer

The RL script is intentionally small and transparent. It does this:

1. sample groups of completions from the current policy
2. score them with exact-match or the Synth task app
3. normalize rewards groupwise (GRPO-style)
4. compute sequence importance ratios from old vs current policy logprobs
5. clip those **sequence weights** and keep all completion tokens in the objective

That is the important CISPO-inspired behavior. For a full production system, the next upgrade would be to pull multi-turn transcripts and step metadata back from the task app / container environment so the trainer can learn from richer traces than just final completions.

## Suggested rollout architecture

- use `serve_vllm_modal.py` for fast OpenAI-compatible inference
- point `policy.config.inference_url` in Synth rollouts at that vLLM endpoint
- let the task app call your container environment for `run_episode` / `score_candidate`
- keep trainer-side weight updates in the HF/Accelerate process

That keeps the Blackwell/Qwen3.5 serve stack independent from the trainer stack, which is the most robust way to get this running quickly.
