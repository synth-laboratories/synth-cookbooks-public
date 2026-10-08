# AGENT

Objective: local LoRA SFT, then Craftax fixture eval. Do not import MLX engine internals; use `ServiceClient` or `/v1/forward_backward`.

Authoritative: `cookbook.yaml`, `runbooks/train_sft.py`, `runbooks/eval_craftax_scripted.py`.

Safe writes: `.out/`, learner checkpoints named in the runbook.

Credentials: none for fake-engine smoke. Live MLX may download a base model.

Smoke: `runbooks/smoke.sh` (scripted Craftax + fake SFT step).

Completion: checkpoint id + fixture eval JSON. Failure: Craftax environment_ref not `env:craftax_fixture`, or SFT HTTP non-200.
