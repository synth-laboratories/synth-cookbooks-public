"""LoRA SFT smoke against the current synth-mlx-rl service (fake engine in tests)."""

from __future__ import annotations

import argparse
import json

from synth_mlx_rl import AdamParams, SamplingParams, ServiceClient

TRAINING = [
    [
        {"role": "system", "content": "You are a Craftax survival policy. Answer briefly."},
        {"role": "user", "content": "First resource to gather?"},
        {"role": "assistant", "content": "wood"},
    ],
    [
        {"role": "system", "content": "You are a Craftax survival policy. Answer briefly."},
        {"role": "user", "content": "Say the local learner readiness token."},
        {"role": "assistant", "content": "LOCAL_MLX_READY"},
    ],
]
PROBE = [
    {"role": "system", "content": "You are a Craftax survival policy. Answer briefly."},
    {"role": "user", "content": "Say the local learner readiness token."},
]


def run(*, base_url: str, steps: int) -> dict:
    with ServiceClient(base_url) as service:
        trainer = service.create_lora_training_client()
        datums = [
            trainer.datum_from_messages(messages, assistant_only=True) for messages in TRAINING
        ]
        baseline = trainer.save_weights_and_get_sampling_client()
        before = baseline.sample(PROBE, SamplingParams(max_tokens=32, temperature=0.0)).result()
        optimizer = AdamParams(learning_rate=5e-5, weight_decay=0.0, max_grad_norm=1.0)
        last_loss = None
        for _ in range(steps):
            forward = trainer.forward_backward(datums, "cross_entropy").result()
            trainer.optim_step(optimizer).result()
            last_loss = forward.loss
        trained = trainer.save_weights_and_get_sampling_client("sft-craftax-smoke")
        after = trained.sample(PROBE, SamplingParams(max_tokens=32, temperature=0.0)).result()
        recheck = baseline.sample(PROBE, SamplingParams(max_tokens=32, temperature=0.0)).result()
        return {
            "before": before.samples[0].text,
            "after": after.samples[0].text,
            "baseline_pin_held": recheck.samples[0].text == before.samples[0].text,
            "loss": last_loss,
            "checkpoint": "sft-craftax-smoke",
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8787")
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(base_url=args.base_url, steps=args.steps), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
