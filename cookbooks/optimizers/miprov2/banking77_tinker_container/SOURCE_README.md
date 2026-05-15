# Banking77 Tinker Container

Public `synth-containers` runtime for Banking77 exact-label inference through
Tinker's Python `SamplingClient`. It uses Qwen's native tokenizer tool template
by default and parses the resulting `banking77_classify` tool call.

The default 4B baseline model is `Qwen/Qwen3.5-4B`. The older May ReportBench
gold pin, `Qwen/Qwen3-4B-Instruct-2507`, is still accepted as an override via
`policy.config.model`.

By default, rollouts use `label_mode=confusable7`, the same 7-intent confusable
slice as the Banking77 gold harness:

```text
pending_card_payment
pending_transfer
pending_top_up
pending_cash_withdrawal
cash_withdrawal_charge
cash_withdrawal_not_recognised
declined_cash_withdrawal
```

Set `policy.config.label_mode` to `full77` to evaluate against all Banking77
intents. Explicit `policy.config.labels` overrides the mode.

Run locally:

```bash
PYTHONPATH=packages/synth-containers/src:cookbooks/optimizers/miprov2/banking77_tinker_container \
TINKER_API_KEY="$TINKER_API_KEY" \
PORT=8943 python cookbooks/optimizers/miprov2/banking77_tinker_container/synth_service_app.py
```

Then call `/health`, `/task_info`, and `/rollout` on `http://127.0.0.1:8943`.

Minimal rollout payload:

```json
{
  "seed": 0,
  "policy": {
    "config": {
      "model": "Qwen/Qwen3.5-4B",
      "label_mode": "confusable7",
      "native_tool_calling": true,
      "labels": [
        "pending_card_payment",
        "pending_transfer",
        "pending_top_up",
        "pending_cash_withdrawal",
        "cash_withdrawal_charge",
        "cash_withdrawal_not_recognised",
        "declined_cash_withdrawal"
      ],
      "temperature": 0.0,
      "max_tokens": 256
    }
  }
}
```
