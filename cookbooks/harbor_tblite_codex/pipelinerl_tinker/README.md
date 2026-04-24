# PipelineRL via Tinker

This folder holds the optimizer side of the Harbor/TBLite Codex cookbook.

The MVP shape is intentionally split from the container implementation:

- `../tblite_codex_container/` owns the Harbor task pool, Codex worker runtime,
  reward, and verdict surface.
- this folder owns the PipelineRL/Tinker training loop that consumes rollout
  records from that runtime.
- `../run.py` is the public entrypoint that writes a small dry-run plan by
  default and can be extended to launch a real Tinker job once credentials and
  compute are configured.

## Inputs

Expected rollout records:

```json
{
  "task_id": "application-debug",
  "trace_correlation_id": "tblite-seed-0",
  "messages": [{"role": "user", "content": "Solve the Terminal-Bench Lite task."}],
  "observation": {"instruction": "Create result.txt ..."},
  "action": {"agent": "codex", "model": "openai/gpt-5.4-nano"},
  "reward": 0.0,
  "verdict": {"success": false, "error": "verifier output"}
}
```

## MVP Training Contract

The first Tinker loop should optimize the Codex worker policy against the
binary verifier reward emitted by Harbor/TBLite. Report the result as:

- `positive` when heldout reward improves
- `negative` when heldout reward regresses
- `inconclusive` when the run is underpowered or ties baseline

Do not claim an uplift from train reward alone.
