# Chart H: Reward Diagnostics

This chart backs the GEPA platform blog section that separates aggregate
coverage from task-specific reward signal.

`build_chart.py` reads the final GEPA eval evidence:

- `cookbooks/optimizers/gepa/evals/evidence/heldout_evaluations.jsonl`
- `cookbooks/optimizers/gepa/evals/evidence/benchmarks/harvey_lab/summary.json`
- `cookbooks/optimizers/gepa/evals/evidence/benchmarks/dungeongrid/summary.json`

It emits:

- `figures/reward_diagnostics_data.json`
- `frontend/src/components/blog/posts/introducing-gepa-platform/data/reward_diagnostics_data.json`

The Harvey panel uses the final best candidate from each stack and shows
fractional rubric reward for every heldout row. The DungeonGrid panel uses the
final best candidate from each stack and counts heldout episode achievement
signals such as `objective.item_recovered` directly from
`reward_details.achievements`.

Regenerate:

```bash
python cookbooks/blogs/oss-containers-and-gepa/chart-h-reward-diagnostics/build_chart.py
```
