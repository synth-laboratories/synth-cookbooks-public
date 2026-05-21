# Chart B — Qualitative Prompt Diff

Side-by-side display of the best-candidate prompts discovered by Synth
GEPA and gepa-ai on the same cookbook, with diffs highlighted. Lets
the reader see *how* each implementation's proposer rewrites prompts
differently, not just which scored higher.

## Cookbooks shown

Pick one cookbook per lever shape so the diff covers the full surface:

- **Banking77** — single-prompt classification.
- **TBLite** — agentic shell-edit.
- **Crafter** — long-horizon ReAct game loop.

(Code review is omitted from this chart because three-module diffs are
hard to read side-by-side; covered in the head-to-head table instead.)

## Layout

```
chart-b-prompt-diff/
  README.md
  source_runs.md               # Which run_id is the source for each pair
  prompts/
    banking77/
      synth_gepa_best.txt      # Extracted from best_candidate.json
      gepa_ai_best.txt
      diff.md                  # Annotated diff for the post
    tblite/
      synth_gepa_best.txt
      gepa_ai_best.txt
      diff.md
    crafter/
      synth_gepa_best.txt
      gepa_ai_best.txt
      diff.md
  extract_prompts.py           # Pulls best candidates from runs/
  figures/
    prompt_diff_banking77.svg  # Optional rendered diff for the post
    prompt_diff_tblite.svg
    prompt_diff_crafter.svg
```

## Reproduce

```bash
python extract_prompts.py      # Pulls best_candidate.json from upstream runs
# diff.md files are hand-annotated; re-run after each rerun
```

## Status

- [ ] Source runs picked for all 3 cookbooks.
- [ ] `extract_prompts.py` writes `prompts/*/synth_gepa_best.txt` and
      `prompts/*/gepa_ai_best.txt`.
- [ ] `diff.md` written for each pair with 1-2 sentence summary of what
      each implementation's proposer consistently does differently.
- [ ] Section in the blog MDX renders the pairs.
