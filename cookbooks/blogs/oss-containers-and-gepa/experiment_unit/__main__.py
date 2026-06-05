"""CLI for the blog experiment unit.

  uv run python -m experiment_unit              # status table (the executable audit)
  uv run python -m experiment_unit status       #   same
  uv run python -m experiment_unit show <key>   # per-check detail + reproduce command
  uv run python -m experiment_unit plan         # only what needs work, with commands
  uv run python -m experiment_unit packet       # publication paths + full dirty list
  uv run python -m experiment_unit run [opts]   # execute RERUN/MISSING experiments, parallel + 30m cap
      --time-limit N    hard per-arm cap in seconds (default 1800)
      --max-parallel N  concurrent arms (default 4)
      --only KEY        restrict to experiments matching KEY
      --dry-run         print the plan, run nothing
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from blog_paths import BLOG_ROOT, EVIDENCE_DIR, EXPERIMENTS_DIR, FRONTEND_DATA_DIR, FRONTEND_ROOT, REPO_ROOT
from .model import CheckStatus, ExperimentKind, Verdict
from .registry import in_scope

_MARK = {CheckStatus.PASS: "ok", CheckStatus.FAIL: "FAIL", CheckStatus.UNKNOWN: "?"}
_VERDICT_MARK = {
    Verdict.FINISHED: "FINISHED",
    Verdict.REPLUMB: "REPLUMB ",
    Verdict.RERUN: "RERUN   ",
    Verdict.MISSING: "MISSING ",
}


def _kind(exp) -> str:
    return "D draft" if exp.kind is ExperimentKind.PROPOSER_SWEEP else "A/C"


def _is_active_launch(exp) -> bool:
    return exp.kind is ExperimentKind.HEAD_TO_HEAD


def _active_launch_experiments():
    return tuple(exp for exp in in_scope() if _is_active_launch(exp))


def _draft_experiments():
    return tuple(exp for exp in in_scope() if not _is_active_launch(exp))


def cmd_status() -> int:
    print("active launch run evidence:")
    print(f"{'experiment':28} {'charts':7} {'verdict':9} note")
    print("-" * 88)
    for exp in _active_launch_experiments():
        v = exp.verdict()
        note = exp.mandate.split(":")[0] if exp.mandate else _status_note(exp, v)
        print(f"{exp.label + ' (' + _kind(exp) + ')':28} {'/'.join(exp.charts):7} "
              f"{_VERDICT_MARK[v]} {note}")

    draft = _draft_experiments()
    if draft:
        print()
        print("post-launch draft/debug run evidence:")
        print(f"{'experiment':28} {'charts':7} {'verdict':9} note")
        print("-" * 88)
        for exp in draft:
            v = exp.verdict()
            note = exp.mandate.split(":")[0] if exp.mandate else _status_note(exp, v)
            print(f"{exp.label + ' (' + _kind(exp) + ')':28} {'/'.join(exp.charts):7} "
                  f"{_VERDICT_MARK[v]} {note}")
    print()
    counts: dict[Verdict, int] = {}
    for exp in _active_launch_experiments():
        counts[exp.verdict()] = counts.get(exp.verdict(), 0) + 1
    print("active summary: " + "  ".join(f"{v.value}={n}" for v, n in counts.items()))
    if draft:
        draft_counts: dict[Verdict, int] = {}
        for exp in draft:
            draft_counts[exp.verdict()] = draft_counts.get(exp.verdict(), 0) + 1
        print("draft summary: " + "  ".join(f"{v.value}={n}" for v, n in draft_counts.items()))
    _print_publication_packet()
    return 0


def _status_note(exp, v: Verdict) -> str:
    if v is Verdict.FINISHED:
        if exp.kind is ExperimentKind.PROPOSER_SWEEP:
            return "post-launch Chart D manifests present, numeric, and metric semantics checked"
        return "evidence present, parity checks pass, candidate/rollout floor passes, in live aggregate"
    failed = [c.name for c in exp.checks() if c.status is CheckStatus.FAIL]
    return "failing: " + ", ".join(failed) if failed else ""


def cmd_show(key: str) -> int:
    matches = [e for e in in_scope() if key in e.key or key in e.label.lower()]
    if not matches:
        print(f"no experiment matching {key!r}", file=sys.stderr)
        return 1
    for exp in matches:
        print(f"\n=== {exp.label}  [{exp.kind.value}]  charts {'/'.join(exp.charts)} ===")
        print(f"container={exp.container}  arms={', '.join(exp.arms)}")
        p = exp.parity
        print(f"lock: proposer={p.proposer_model}/{p.proposer_auth}  "
              f"policy={p.policy_model}  route={p.policy_route}")
        if exp.mandate:
            print(f"MANDATE (forces RERUN): {exp.mandate}")
        else:
            for c in exp.checks():
                print(f"  [{_MARK[c.status]:>4}] {c.name:18} {c.detail}")
        print(f"run verdict: {exp.verdict().value.upper()}")
        print("reproduce:")
        for line in exp.reproduce():
            print(f"  {line}")
    print("\nPublication readiness is reported by `python -m experiment_unit status`.")
    return 0


def cmd_plan() -> int:
    todo = [e for e in _active_launch_experiments() if e.verdict() is not Verdict.FINISHED]
    if not todo:
        print("all active launch run evidence FINISHED.")
        _print_publication_packet()
    else:
        for exp in todo:
            print(f"\n# {exp.label} ({'/'.join(exp.charts)}) -> {exp.verdict().value.upper()}")
            if exp.mandate:
                print(f"#   reason: {exp.mandate}")
            for line in exp.reproduce():
                print(f"  {line}")

    draft_todo = [e for e in _draft_experiments() if e.verdict() is not Verdict.FINISHED]
    if draft_todo:
        print()
        print("# Post-launch draft/debug work (not part of the launch gate)")
        for exp in draft_todo:
            print(f"\n# {exp.label} ({'/'.join(exp.charts)}) -> {exp.verdict().value.upper()}")
            if exp.mandate:
                print(f"#   reason: {exp.mandate}")
            for line in exp.reproduce():
                print(f"  {line}")
    return 0


def cmd_packet() -> int:
    print("publication packet required paths:")
    for path in _launch_publication_paths():
        print(f"  {_rel_path(path)}")
    print()
    print(f"frontend mirror root: {FRONTEND_ROOT}")

    refs, missing_refs, mismatched_refs = _source_evidence_refs(_source_evidence_files())
    print()
    print("source-evidence refs:")
    for path in sorted(refs):
        print(f"  {_rel_path(path)}")
    if missing_refs:
        print()
        print("missing source-evidence refs:")
        for path in missing_refs:
            print(f"  {path}")
    if mismatched_refs:
        print()
        print("stale source-evidence refs:")
        for detail in mismatched_refs:
            print(f"  {detail}")

    print()
    _print_dirty_paths("required dirty status", _launch_publication_paths())

    print()
    print("Chart D draft/debug paths (warning only):")
    for path in _chart_d_draft_paths():
        print(f"  {_rel_path(path)}")
    print()
    _print_dirty_paths("Chart D draft/debug dirty status", _chart_d_draft_paths())

    print()
    print("TBLite quarantine paths (warning only):")
    for path in _tblite_quarantine_paths():
        print(f"  {_rel_path(path)}")
    print()
    _print_dirty_paths("TBLite quarantine dirty status", _tblite_quarantine_paths())

    residual_dirty, residual_error = _residual_gepa_dirty_status()
    print()
    print("Other GEPA workspace dirty status (warning only):")
    if residual_error:
        print(f"  ERROR {residual_error}")
    elif not residual_dirty:
        print("  clean")
    else:
        groups = _dirty_groups(residual_dirty)
        print("  " + ", ".join(f"{name}={count}" for name, count in groups))
        for line in residual_dirty:
            print(f"  {line}")
    return 0


def cmd_run(argv: list[str]) -> int:
    from .runner import DEFAULT_TIME_LIMIT, run_experiments

    time_limit, max_parallel, only, dry_run = DEFAULT_TIME_LIMIT, 4, "", False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--time-limit":
            time_limit = int(argv[i + 1]); i += 2
        elif a == "--max-parallel":
            max_parallel = int(argv[i + 1]); i += 2
        elif a == "--only":
            only = argv[i + 1]; i += 2
        elif a == "--dry-run":
            dry_run = True; i += 1
        else:
            print(f"unknown run option {a!r}", file=sys.stderr)
            return 1
    candidates = in_scope() if only else _active_launch_experiments()
    todo = [
        e for e in candidates
        if e.verdict() in {Verdict.RERUN, Verdict.MISSING}
        and (not only or only in e.key or only in e.label.lower())
    ]
    if not todo:
        print("nothing to run (no matching active-launch RERUN/MISSING experiments).")
        return 0
    return run_experiments(todo, time_limit, max_parallel, dry_run)


@dataclass(frozen=True)
class PacketCheck:
    name: str
    ok: bool
    detail: str
    required: bool = True


_ACTIVE_RECORD_NAMES = (
    "healthbench_pro__chart_a__synth_gepa",
    "healthbench_pro__chart_a__gepa_ai",
    "banking77__chart_a__synth_gepa",
    "banking77__chart_a__gepa_ai",
    "hotpotqa__chart_a__synth_gepa",
    "hotpotqa__chart_a__gepa_ai",
    "tau2_retail__chart_a__synth_gepa",
    "tau2_retail__chart_a__gepa_ai",
)

_DRAFT_RECORD_NAMES = (
    "healthbench_pro__chart_d__gpt-5.4-nano",
    "healthbench_pro__chart_d__gpt-5.4-mini",
    "healthbench_pro__chart_d__gpt-5.4",
    "tau2_retail__chart_d__gpt-5.4-nano",
    "tau2_retail__chart_d__gpt-5.4-mini",
    "tau2_retail__chart_d__gpt-5.4",
)


def _record_readmes(names: tuple[str, ...]) -> list[Path]:
    return [
        EXPERIMENTS_DIR / name / "README.md"
        for name in names
    ]


def _expected_record_readmes() -> list[Path]:
    return _record_readmes(_ACTIVE_RECORD_NAMES)


def _draft_record_readmes() -> list[Path]:
    return _record_readmes(_DRAFT_RECORD_NAMES)


def _publication_packet_checks() -> list[PacketCheck]:
    checks: list[PacketCheck] = []
    source_evidence = _source_evidence_files()
    missing_sources = [str(path.relative_to(REPO_ROOT)) for path in source_evidence if not path.exists()]
    checks.append(PacketCheck(
        "chart_source_evidence",
        not missing_sources,
        "A/C source_evidence.json present" if not missing_sources
        else "missing " + ", ".join(missing_sources),
    ))
    checks.append(_source_evidence_refs_check(source_evidence))

    checks.append(PacketCheck(
        "frontend_checkout",
        FRONTEND_DATA_DIR.exists(),
        f"checking {FRONTEND_ROOT}",
    ))

    checks.append(_frontend_mirrors_check())

    checks.append(_launch_copy_guard_check())

    chart_d_semantics = _chart_d_metric_semantics_check()
    checks.append(PacketCheck(
        chart_d_semantics.name,
        chart_d_semantics.ok,
        chart_d_semantics.detail,
        required=False,
    ))

    record_readmes = _expected_record_readmes()
    present_records = [path for path in record_readmes if path.exists()]
    checks.append(PacketCheck(
        "experiment_records_backfill",
        len(present_records) == len(record_readmes),
        f"{len(present_records)}/{len(record_readmes)} active A/C per-cell README records present",
    ))

    draft_record_readmes = _draft_record_readmes()
    present_draft_records = [path for path in draft_record_readmes if path.exists()]
    checks.append(PacketCheck(
        "draft_chart_d_records",
        len(present_draft_records) == len(draft_record_readmes),
        f"{len(present_draft_records)}/{len(draft_record_readmes)} draft Chart D README records present",
        required=False,
    ))

    dirty_detail = _dirty_publication_paths()
    checks.append(PacketCheck(
        "public_evidence_commit",
        dirty_detail is None,
        "tracked in current commit" if dirty_detail is None else dirty_detail,
    ))

    ignored_detail = _ignored_paths_detail(_ignored_launch_nuisance_paths(), "launch-adjacent")
    checks.append(PacketCheck(
        "ignored_launch_nuisance",
        ignored_detail is None,
        "no ignored local launch-adjacent nuisance files" if ignored_detail is None else ignored_detail,
        required=False,
    ))

    chart_d_detail = _dirty_paths_detail(_chart_d_draft_paths(), "Chart D draft/debug")
    checks.append(PacketCheck(
        "chart_d_draft_dirty",
        chart_d_detail is None,
        "clean or separately tracked" if chart_d_detail is None else chart_d_detail,
        required=False,
    ))

    tblite_detail = _dirty_paths_detail(_tblite_quarantine_paths(), "TBLite quarantine")
    checks.append(PacketCheck(
        "tblite_quarantine_dirty",
        tblite_detail is None,
        "clean or separately tracked" if tblite_detail is None else tblite_detail,
        required=False,
    ))

    residual_detail = _residual_gepa_dirty_detail()
    checks.append(PacketCheck(
        "gepa_workspace_dirty",
        residual_detail is None,
        "clean outside launch packet and TBLite quarantine"
        if residual_detail is None else residual_detail,
        required=False,
    ))
    return checks


def _source_evidence_files() -> list[Path]:
    return [
        BLOG_ROOT / "charts" / "chart-a-head-to-head" / "figures" / "source_evidence.json",
        BLOG_ROOT / "charts" / "chart-c-use-case-coverage" / "figures" / "source_evidence.json",
    ]


def _frontend_mirror_pairs() -> list[tuple[Path, Path]]:
    return [
        (
            BLOG_ROOT / "charts" / "chart-a-head-to-head" / "figures" / "head_to_head_data.json",
            FRONTEND_DATA_DIR / "core_head_to_head_data.json",
        ),
        (
            BLOG_ROOT / "charts" / "chart-c-use-case-coverage" / "figures" / "use_case_heldout_coverage_data.json",
            FRONTEND_DATA_DIR / "use_case_heldout_coverage_data.json",
        ),
    ]


def _launch_copy_guard_files() -> list[Path]:
    return [
        BLOG_ROOT / "README.md",
        BLOG_ROOT / "charts" / "README.md",
        BLOG_ROOT / "charts" / "chart-a-head-to-head" / "README.md",
        BLOG_ROOT / "charts" / "chart-a-head-to-head" / "build_chart.py",
        BLOG_ROOT / "charts" / "chart-a-head-to-head" / "figures" / "head_to_head.svg",
        BLOG_ROOT / "charts" / "chart-c-use-case-coverage" / "README.md",
    ]


def _launch_copy_guard_check() -> PacketCheck:
    banned = (
        "1932ee",
        "current draft",
        "draft-local",
        "draft post",
        "draft configs",
        "draft manifests",
        "matched-budget",
        "matched budgets",
        "same budget",
        "apples-to-apples",
        "pre-rerun",
        "most ready",
        "hard ground truth",
        "ungameable",
        "ungamable",
    )
    failures: list[str] = []
    for path in _launch_copy_guard_files():
        if not path.exists():
            failures.append(f"{_rel_path(path)}: missing")
            continue
        text = path.read_text(errors="replace").lower()
        for term in banned:
            if term in text:
                failures.append(f"{_rel_path(path)}: stale launch copy term {term!r}")
    return PacketCheck(
        "launch_copy_guard",
        not failures,
        "active launch docs/producers avoid stale draft or overclaim terms"
        if not failures else "; ".join(failures[:5]) + ("; ..." if len(failures) > 5 else ""),
    )


def _frontend_mirrors_check() -> PacketCheck:
    failures: list[str] = []
    for source, mirror in _frontend_mirror_pairs():
        source_label = _rel_path(source)
        mirror_label = str(mirror)
        if not source.exists():
            failures.append(f"{source_label}: missing producer output")
            continue
        if not mirror.exists():
            failures.append(f"{mirror_label}: missing frontend mirror")
            continue
        source_bytes = source.read_bytes()
        mirror_bytes = mirror.read_bytes()
        if source_bytes != mirror_bytes:
            failures.append(
                f"{mirror_label}: differs from {source_label} "
                f"({hashlib.sha256(mirror_bytes).hexdigest()[:12]} != "
                f"{hashlib.sha256(source_bytes).hexdigest()[:12]})"
            )
    return PacketCheck(
        "frontend_mirrors",
        not failures,
        "A/C frontend mirrors byte-match producer output"
        if not failures else "; ".join(failures[:3]) + ("; ..." if len(failures) > 3 else ""),
    )


def _iter_repo_path_refs(value) -> list[tuple[Path, str | None, int | None]]:
    refs: list[tuple[Path, str | None, int | None]] = []
    if isinstance(value, dict):
        for key in ("path", "manifest_snapshot_path"):
            ref = value.get(key)
            if isinstance(ref, str) and ref.startswith("cookbooks/"):
                refs.append((
                    REPO_ROOT / ref,
                    value.get("sha256") if isinstance(value.get("sha256"), str) else None,
                    value.get("bytes") if isinstance(value.get("bytes"), int) else None,
                ))
        for child in value.values():
            refs.extend(_iter_repo_path_refs(child))
    elif isinstance(value, list):
        for child in value:
            refs.extend(_iter_repo_path_refs(child))
    return refs


def _source_evidence_refs_check(source_evidence: list[Path]) -> PacketCheck:
    refs, missing, mismatched = _source_evidence_refs(source_evidence)
    failures = missing + mismatched
    return PacketCheck(
        "source_evidence_refs",
        not failures,
        f"{len(refs)} source-evidence refs exist and hashes match"
        if not failures else "; ".join(failures[:5]) + ("; ..." if len(failures) > 5 else ""),
    )


def _source_evidence_refs(source_evidence: list[Path]) -> tuple[set[Path], list[str], list[str]]:
    missing: list[str] = []
    mismatched: list[str] = []
    refs: set[Path] = set()
    seen_hashes: set[tuple[Path, str | None, int | None]] = set()
    for path in source_evidence:
        if not path.exists():
            continue
        try:
            for ref, expected_sha, expected_bytes in _iter_repo_path_refs(json.loads(path.read_text())):
                refs.add(ref)
                seen_hashes.add((ref, expected_sha, expected_bytes))
        except json.JSONDecodeError as exc:
            missing.append(f"{path.relative_to(REPO_ROOT)} invalid json ({exc})")
    for ref in sorted(refs):
        if not ref.exists():
            missing.append(str(ref.relative_to(REPO_ROOT)))
    for ref, expected_sha, expected_bytes in sorted(seen_hashes, key=lambda item: str(item[0])):
        if not ref.exists() or (expected_sha is None and expected_bytes is None):
            continue
        data = ref.read_bytes()
        actual_sha = hashlib.sha256(data).hexdigest()
        actual_bytes = len(data)
        rel = str(ref.relative_to(REPO_ROOT))
        if expected_sha is not None and actual_sha != expected_sha:
            mismatched.append(f"{rel}: sha256 {actual_sha} != {expected_sha}")
        if expected_bytes is not None and actual_bytes != expected_bytes:
            mismatched.append(f"{rel}: bytes {actual_bytes} != {expected_bytes}")
    return refs, missing, mismatched


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _chart_d_metric_semantics_check() -> PacketCheck:
    paths = [
        (
            BLOG_ROOT / "charts" / "chart-d-proposer-scaling" / "figures" / "proposer_scaling_data.json",
            True,
        ),
        (
            FRONTEND_DATA_DIR / "proposer_scaling_data.json",
            False,
        ),
    ]
    failures: list[str] = []
    notes: list[str] = []
    for path, required in paths:
        label = str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)
        if not path.exists():
            if required:
                failures.append(f"{label}: missing")
            else:
                notes.append(f"{label}: absent because Chart D is removed from the active frontend")
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            failures.append(f"{label}: invalid json ({exc})")
            continue
        if "heldout scoring was skipped" not in data.get("description", ""):
            failures.append(f"{label}: description does not state heldout scoring was skipped")
        cells = data.get("cells")
        if not isinstance(cells, list):
            failures.append(f"{label}: cells is not a list")
            continue
        for cell in cells:
            cell_label = f"{label}:{cell.get('task')}x{cell.get('proposer_slug')}"
            if "seed_reward" in cell:
                failures.append(f"{cell_label}: ambiguous seed_reward present")
            for field in (
                "initial_observed_reward",
                "comparison_heldout_seed_reward",
                "best_observed_reward",
            ):
                if not _is_number(cell.get(field)):
                    failures.append(f"{cell_label}: {field}={cell.get(field)!r}")
            if cell.get("best_observed_reward_source") not in {"heldout_reward", "train_reward"}:
                failures.append(
                    f"{cell_label}: best_observed_reward_source={cell.get('best_observed_reward_source')!r}"
                )
            curve = cell.get("curve")
            if not isinstance(curve, dict):
                failures.append(f"{cell_label}: curve missing")
                continue
            if "seed" in curve:
                failures.append(f"{cell_label}: ambiguous curve.seed present")
            if not _is_number(curve.get("initial")):
                failures.append(f"{cell_label}: curve.initial={curve.get('initial')!r}")

    md_path = BLOG_ROOT / "charts" / "chart-d-proposer-scaling" / "figures" / "proposer_scaling.md"
    if md_path.exists() and "| seed |" in md_path.read_text():
        failures.append(f"{md_path.relative_to(REPO_ROOT)}: ambiguous seed column present")

    return PacketCheck(
        "chart_d_metric_semantics",
        not failures,
        "Chart D separates observed reward from heldout seed context"
        + (f"; {'; '.join(notes)}" if notes else "")
        if not failures else "; ".join(failures[:3]) + ("; ..." if len(failures) > 3 else ""),
    )


def _dirty_publication_paths() -> str | None:
    return _dirty_paths_detail(_launch_publication_paths(), "launch evidence")


def _launch_publication_paths() -> list[Path]:
    launch_benchmarks = ("healthbench", "tau2_retail", "banking77", "hotpotqa")
    return [
        BLOG_ROOT / "README.md",
        BLOG_ROOT / "charts" / "README.md",
        BLOG_ROOT / "charts" / "chart-a-head-to-head",
        BLOG_ROOT / "charts" / "chart-c-use-case-coverage",
        BLOG_ROOT / "experiment_unit",
        BLOG_ROOT / "experiment_records" / "README.md",
        *_expected_record_readmes(),
        EVIDENCE_DIR / "heldout_evaluations.jsonl",
        EVIDENCE_DIR / "train_evaluations.jsonl",
        EVIDENCE_DIR / "candidate_timeline.jsonl",
        EVIDENCE_DIR / "commands.jsonl",
        *(EVIDENCE_DIR / "benchmarks" / benchmark for benchmark in launch_benchmarks),
    ]


def _chart_d_draft_paths() -> list[Path]:
    return [
        BLOG_ROOT / "charts" / "chart-d-proposer-scaling",
        *_draft_record_readmes(),
    ]


def _tblite_quarantine_paths() -> list[Path]:
    return [REPO_ROOT / "cookbooks" / "optimizers" / "gepa" / "tblite_container"]


def _gepa_workspace_paths() -> list[Path]:
    return [REPO_ROOT / "cookbooks" / "optimizers" / "gepa"]


def _rel_path(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def _dirty_paths_detail(paths: list[Path], label: str) -> str | None:
    dirty, error = _dirty_status(paths)
    if error:
        return error
    if not dirty:
        return None
    groups = _dirty_groups(dirty)
    group_detail = ", ".join(f"{name}={count}" for name, count in groups)
    sample = "; ".join(dirty[:6])
    suffix = "; ..." if len(dirty) > 6 else ""
    return f"{len(dirty)} dirty/uncommitted {label} paths ({group_detail}; examples: {sample}{suffix})"


def _residual_gepa_dirty_detail() -> str | None:
    dirty, error = _residual_gepa_dirty_status()
    if error:
        return error
    if not dirty:
        return None
    groups = _dirty_groups(dirty)
    group_detail = ", ".join(f"{name}={count}" for name, count in groups)
    sample = "; ".join(dirty[:6])
    suffix = "; ..." if len(dirty) > 6 else ""
    return (
        f"{len(dirty)} dirty/uncommitted non-launch GEPA workspace paths "
        f"({group_detail}; examples: {sample}{suffix})"
    )


def _ignored_launch_nuisance_paths() -> list[Path]:
    chart_a_configs = BLOG_ROOT / "charts" / "chart-a-head-to-head" / "configs"
    chart_d = BLOG_ROOT / "charts" / "chart-d-proposer-scaling"
    candidates = [
        BLOG_ROOT / "RERUN_HANDOFF.md",
        *chart_a_configs.glob("**/*"),
        *chart_d.glob("run_*_sweep.sh"),
        chart_d / "watch_chartd_board.py",
    ]
    return [path for path in candidates if path.exists()]


def _ignored_paths_detail(paths: list[Path], label: str) -> str | None:
    ignored, error = _ignored_status(paths)
    if error:
        return error
    if not ignored:
        return None
    sample = "; ".join(ignored[:6])
    suffix = "; ..." if len(ignored) > 6 else ""
    return f"{len(ignored)} ignored local {label} paths (examples: {sample}{suffix})"


def _residual_gepa_dirty_status() -> tuple[list[str], str | None]:
    dirty, error = _dirty_status(_gepa_workspace_paths())
    if error:
        return [], error
    excluded_paths = [
        *(_rel_path(path) for path in _launch_publication_paths()),
        *(_rel_path(path) for path in _tblite_quarantine_paths()),
    ]
    residual = [
        line for line in dirty
        if not any(path in line for path in excluded_paths)
    ]
    return residual, None


def _dirty_status(paths: list[Path]) -> tuple[list[str], str | None]:
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain", "--", *(_rel_path(path) for path in paths)],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return [], f"git status unavailable: {exc}"
    if proc.returncode != 0:
        return [], (proc.stderr or proc.stdout or "git status failed").strip()
    dirty = [line for line in proc.stdout.splitlines() if line.strip()]
    return dirty, None


def _ignored_status(paths: list[Path]) -> tuple[list[str], str | None]:
    if not paths:
        return [], None
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain", "--ignored", "--", *(_rel_path(path) for path in paths)],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return [], f"git status unavailable: {exc}"
    if proc.returncode != 0:
        return [], (proc.stderr or proc.stdout or "git status failed").strip()
    ignored = [line for line in proc.stdout.splitlines() if line.startswith("!!")]
    return ignored, None


def _print_dirty_paths(title: str, paths: list[Path]) -> None:
    dirty, error = _dirty_status(paths)
    print(f"{title}:")
    if error:
        print(f"  ERROR {error}")
        return
    if not dirty:
        print("  clean")
        return
    groups = _dirty_groups(dirty)
    print("  " + ", ".join(f"{name}={count}" for name, count in groups))
    for line in dirty:
        print(f"  {line}")


def _dirty_groups(dirty: list[str]) -> list[tuple[str, int]]:
    buckets = {
        "blog_packet": "cookbooks/blogs/oss-containers-and-gepa",
        "evals_evidence": "cookbooks/optimizers/gepa/evals/evidence",
        "tblite": "cookbooks/optimizers/gepa/tblite_container",
        "gepa_evals": "cookbooks/optimizers/gepa/evals",
        "gepa_containers": "cookbooks/optimizers/gepa",
    }
    counts = {name: 0 for name in buckets}
    other = 0
    for line in dirty:
        for name, marker in buckets.items():
            if marker in line:
                counts[name] += 1
                break
        else:
            other += 1
    out = [(name, count) for name, count in counts.items() if count]
    if other:
        out.append(("other", other))
    return out


def _print_publication_packet() -> None:
    checks = _publication_packet_checks()
    ready = all(check.ok for check in checks if check.required)
    print()
    print(f"publication packet: {'READY' if ready else 'PENDING'}")
    for check in checks:
        if check.required:
            mark = "ok" if check.ok else "FAIL"
        else:
            mark = "note" if check.ok else "WARN"
        print(f"  [{mark:>4}] {check.name:28} {check.detail}")
    if not ready:
        print("  do not publish: run evidence is not the same as a public launch packet.")


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "status"
    if cmd == "status":
        return cmd_status()
    if cmd == "show":
        if len(argv) < 2:
            print("usage: show <key>", file=sys.stderr)
            return 1
        return cmd_show(argv[1])
    if cmd == "plan":
        return cmd_plan()
    if cmd == "packet":
        return cmd_packet()
    if cmd == "run":
        return cmd_run(argv[1:])
    print(f"unknown command {cmd!r}; use status | show <key> | plan | packet | run", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
