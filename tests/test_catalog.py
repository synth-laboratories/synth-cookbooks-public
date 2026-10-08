from __future__ import annotations

from cookbook_tools.manifest import catalog_ids, iter_manifests, repo_root
from cookbook_tools.preflight import run_preflight


def test_catalog_matches_manifests() -> None:
    root = repo_root()
    ids = {manifest.cookbook_id for manifest in iter_manifests(root)}
    assert ids == set(catalog_ids(root))
    assert ids == {
        "gepa-banking77",
        "evals-healthbench",
        "mlx-sft-craftax",
        "remote-cispo-gsm8k",
    }


def test_preflight_schema() -> None:
    manifest = next(
        item for item in iter_manifests() if item.cookbook_id == "gepa-banking77"
    )
    result = run_preflight(manifest)
    assert result["schema_version"] == "synth.cookbook.preflight.v1"
    assert result["cookbook_id"] == "gepa-banking77"
    assert "checks" in result
