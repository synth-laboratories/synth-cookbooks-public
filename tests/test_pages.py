from __future__ import annotations

import json
from pathlib import Path

from cookbook_tools.manifest import repo_root


def test_every_cookbook_has_evidence_and_report() -> None:
    root = repo_root()
    pages = json.loads((root / "catalog" / "generated" / "pages.json").read_text())
    slugs = {page["slug"] for page in pages["pages"]}
    assert slugs == {
        "gepa-banking77",
        "evals-healthbench",
        "mlx-sft-craftax",
        "remote-cispo-gsm8k",
    }
    for slug in slugs:
        cookbook = root / "cookbooks" / slug
        assert (cookbook / "REPORT.md").is_file()
        evidence = json.loads((cookbook / "evidence" / "evidence.json").read_text())
        assert evidence["cookbook_id"] == slug
        assert "not_claimed" in evidence
