from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REQUIRED_FIELDS = (
    "schema_version",
    "id",
    "title",
    "summary",
    "version",
    "status",
    "classification",
    "entrypoints",
)


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "catalog" / "catalog.v1.json").is_file() and (parent / "cookbooks").is_dir():
            return parent
    raise RuntimeError("could not locate cookbook repository root")


@dataclass(frozen=True)
class CookbookManifest:
    path: Path
    data: dict[str, Any]

    @property
    def cookbook_id(self) -> str:
        return str(self.data["id"])

    @property
    def classification(self) -> dict[str, Any]:
        return dict(self.data.get("classification") or {})


def load_manifest(path: Path) -> CookbookManifest:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: cookbook.yaml must be a mapping")
    missing = [field for field in REQUIRED_FIELDS if field not in data]
    if missing:
        raise ValueError(f"{path}: missing fields {missing}")
    if data.get("schema_version") != "synth.cookbook.v1":
        raise ValueError(f"{path}: unsupported schema_version {data.get('schema_version')!r}")
    return CookbookManifest(path=path, data=data)


def iter_manifests(root: Path | None = None) -> list[CookbookManifest]:
    root = root or repo_root()
    paths = sorted((root / "cookbooks").glob("*/cookbook.yaml"))
    return [load_manifest(path) for path in paths]


def catalog_ids(root: Path | None = None) -> list[str]:
    root = root or repo_root()
    payload = json.loads((root / "catalog" / "catalog.v1.json").read_text())
    return [str(row["id"]) for row in payload.get("cookbooks") or []]
