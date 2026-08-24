from __future__ import annotations

import csv
import hashlib
import shutil
from datetime import date
from pathlib import Path
from typing import Any


class LocalStorage:
    """Filesystem storage adapter used locally before Azure Blob is added."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.inputs = self.root
        self.state = self.root / "state"
        self.working = self.root / "working"
        self.outputs = self.root / "outputs"
        self.manifests = self.root / "manifests"
        self.proposals = self.manifests / "proposals"
        self.archives = self.root / "archives"

    def read_csv(self, name: str) -> list[dict[str, str]]:
        path = self.inputs / name
        with path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def read_state_csv(self, name: str) -> list[dict[str, str]]:
        path = self.state / name
        if not path.exists():
            return []
        with path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def prepare_run(self, run_id: str) -> Path:
        run_dir = self.working / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        for name in ("assets.csv", "rates.csv"):
            shutil.copy2(self.inputs / name, run_dir / name)
        return run_dir

    def ensure_runtime_directories(self) -> None:
        for path in (self.state, self.working, self.outputs, self.manifests, self.proposals, self.archives):
            path.mkdir(parents=True, exist_ok=True)

    def hash_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def manifest_path(self, run_id: str) -> Path:
        return self.manifests / f"{run_id}.json"

    def serialize_path(self, path: Path) -> str:
        return str(path.relative_to(self.root))

    def output_metadata(self, run_dir: Path) -> list[dict[str, Any]]:
        return [
            {"name": path.name, "path": self.serialize_path(path)}
            for path in sorted(run_dir.rglob("*"))
            if path.is_file() and path.name not in {"assets.csv", "rates.csv"}
        ]

    def publish_outputs(self, run_dir: Path, output_date: date, run_id: str) -> list[dict[str, Any]]:
        """Copy run artifacts into a dated, non-overwriting output folder."""
        output_dir = self.outputs / f"{output_date:%Y}" / f"{output_date:%m}" / f"{output_date:%d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        published: list[dict[str, Any]] = []
        for source in sorted(run_dir.rglob("*")):
            if not source.is_file() or source.name in {"assets.csv", "rates.csv"}:
                continue
            destination = output_dir / f"{source.stem}_{run_id}{source.suffix}"
            shutil.copy2(source, destination)
            published.append({"name": destination.name, "path": self.serialize_path(destination)})
        return published
