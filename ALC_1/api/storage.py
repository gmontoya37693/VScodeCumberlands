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
        self.inputs = self.root / "inputs"
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

    def sync_down(self) -> None:
        """No-op for local disk; overridden by Blob-backed storage."""

    def sync_up(self) -> None:
        """No-op for local disk; overridden by Blob-backed storage."""

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


class BlobStorage(LocalStorage):
    """Blob-synced storage for a single-instance (replica=1) deployment.

    Reuses LocalStorage's local directory layout as a scratch mirror, pulling
    it from Blob at startup and pushing it back after every write or run. This
    is only safe because exactly one instance runs at a time; it is not a
    distributed lock and does not coordinate multiple replicas.
    """

    def __init__(self, root: Path, account_url: str, container: str) -> None:
        super().__init__(root)
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient

        credential = DefaultAzureCredential()
        service_client = BlobServiceClient(account_url=account_url, credential=credential)
        self.container = service_client.get_container_client(container)

    def _pull_prefix(self, prefix: str, local_dir: Path) -> None:
        local_dir.mkdir(parents=True, exist_ok=True)
        for blob in self.container.list_blobs(name_starts_with=f"{prefix}/"):
            relative = blob.name[len(prefix) + 1 :]
            if not relative:
                continue
            destination = local_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as handle:
                self.container.get_blob_client(blob.name).download_blob().readinto(handle)

    def _push_dir(self, local_dir: Path, prefix: str) -> None:
        if not local_dir.exists():
            return
        for path in sorted(local_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(local_dir).as_posix()
            with path.open("rb") as handle:
                self.container.get_blob_client(f"{prefix}/{relative}").upload_blob(handle, overwrite=True)

    def ensure_runtime_directories(self) -> None:
        super().ensure_runtime_directories()
        if not self.container.exists():
            self.container.create_container()
        self.sync_down()

    def sync_down(self) -> None:
        """Refresh the local mirror from Blob (inputs, state, manifests, archives)."""
        for prefix, local_dir in (
            ("inputs", self.inputs),
            ("state", self.state),
            ("manifests", self.manifests),
            ("archives", self.archives),
        ):
            self._pull_prefix(prefix, local_dir)

    def sync_up(self) -> None:
        """Push local mirror changes to Blob (inputs, state, manifests, archives, outputs)."""
        for prefix, local_dir in (
            ("inputs", self.inputs),
            ("state", self.state),
            ("manifests", self.manifests),
            ("archives", self.archives),
            ("outputs", self.outputs),
        ):
            self._push_dir(local_dir, prefix)

    def publish_outputs(self, run_dir: Path, output_date: date, run_id: str) -> list[dict[str, Any]]:
        published = super().publish_outputs(run_dir, output_date, run_id)
        self.sync_up()
        return published
