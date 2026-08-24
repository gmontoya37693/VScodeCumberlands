from __future__ import annotations

import csv
import json
import shutil
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .input_changes import ASSET_ALLOWED_FIELDS
from .storage import LocalStorage


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_proposal(storage: LocalStorage, proposal_id: str) -> tuple[dict[str, Any], Path]:
    if not proposal_id or Path(proposal_id).name != proposal_id or not proposal_id.isalnum():
        raise ValueError("invalid proposal ID")
    path = storage.proposals / f"{proposal_id}.json"
    if not path.exists():
        raise ValueError("proposal not found")
    return json.loads(path.read_text(encoding="utf-8")), path


def _write_csv_atomic(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _archive_input(storage: LocalStorage, path: Path, proposal_id: str) -> str:
    destination = storage.archives / f"{path.stem}_{datetime.now():%Y%m%dT%H%M%S}_{proposal_id}{path.suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    return storage.serialize_path(destination)


def _same_date(value: str, target: str) -> bool:
    target_date = date.fromisoformat(target)
    for fmt in ("%Y-%m-%d", "%m/%d/%y", "%m/%d/%Y"):
        try:
            return datetime.strptime(value.strip(), fmt).date() == target_date
        except ValueError:
            continue
    return False


def _commit_rate(storage: LocalStorage, proposal: dict[str, Any]) -> str:
    path = storage.inputs / "rates.csv"
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if fieldnames != ["effective_date", "bank_rate_annual"]:
        raise ValueError("rates.csv has an unexpected header")

    effective_date = str(proposal["effective_date"])
    new_value = str(proposal["new_value"])
    replaced = False
    for row in rows:
        if _same_date(row.get("effective_date", ""), effective_date):
            row["bank_rate_annual"] = new_value
            replaced = True
    if not replaced:
        rows.append({"effective_date": effective_date, "bank_rate_annual": new_value})
    rows.sort(key=lambda row: row.get("effective_date", ""))
    archive_path = _archive_input(storage, path, str(proposal["proposal_id"]))
    _write_csv_atomic(path, fieldnames, rows)
    return archive_path


def _commit_asset(storage: LocalStorage, proposal: dict[str, Any]) -> str:
    path = storage.inputs / "assets.csv"
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames or set(fieldnames) != ASSET_ALLOWED_FIELDS:
        raise ValueError("assets.csv has an unexpected header")

    asset_id = str(proposal["asset_id"])
    changes = dict(proposal["new_value"])
    changed = False
    if proposal["action"] == "add":
        row = {field: str(changes.get(field, "")) for field in fieldnames}
        rows.append(row)
        changed = True
    else:
        for row in rows:
            if row.get("asset_id") == asset_id:
                row.update({key: str(value) for key, value in changes.items()})
                changed = True
                break
    if not changed:
        raise ValueError("asset no longer matches the proposal")

    archive_path = _archive_input(storage, path, str(proposal["proposal_id"]))
    _write_csv_atomic(path, fieldnames, rows)
    return archive_path


def commit_input_change(
    storage: LocalStorage,
    proposal_id: str,
    approver: str,
    approval_id: str,
) -> dict[str, Any]:
    if not approver.strip() or not approval_id.strip():
        raise ValueError("approver and approval_id are required")

    proposal, proposal_path = _load_proposal(storage, proposal_id)
    if proposal.get("status") != "pending_approval":
        raise ValueError(f"proposal is not pending approval: {proposal.get('status')}")

    input_path = storage.inputs / str(proposal["input_file"])
    if not input_path.exists():
        raise ValueError("protected input file not found")
    if storage.hash_file(input_path) != proposal.get("input_hash"):
        raise ValueError("INPUT_CHANGED: create a new proposal")

    if proposal["type"] == "rate_change":
        archive_path = _commit_rate(storage, proposal)
    elif proposal["type"] == "asset_change":
        archive_path = _commit_asset(storage, proposal)
    else:
        raise ValueError("unsupported proposal type")

    proposal.update(
        {
            "status": "committed",
            "approved_by": approver,
            "approval_id": approval_id,
            "approved_at": _utc_now(),
            "archive_file": archive_path,
        }
    )
    proposal_path.write_text(json.dumps(proposal, indent=2, sort_keys=True), encoding="utf-8")
    return proposal
