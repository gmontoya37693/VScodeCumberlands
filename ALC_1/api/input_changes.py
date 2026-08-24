from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .storage import LocalStorage

ASSET_REQUIRED_FIELDS = {
    "asset_id",
    "property_id",
    "allocation",
    "asset_description",
    "asset_value",
    "start_date",
    "lifespan",
    "bank_rate_annual",
    "nim_annual",
    "gl_account",
    "status",
}
ASSET_ALLOWED_FIELDS = ASSET_REQUIRED_FIELDS | {
    "tax_amount",
    "admin_expense",
    "risk_cost_recovery",
    "salvage_value",
    "salvage_periods",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _proposal_path(storage: LocalStorage, proposal_id: str) -> Path:
    return storage.proposals / f"{proposal_id}.json"


def _save_proposal(storage: LocalStorage, proposal: dict[str, Any]) -> dict[str, Any]:
    path = _proposal_path(storage, str(proposal["proposal_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(proposal, indent=2, sort_keys=True), encoding="utf-8")
    return proposal


def _csv_date_matches(value: str | None, target: date) -> bool:
    if not value:
        return False
    for fmt in ("%Y-%m-%d", "%m/%d/%y", "%m/%d/%Y"):
        try:
            if datetime.strptime(value.strip(), fmt).date() == target:
                return True
        except ValueError:
            continue
    return False


def propose_rate_change(
    storage: LocalStorage,
    operator: str,
    effective_date: date,
    bank_rate_annual: float,
) -> dict[str, Any]:
    if not 0 <= bank_rate_annual <= 1:
        raise ValueError("bank_rate_annual must be between 0 and 1")

    records = storage.read_csv("rates.csv")
    old_value = next(
        (
            row.get("bank_rate_annual")
            for row in records
            if _csv_date_matches(row.get("effective_date"), effective_date)
        ),
        None,
    )
    proposal = {
        "proposal_id": uuid.uuid4().hex,
        "type": "rate_change",
        "status": "pending_approval",
        "requested_by": operator,
        "requested_at": _utc_now(),
        "input_file": "rates.csv",
        "input_hash": storage.hash_file(storage.inputs / "rates.csv"),
        "effective_date": effective_date.isoformat(),
        "old_value": old_value,
        "new_value": bank_rate_annual,
    }
    return _save_proposal(storage, proposal)


def propose_asset_change(
    storage: LocalStorage,
    operator: str,
    action: str,
    asset_id: str,
    changes: dict[str, Any],
) -> dict[str, Any]:
    if action not in {"add", "update"}:
        raise ValueError("action must be 'add' or 'update'")
    if not asset_id.strip():
        raise ValueError("asset_id is required")
    unknown = set(changes) - ASSET_ALLOWED_FIELDS
    if unknown:
        raise ValueError(f"unknown asset fields: {sorted(unknown)}")
    if action == "add":
        missing = ASSET_REQUIRED_FIELDS - set(changes)
        if missing:
            raise ValueError(f"new asset is missing fields: {sorted(missing)}")
    elif "asset_id" in changes and changes["asset_id"] != asset_id:
        raise ValueError("changes.asset_id must match asset_id")

    records = storage.read_csv("assets.csv")
    existing = next((row for row in records if row.get("asset_id") == asset_id), None)
    if action == "add" and existing:
        raise ValueError(f"asset already exists: {asset_id}")
    if action == "update" and not existing:
        raise ValueError(f"asset not found: {asset_id}")

    proposal = {
        "proposal_id": uuid.uuid4().hex,
        "type": "asset_change",
        "action": action,
        "status": "pending_approval",
        "requested_by": operator,
        "requested_at": _utc_now(),
        "input_file": "assets.csv",
        "input_hash": storage.hash_file(storage.inputs / "assets.csv"),
        "asset_id": asset_id,
        "old_value": existing,
        "new_value": changes,
    }
    return _save_proposal(storage, proposal)
