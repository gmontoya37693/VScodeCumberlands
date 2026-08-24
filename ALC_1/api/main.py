from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .engine_runner import run_engine
from .commit_changes import commit_input_change
from .input_changes import propose_asset_change, propose_rate_change
from .storage import LocalStorage


ROOT = Path(os.environ.get("ALC_ROOT", Path(__file__).resolve().parents[1]))
storage = LocalStorage(ROOT)
storage.ensure_runtime_directories()
app = FastAPI(title="ALC V1 API", version="1.0.0")


class DailyRunRequest(BaseModel):
    operator: str = Field(min_length=1)
    as_of: date
    billing_day: int = Field(default=22, ge=1, le=31)


class MonthRunRequest(BaseModel):
    operator: str = Field(min_length=1)
    month: str = Field(pattern=r"^\d{4}-\d{2}$")
    billing_day: int = Field(default=22, ge=1, le=31)


class SnapshotRunRequest(DailyRunRequest):
    pass


class ScheduleRunRequest(BaseModel):
    operator: str = Field(min_length=1)
    asset_id: str | None = None


class OnePagerRunRequest(ScheduleRunRequest):
    pass


class RateProposalRequest(BaseModel):
    operator: str = Field(min_length=1)
    effective_date: date
    bank_rate_annual: float = Field(ge=0, le=1)


class AssetProposalRequest(BaseModel):
    operator: str = Field(min_length=1)
    action: str = Field(pattern=r"^(add|update)$")
    asset_id: str = Field(min_length=1)
    changes: dict[str, Any]


class InputCommitRequest(BaseModel):
    proposal_id: str = Field(min_length=1)
    approver: str = Field(min_length=1)
    approval_id: str = Field(min_length=1)


def _input_response(name: str) -> dict[str, Any]:
    path = storage.inputs / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Input not found: {name}")
    return {
        "source": name,
        "retrieved_at": date.today().isoformat(),
        "records": storage.read_csv(name),
    }


def _run_response(result: Any, **details: str) -> dict[str, Any]:
    files = storage.publish_outputs(result.run_dir, result.output_date, result.run_id)
    return {
        "status": "completed",
        "run_id": result.run_id,
        "command": result.command,
        **details,
        "stdout": result.stdout,
        "files": files,
    }


def _execute(command: str, operator: str, **kwargs: Any) -> Any:
    try:
        return run_engine(storage, command, operator, **kwargs)
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "alc-v1-api"}


@app.get("/api/v1/assets")
def get_assets() -> dict[str, Any]:
    return _input_response("assets.csv")


@app.get("/api/v1/rates")
def get_rates() -> dict[str, Any]:
    return _input_response("rates.csv")


@app.post("/api/v1/rates/propose")
def create_rate_proposal(request: RateProposalRequest) -> dict[str, Any]:
    try:
        return propose_rate_change(
            storage,
            request.operator,
            request.effective_date,
            request.bank_rate_annual,
        )
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/assets/propose")
def create_asset_proposal(request: AssetProposalRequest) -> dict[str, Any]:
    try:
        return propose_asset_change(
            storage,
            request.operator,
            request.action,
            request.asset_id,
            request.changes,
        )
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/input-changes/commit")
def commit_change(request: InputCommitRequest) -> dict[str, Any]:
    try:
        return commit_input_change(
            storage,
            request.proposal_id,
            request.approver,
            request.approval_id,
        )
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _state_response(name: str) -> dict[str, Any]:
    return {
        "source": f"state/{name}",
        "retrieved_at": date.today().isoformat(),
        "records": storage.read_state_csv(name),
    }


@app.get("/api/v1/state/posted-invoices")
def get_posted_invoices() -> dict[str, Any]:
    return _state_response("posted_invoices.csv")


@app.get("/api/v1/state/bank-payable")
def get_bank_payable() -> dict[str, Any]:
    return _state_response("bank_payable.csv")


@app.get("/api/v1/state/closed-periods")
def get_closed_periods() -> dict[str, Any]:
    return _state_response("closed_periods.csv")


@app.get("/api/v1/manifests/{run_id}")
def get_manifest(run_id: str) -> dict[str, Any]:
    if not run_id or Path(run_id).name != run_id or not run_id.isalnum():
        raise HTTPException(status_code=400, detail="Invalid run ID")
    path = storage.manifest_path(run_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run manifest not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/api/v1/runs/daily")
@app.post("/api/v1/runs/daily-preview")
def run_daily(request: DailyRunRequest) -> dict[str, Any]:
    result = _execute(
        "daily",
        request.operator,
        as_of=request.as_of,
        billing_day=request.billing_day,
    )
    return _run_response(result, as_of=request.as_of.isoformat())


@app.post("/api/v1/runs/snapshot")
def run_snapshot(request: SnapshotRunRequest) -> dict[str, Any]:
    result = _execute("snapshot", request.operator, as_of=request.as_of)
    return _run_response(result, as_of=request.as_of.isoformat())


@app.post("/api/v1/runs/schedule")
def run_schedule(request: ScheduleRunRequest) -> dict[str, Any]:
    result = _execute("schedule", request.operator, asset_id=request.asset_id)
    return _run_response(result, asset_id=request.asset_id or "")


@app.post("/api/v1/runs/invoice")
def run_invoice(request: MonthRunRequest) -> dict[str, Any]:
    result = _execute(
        "invoice",
        request.operator,
        month=request.month,
        billing_day=request.billing_day,
    )
    return _run_response(result, month=request.month)


@app.post("/api/v1/runs/bank-payable")
def run_bank_payable(request: MonthRunRequest) -> dict[str, Any]:
    result = _execute(
        "bank-payable",
        request.operator,
        month=request.month,
        billing_day=request.billing_day,
    )
    return _run_response(result, month=request.month)


@app.post("/api/v1/runs/month-end")
def run_month_end(request: MonthRunRequest) -> dict[str, Any]:
    bank_result = _execute(
        "bank-payable",
        request.operator,
        month=request.month,
        billing_day=request.billing_day,
    )
    bank_response = _run_response(bank_result, month=request.month)
    close_result = _execute(
        "close-period",
        request.operator,
        month=request.month,
        billing_day=request.billing_day,
    )
    close_response = _run_response(close_result, month=request.month)
    return {
        "status": "completed",
        "run_id": close_result.run_id,
        "command": "month-end",
        "month": request.month,
        "bank_payable_run_id": bank_result.run_id,
        "close_period_run_id": close_result.run_id,
        "bank_payable": bank_response,
        "close_period": close_response,
    }


@app.post("/api/v1/runs/one-pager")
def run_one_pager(request: OnePagerRunRequest) -> dict[str, Any]:
    result = _execute("one-pager", request.operator, asset_id=request.asset_id)
    return _run_response(result, asset_id=request.asset_id or "")
