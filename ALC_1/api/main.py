from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import uuid
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .engine_runner import run_engine
from .commit_changes import commit_input_change
from .input_changes import propose_asset_change, propose_rate_change
from .storage import BlobStorage, LocalStorage


logger = logging.getLogger("alc.api")


ROOT = Path(os.environ.get("ALC_ROOT", Path(__file__).resolve().parents[1]))
STORAGE_MODE = os.environ.get("ALC_STORAGE_MODE", "local").lower()
if STORAGE_MODE == "blob":
    storage = BlobStorage(
        ROOT,
        os.environ["AZURE_STORAGE_ACCOUNT_URL"],
        os.environ.get("ALC_BLOB_CONTAINER", "alc"),
    )
else:
    storage = LocalStorage(ROOT)
storage.ensure_runtime_directories()
app = FastAPI(title="ALC V1 API", version="1.0.0")

# Single-instance deployment (replica=1): guards shared state files from
# overlapping requests within this one process. Not a distributed lock.
_write_lock = threading.Lock()
_api_key = os.environ.get("ALC_API_KEY")


@app.middleware("http")
async def add_request_id(request: Request, call_next: Any) -> Any:
    """Assign one request_id per HTTP call, before auth/routing/endpoint code runs."""
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    logger.info("request_id=%s method=%s path=%s", request_id, request.method, request.url.path)
    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    logger.info("request_id=%s status=%s", request_id, response.status_code)
    return response


@app.exception_handler(HTTPException)
async def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
    """Echo the same request_id back in error bodies, not just success bodies."""
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(status_code=exc.status_code, content={"request_id": request_id, "detail": exc.detail})


def get_request_id(request: Request) -> str:
    return request.state.request_id


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    if not _api_key or not x_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    if not secrets.compare_digest(x_api_key, _api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


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


# Manifest fields that are internal bookkeeping, not part of a run's summary.
_MANIFEST_SUMMARY_EXCLUDE = {
    "run_id",
    "timestamp",
    "command",
    "script_version",
    "operator",
    "backups",
    "input_hashes_before",
    "input_hashes_after",
}


def _input_response(name: str, request_id: str) -> dict[str, Any]:
    logger.info("request_id=%s endpoint=input name=%s", request_id, name)
    path = storage.inputs / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Input not found: {name}")
    return {
        "request_id": request_id,
        "source": name,
        "retrieved_at": date.today().isoformat(),
        "records": storage.read_csv(name),
    }


def _run_response(result: Any, request_id: str, **details: str) -> dict[str, Any]:
    files = storage.publish_outputs(result.run_dir, result.output_date, result.run_id)
    manifest_path = storage.manifest_path(result.run_id)
    summary: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = {k: v for k, v in manifest.items() if k not in _MANIFEST_SUMMARY_EXCLUDE}
    logger.info("request_id=%s endpoint=run command=%s run_id=%s", request_id, result.command, result.run_id)
    return {
        "request_id": request_id,
        "status": "completed",
        "run_id": result.run_id,
        "command": result.command,
        **details,
        "summary": summary,
        "files": files,
        "warnings": [],
        "errors": [],
        "stdout": result.stdout,
    }


def _execute(command: str, operator: str, **kwargs: Any) -> Any:
    try:
        return run_engine(storage, command, operator, **kwargs)
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "alc-v1-api"}


@app.get("/api/v1/assets", dependencies=[Depends(require_api_key)])
def get_assets(request_id: str = Depends(get_request_id)) -> dict[str, Any]:
    return _input_response("assets.csv", request_id)


@app.get("/api/v1/rates", dependencies=[Depends(require_api_key)])
def get_rates(request_id: str = Depends(get_request_id)) -> dict[str, Any]:
    return _input_response("rates.csv", request_id)


@app.post("/api/v1/rates/propose", dependencies=[Depends(require_api_key)])
def create_rate_proposal(
    request: RateProposalRequest, request_id: str = Depends(get_request_id)
) -> dict[str, Any]:
    logger.info("request_id=%s endpoint=rates/propose operator=%s", request_id, request.operator)
    try:
        result = propose_rate_change(
            storage,
            request.operator,
            request.effective_date,
            request.bank_rate_annual,
        )
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"request_id": request_id, **result}


@app.post("/api/v1/assets/propose", dependencies=[Depends(require_api_key)])
def create_asset_proposal(
    request: AssetProposalRequest, request_id: str = Depends(get_request_id)
) -> dict[str, Any]:
    logger.info("request_id=%s endpoint=assets/propose operator=%s", request_id, request.operator)
    try:
        result = propose_asset_change(
            storage,
            request.operator,
            request.action,
            request.asset_id,
            request.changes,
        )
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"request_id": request_id, **result}


@app.post("/api/v1/input-changes/commit", dependencies=[Depends(require_api_key)])
def commit_change(
    request: InputCommitRequest, request_id: str = Depends(get_request_id)
) -> dict[str, Any]:
    logger.info("request_id=%s endpoint=input-changes/commit proposal_id=%s", request_id, request.proposal_id)
    with _write_lock:
        try:
            result = commit_input_change(
                storage,
                request.proposal_id,
                request.approver,
                request.approval_id,
            )
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"request_id": request_id, **result}


def _state_response(name: str, request_id: str) -> dict[str, Any]:
    logger.info("request_id=%s endpoint=state name=%s", request_id, name)
    return {
        "request_id": request_id,
        "source": f"state/{name}",
        "retrieved_at": date.today().isoformat(),
        "records": storage.read_state_csv(name),
    }


@app.get("/api/v1/state/posted-invoices", dependencies=[Depends(require_api_key)])
def get_posted_invoices(request_id: str = Depends(get_request_id)) -> dict[str, Any]:
    return _state_response("posted_invoices.csv", request_id)


@app.get("/api/v1/state/bank-payable", dependencies=[Depends(require_api_key)])
def get_bank_payable(request_id: str = Depends(get_request_id)) -> dict[str, Any]:
    return _state_response("bank_payable.csv", request_id)


@app.get("/api/v1/state/closed-periods", dependencies=[Depends(require_api_key)])
def get_closed_periods(request_id: str = Depends(get_request_id)) -> dict[str, Any]:
    return _state_response("closed_periods.csv", request_id)


@app.get("/api/v1/manifests/{run_id}", dependencies=[Depends(require_api_key)])
def get_manifest(run_id: str, request_id: str = Depends(get_request_id)) -> dict[str, Any]:
    logger.info("request_id=%s endpoint=manifests run_id=%s", request_id, run_id)
    if not run_id or Path(run_id).name != run_id or not run_id.isalnum():
        raise HTTPException(status_code=400, detail="Invalid run ID")
    path = storage.manifest_path(run_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run manifest not found")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {"request_id": request_id, **manifest}


@app.get("/api/v1/outputs", dependencies=[Depends(require_api_key)])
def list_outputs(request_id: str = Depends(get_request_id)) -> dict[str, Any]:
    logger.info("request_id=%s endpoint=outputs/list", request_id)
    if not isinstance(storage, BlobStorage):
        raise HTTPException(status_code=400, detail="Output listing requires blob storage mode")
    return {"request_id": request_id, "files": storage.list_outputs()}


@app.get("/api/v1/outputs/download", dependencies=[Depends(require_api_key)])
def get_output_download_link(
    blob_name: str, request_id: str = Depends(get_request_id)
) -> dict[str, Any]:
    logger.info("request_id=%s endpoint=outputs/download blob_name=%s", request_id, blob_name)
    if not isinstance(storage, BlobStorage):
        raise HTTPException(status_code=400, detail="Download links require blob storage mode")
    try:
        url = storage.generate_download_url(blob_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"request_id": request_id, "blob_name": blob_name, "url": url, "expires_in_minutes": 15}


@app.get("/api/v1/inputs/download", dependencies=[Depends(require_api_key)])
def get_input_download_link(
    name: str, request_id: str = Depends(get_request_id)
) -> dict[str, Any]:
    logger.info("request_id=%s endpoint=inputs/download name=%s", request_id, name)
    if not isinstance(storage, BlobStorage):
        raise HTTPException(status_code=400, detail="Download links require blob storage mode")
    if name not in {"assets.csv", "rates.csv"}:
        raise HTTPException(status_code=400, detail="name must be assets.csv or rates.csv")
    try:
        url = storage.generate_download_url(f"inputs/{name}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"request_id": request_id, "blob_name": f"inputs/{name}", "url": url, "expires_in_minutes": 15}


@app.post("/api/v1/runs/daily-preview", dependencies=[Depends(require_api_key)])
@app.post("/api/v1/runs/daily", dependencies=[Depends(require_api_key)])
def run_daily(request: DailyRunRequest, request_id: str = Depends(get_request_id)) -> dict[str, Any]:
    result = _execute(
        "daily",
        request.operator,
        as_of=request.as_of,
        billing_day=request.billing_day,
    )
    return _run_response(result, request_id, as_of=request.as_of.isoformat())


@app.post("/api/v1/runs/snapshot", dependencies=[Depends(require_api_key)])
def run_snapshot(request: SnapshotRunRequest, request_id: str = Depends(get_request_id)) -> dict[str, Any]:
    result = _execute("snapshot", request.operator, as_of=request.as_of)
    return _run_response(result, request_id, as_of=request.as_of.isoformat())


@app.post("/api/v1/runs/schedule", dependencies=[Depends(require_api_key)])
def run_schedule(request: ScheduleRunRequest, request_id: str = Depends(get_request_id)) -> dict[str, Any]:
    result = _execute("schedule", request.operator, asset_id=request.asset_id)
    return _run_response(result, request_id, asset_id=request.asset_id or "")


@app.post("/api/v1/runs/invoice", dependencies=[Depends(require_api_key)])
def run_invoice(request: MonthRunRequest, request_id: str = Depends(get_request_id)) -> dict[str, Any]:
    with _write_lock:
        result = _execute(
            "invoice",
            request.operator,
            month=request.month,
            billing_day=request.billing_day,
        )
        return _run_response(result, request_id, month=request.month)


@app.post("/api/v1/runs/bank-payable", dependencies=[Depends(require_api_key)])
def run_bank_payable(request: MonthRunRequest, request_id: str = Depends(get_request_id)) -> dict[str, Any]:
    with _write_lock:
        result = _execute(
            "bank-payable",
            request.operator,
            month=request.month,
            billing_day=request.billing_day,
        )
        return _run_response(result, request_id, month=request.month)


@app.post("/api/v1/runs/month-end", dependencies=[Depends(require_api_key)])
def run_month_end(request: MonthRunRequest, request_id: str = Depends(get_request_id)) -> dict[str, Any]:
    with _write_lock:
        bank_result = _execute(
            "bank-payable",
            request.operator,
            month=request.month,
            billing_day=request.billing_day,
        )
        bank_response = _run_response(bank_result, request_id, month=request.month)
        close_result = _execute(
            "close-period",
            request.operator,
            month=request.month,
            billing_day=request.billing_day,
        )
        close_response = _run_response(close_result, request_id, month=request.month)
        return {
            "request_id": request_id,
            "status": "completed",
            "run_id": close_result.run_id,
            "command": "month-end",
            "month": request.month,
            "bank_payable_run_id": bank_result.run_id,
            "close_period_run_id": close_result.run_id,
            "bank_payable": bank_response,
            "close_period": close_response,
        }


@app.post("/api/v1/runs/one-pager", dependencies=[Depends(require_api_key)])
def run_one_pager(request: OnePagerRunRequest, request_id: str = Depends(get_request_id)) -> dict[str, Any]:
    result = _execute("one-pager", request.operator, asset_id=request.asset_id)
    return _run_response(result, request_id, asset_id=request.asset_id or "")
