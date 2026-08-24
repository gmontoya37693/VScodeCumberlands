from __future__ import annotations

import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .storage import LocalStorage


@dataclass(frozen=True)
class EngineResult:
    run_id: str
    command: str
    stdout: str
    stderr: str
    run_dir: Path
    output_date: date


def run_engine(
    storage: LocalStorage,
    command: str,
    operator: str,
    *,
    as_of: date | None = None,
    month: str | None = None,
    asset_id: str | None = None,
    output_date: date | None = None,
    billing_day: int = 22,
) -> EngineResult:
    """Run one engine command against an isolated copy of the source inputs."""
    run_id = uuid.uuid4().hex
    run_dir = storage.prepare_run(run_id)
    state_dir = storage.state
    run_dir.mkdir(parents=True, exist_ok=True)

    args = [
        sys.executable,
        str(storage.root / "alc_item_sheet_tracker.py"),
        command,
        "--operator",
        operator,
        "--assets",
        str(run_dir / "assets.csv"),
        "--rates",
        str(run_dir / "rates.csv"),
        "--posted-ledger",
        str(state_dir / "posted_invoices.csv"),
        "--closed-periods",
        str(state_dir / "closed_periods.csv"),
        "--baseline-config",
        str(state_dir / "baseline_config.json"),
        "--manifest-dir",
        str(storage.manifests),
        "--backup-dir",
        str(storage.archives),
        "--billing-day",
        str(billing_day),
    ]
    if command == "daily":
        if as_of is None:
            raise ValueError("as_of is required for daily runs")
        args.extend(["--as-of", as_of.isoformat()])
        output_date = output_date or as_of
    elif command == "snapshot":
        if as_of is None:
            raise ValueError("as_of is required for snapshot runs")
        args.extend(["--as-of", as_of.isoformat(), "--output", str(run_dir / "snapshot.csv")])
        output_date = output_date or as_of
    elif command in {"invoice", "bank-payable"}:
        if month is None:
            raise ValueError(f"month is required for {command} runs")
        args.extend(["--month", month])
        output_date = output_date or date.fromisoformat(f"{month}-01")
        if command == "invoice":
            args.extend(
                [
                    "--output",
                    str(run_dir / f"invoices_{month}.csv"),
                    "--one-pager-output",
                    str(run_dir / f"one-pager_{month}.xlsx"),
                ]
            )
        else:
            args.extend(["--bank-payable-file", str(run_dir / f"bank-payable_{month}.csv")])
    elif command == "schedule":
        args.extend(["--output", str(run_dir / "schedule.csv")])
    elif command == "one-pager":
        args.extend(["--output", str(run_dir / "one-pager.xlsx")])

    output_date = output_date or date.today()

    if asset_id and command in {"schedule", "one-pager"}:
        args.extend(["--asset-id", asset_id])

    completed = subprocess.run(
        args,
        cwd=storage.root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())

    return EngineResult(
        run_id=run_id,
        command=command,
        stdout=completed.stdout,
        stderr=completed.stderr,
        run_dir=run_dir,
        output_date=output_date,
    )
