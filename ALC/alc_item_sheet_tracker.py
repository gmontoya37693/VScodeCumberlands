#!/usr/bin/env python3
"""
ALC Item Sheet Tracker

Objective
---------
Track leased assets for Acento Leasing Company, recompute installment values
when bank rates change, and produce daily and monthly operational reports.

Builder
-------
German Montoya - Data Scientist & AI Specialist

For
---
Acento Leasing Company (ALC)

What the script does
--------------------
- Tracks each asset by property/allocation
- Recalculates the monthly payment using bank rate + NIM
- Produces a point-in-time portfolio snapshot
- Produces a monthly invoice list
- Produces a monthly bank-payable summary
- Supports reconciliation between asset balances and bank debt

Inputs
------
1) assets.csv
    Required columns:
    asset_id,property_id,allocation,asset_description,asset_value,start_date,
    term_months,bank_rate_annual,nim_annual,gl_account,status

    Optional columns:
    salvage_value

2) rates.csv
    Required columns:
    effective_date,bank_rate_annual

3) As-of date or target month passed through the command line

Outputs
-------
- Daily portfolio report
- Asset-level snapshot with balance and amortization progress
- Monthly invoice lines
- Monthly bank-payable summary
- CSV exports when requested

Suggested operation
-------------------
1) Run the script daily.
2) If a new asset, rate change, invoice run, or bank payment is due, follow
    the command prompt and complete the action.
3) If nothing changed, generate the daily report and exit.
4) Run month-end invoicing once per month for all active assets.

Date format: YYYY-MM-DD
Rate format: decimal annual rate (5% = 0.05)
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Tuple


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def add_months(d: date, months: int) -> date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    return date(y, m, 1)


def months_between(start_month: date, end_month: date) -> int:
    return (end_month.year - start_month.year) * 12 + (end_month.month - start_month.month)


def pmt(principal: float, monthly_rate: float, n_periods: int) -> float:
    if n_periods <= 0:
        return 0.0
    if abs(monthly_rate) < 1e-12:
        return principal / n_periods
    return principal * (monthly_rate / (1.0 - math.pow(1.0 + monthly_rate, -n_periods)))


@dataclass
class Asset:
    asset_id: str
    property_id: str
    allocation: str
    asset_description: str
    asset_value: float
    start_date: date
    term_months: int
    bank_rate_annual: float
    nim_annual: float
    gl_account: str
    status: str
    salvage_value: float = 0.0

    @property
    def start_month(self) -> date:
        return month_start(self.start_date)

    @property
    def final_month(self) -> date:
        return add_months(self.start_month, self.term_months)


def load_assets(path: Path) -> List[Asset]:
    out: List[Asset] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "asset_id",
            "property_id",
            "allocation",
            "asset_description",
            "asset_value",
            "start_date",
            "term_months",
            "bank_rate_annual",
            "nim_annual",
            "gl_account",
            "status",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"assets.csv missing columns: {sorted(missing)}")

        for row in reader:
            out.append(
                Asset(
                    asset_id=row["asset_id"].strip(),
                    property_id=row["property_id"].strip(),
                    allocation=row["allocation"].strip(),
                    asset_description=row["asset_description"].strip(),
                    asset_value=float(row["asset_value"]),
                    start_date=parse_date(row["start_date"]),
                    term_months=int(row["term_months"]),
                    bank_rate_annual=float(row["bank_rate_annual"]),
                    nim_annual=float(row["nim_annual"]),
                    gl_account=row["gl_account"].strip(),
                    status=row["status"].strip().lower(),
                    salvage_value=float(row.get("salvage_value", "0") or 0),
                )
            )
    return out


def load_rates(path: Path) -> List[Tuple[date, float]]:
    rates: List[Tuple[date, float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"effective_date", "bank_rate_annual"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"rates.csv missing columns: {sorted(missing)}")

        for row in reader:
            rates.append((month_start(parse_date(row["effective_date"])), float(row["bank_rate_annual"])))

    rates.sort(key=lambda x: x[0])
    return rates


def bank_rate_for_month(month: date, default_rate: float, rate_table: List[Tuple[date, float]]) -> float:
    chosen = default_rate
    for eff, rate in rate_table:
        if eff <= month:
            chosen = rate
        else:
            break
    return chosen


def amortize_until(asset: Asset, as_of: date, rate_table: List[Tuple[date, float]]) -> Dict[str, float]:
    as_of_month = month_start(as_of)
    if as_of_month < asset.start_month:
        return {
            "installments_paid": 0,
            "interest_paid": 0.0,
            "amortization_paid": 0.0,
            "current_payment": 0.0,
            "balance": asset.asset_value,
            "months_remaining": asset.term_months,
        }

    elapsed = months_between(asset.start_month, as_of_month) + 1
    periods = max(0, min(asset.term_months, elapsed))

    balance = asset.asset_value
    interest_paid = 0.0
    amort_paid = 0.0
    current_payment = 0.0

    for i in range(periods):
        current_month = add_months(asset.start_month, i)
        remaining = asset.term_months - i
        annual_rate = bank_rate_for_month(current_month, asset.bank_rate_annual, rate_table) + asset.nim_annual
        monthly_rate = annual_rate / 12.0

        payment = pmt(balance, monthly_rate, remaining)
        interest = balance * monthly_rate
        principal = payment - interest

        # Keep ending balance from becoming tiny negative due to float rounding.
        if principal > balance:
            principal = balance
            payment = interest + principal

        balance -= principal
        interest_paid += interest
        amort_paid += principal
        current_payment = payment

        if balance <= 1e-8:
            balance = 0.0
            break

    months_remaining = max(0, asset.term_months - periods)
    return {
        "installments_paid": periods,
        "interest_paid": interest_paid,
        "amortization_paid": amort_paid,
        "current_payment": current_payment,
        "balance": balance,
        "months_remaining": months_remaining,
    }


def build_snapshot(assets: List[Asset], rate_table: List[Tuple[date, float]], as_of: date) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for a in assets:
        m = amortize_until(a, as_of, rate_table)
        final_date = add_months(a.start_month, a.term_months)
        days_to_maturity = (final_date - as_of).days
        active = a.status == "active" and m["balance"] > 0 and as_of >= a.start_date

        rows.append(
            {
                "asset_id": a.asset_id,
                "property_id": a.property_id,
                "allocation": a.allocation,
                "status": "active" if active else "inactive",
                "asset_value": round(a.asset_value, 2),
                "start_date": a.start_date.isoformat(),
                "final_date": final_date.isoformat(),
                "term_months": a.term_months,
                "bank_rate_annual_default": round(a.bank_rate_annual, 6),
                "nim_annual": round(a.nim_annual, 6),
                "gl_account": a.gl_account,
                "installments_paid": int(m["installments_paid"]),
                "interest_paid": round(m["interest_paid"], 2),
                "amortization_paid": round(m["amortization_paid"], 2),
                "current_payment": round(m["current_payment"], 2),
                "balance": round(m["balance"], 2),
                "months_remaining": int(m["months_remaining"]),
                "days_to_maturity": days_to_maturity,
            }
        )
    return rows


def rows_total(rows: List[Dict[str, object]]) -> Dict[str, float]:
    total_assets = len(rows)
    active_assets = sum(1 for r in rows if r["status"] == "active")
    principal = sum(float(r["asset_value"]) for r in rows)
    interest_paid = sum(float(r["interest_paid"]) for r in rows)
    amort_paid = sum(float(r["amortization_paid"]) for r in rows)
    balance = sum(float(r["balance"]) for r in rows)
    current_payment = sum(float(r["current_payment"]) for r in rows if r["status"] == "active")
    return {
        "total_assets": total_assets,
        "active_assets": active_assets,
        "asset_value_total": round(principal, 2),
        "interest_paid_total": round(interest_paid, 2),
        "amortization_paid_total": round(amort_paid, 2),
        "balance_total": round(balance, 2),
        "current_payment_active_total": round(current_payment, 2),
    }


def invoice_for_month(rows: List[Dict[str, object]], target_month: date) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for r in rows:
        start_date = parse_date(str(r["start_date"]))
        final_date = parse_date(str(r["final_date"]))
        if month_start(start_date) <= target_month < month_start(final_date) and r["status"] == "active":
            out.append(
                {
                    "invoice_month": target_month.isoformat(),
                    "asset_id": r["asset_id"],
                    "property_id": r["property_id"],
                    "allocation": r["allocation"],
                    "gl_account": r["gl_account"],
                    "invoice_amount": r["current_payment"],
                }
            )
    return out


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_totals(totals: Dict[str, float], title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for k, v in totals.items():
        print(f"{k}: {v}")


def run_snapshot(args: argparse.Namespace) -> None:
    assets = load_assets(Path(args.assets))
    rates = load_rates(Path(args.rates))
    as_of = parse_date(args.as_of)
    rows = build_snapshot(assets, rates, as_of)
    totals = rows_total(rows)

    if args.output:
        write_csv(Path(args.output), rows)
        print(f"snapshot written to {args.output}")

    print_totals(totals, f"ALC Snapshot as of {as_of.isoformat()}")


def run_invoice(args: argparse.Namespace) -> None:
    assets = load_assets(Path(args.assets))
    rates = load_rates(Path(args.rates))
    target_month = month_start(parse_date(args.month + "-01"))
    rows = build_snapshot(assets, rates, target_month)
    invoices = invoice_for_month(rows, target_month)

    total_invoice = round(sum(float(r["invoice_amount"]) for r in invoices), 2)
    print(f"invoice lines: {len(invoices)}")
    print(f"invoice total ({target_month.isoformat()}): {total_invoice}")

    if args.output:
        write_csv(Path(args.output), invoices)
        print(f"invoice file written to {args.output}")


def run_bank_payable(args: argparse.Namespace) -> None:
    assets = load_assets(Path(args.assets))
    rates = load_rates(Path(args.rates))
    target_month = month_start(parse_date(args.month + "-01"))
    rows = build_snapshot(assets, rates, target_month)
    invoices = invoice_for_month(rows, target_month)
    bank_payable = round(sum(float(r["invoice_amount"]) for r in invoices), 2)
    print(f"bank payable for {target_month.isoformat()}: {bank_payable}")


def run_daily(args: argparse.Namespace) -> None:
    assets = load_assets(Path(args.assets))
    rates = load_rates(Path(args.rates))
    as_of = parse_date(args.as_of)
    rows = build_snapshot(assets, rates, as_of)
    totals = rows_total(rows)
    print_totals(totals, f"Daily portfolio report ({as_of.isoformat()})")

    current_month = month_start(as_of)
    invoices = invoice_for_month(rows, current_month)
    due_count = len(invoices)
    due_amount = round(sum(float(r["invoice_amount"]) for r in invoices), 2)
    print(f"\nmonth_to_invoice: {current_month.isoformat()}")
    print(f"invoice_lines_due: {due_count}")
    print(f"invoice_amount_due: {due_amount}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ALC lease tracker")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--assets", default="assets.csv", help="path to assets.csv")
    common.add_argument("--rates", default="rates.csv", help="path to rates.csv")

    s1 = sub.add_parser("snapshot", parents=[common], help="point-in-time asset snapshot")
    s1.add_argument("--as-of", required=True, help="YYYY-MM-DD")
    s1.add_argument("--output", help="csv output path")
    s1.set_defaults(func=run_snapshot)

    s2 = sub.add_parser("invoice", parents=[common], help="invoice list for YYYY-MM")
    s2.add_argument("--month", required=True, help="YYYY-MM")
    s2.add_argument("--output", help="csv output path")
    s2.set_defaults(func=run_invoice)

    s3 = sub.add_parser("bank-payable", parents=[common], help="bank payable for YYYY-MM")
    s3.add_argument("--month", required=True, help="YYYY-MM")
    s3.set_defaults(func=run_bank_payable)

    s4 = sub.add_parser("daily", parents=[common], help="daily operating report")
    s4.add_argument("--as-of", required=True, help="YYYY-MM-DD")
    s4.set_defaults(func=run_daily)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()