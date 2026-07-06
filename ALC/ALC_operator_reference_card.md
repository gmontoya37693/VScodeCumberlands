# ALC Operator Reference Card

## Routine Commands
### 1. Daily Monitoring
- `ALC/scripts/op_daily.sh <operator> <as-of YYYY-MM-DD> [billing_day]`
- Example: `ALC/scripts/op_daily.sh german 2026-07-21 22`

### 2. Invoice Day
- `ALC/scripts/op_invoice.sh <operator> <month YYYY-MM> [billing_day]`
- Example: `ALC/scripts/op_invoice.sh german 2026-07 22`

### 3. Month-End
- `ALC/scripts/op_month_end.sh <operator> <month YYYY-MM> [billing_day]`
- Example: `ALC/scripts/op_month_end.sh german 2026-07 22`

## One-Time Setup
### Baseline
- `ALC/scripts/op_init_baseline.sh <operator> <go-live YYYY-MM-DD> <notes>`
- Example: `ALC/scripts/op_init_baseline.sh german 2026-07-01 "Production start"`

## Files Operators Edit
- `assets.csv`
- `rates.csv`

## Files Operators Review But Do Not Edit
- `posted_invoices.csv`
- `closed_periods.csv`
- `bank_payable.csv`
- `ALC - Asset Calculation Unit.xlsx`
- `run_manifests/`
- `backups/`

## What Daily Run Tells You
- current portfolio totals
- billing month
- billing date
- invoice lines due
- invoice amount due
- reminder if invoice day is tomorrow or today

## What Invoice-Day Run Does
- posts invoice rows into `posted_invoices.csv`
- creates `invoices_YYYY-MM.csv`
- refreshes `ALC - Asset Calculation Unit.xlsx`
- creates manifest and backups

## What Month-End Run Does
- updates `bank_payable.csv`
- stores one row per month
- closes the month in `closed_periods.csv`
- creates manifests and backups

## Important Concepts
### Due Date
- each asset has its own due date pattern based on start date

### Billing Date
- one monthly batch invoicing date
- default example: 22
- weekend shifts to next working day

### Month Close
- separate control step
- once closed, the month is locked

## Never Do This
- do not edit `posted_invoices.csv` manually
- do not edit `closed_periods.csv` manually
- do not create fake production history after baseline
- do not rely on memory instead of the daily run

## Quick Validation
After invoice day:
- invoice CSV exists
- posted ledger updated
- workbook refreshed
- manifest created

After month-end:
- bank payable CSV updated
- closed month recorded
- manifest created
