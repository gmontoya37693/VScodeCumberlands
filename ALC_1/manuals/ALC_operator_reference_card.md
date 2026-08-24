# ALC Operator Reference Card

## Routine Commands
### 1. Daily Monitoring
- `./scripts/op_daily.sh <operator> <as-of YYYY-MM-DD> [billing_day]`
- Example: `./scripts/op_daily.sh ana 2026-07-30 31`

### 2. Invoice Day
- `./scripts/op_invoice.sh <operator> <month YYYY-MM> [billing_day]`
- Example: `./scripts/op_invoice.sh ana 2026-07 31`

### 3. Month-End
- `./scripts/op_month_end.sh <operator> <month YYYY-MM> [billing_day]`
- Example: `./scripts/op_month_end.sh ana 2026-07 31`

## One-Time Setup
### Baseline
- `./scripts/op_init_baseline.sh <operator> <go-live YYYY-MM-DD> <notes>`
- Example: `./scripts/op_init_baseline.sh ana 2026-07-01 "Production start"`

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
- month to invoice
- billing date
- billing window due dates
- invoice lines due
- invoice amount due
- reminder if invoice day is tomorrow or today

## What Invoice-Day Run Does
- creates `invoices_YYYY-MM.csv` as the monthly billing handoff file for a third person to process
- posts the same invoiced rows into `posted_invoices.csv` as the internal invoice history log
- refreshes `ALC - Asset Calculation Unit.xlsx`
- creates manifest and backups
- keeps invoicing active assets until balance reaches zero

## What Month-End Run Does
- updates `bank_payable.csv`
- stores one row per month
- computes payable on asset cost funding basis (not lease base)
- stores bank interest and bank principal component totals
- closes the month in `closed_periods.csv`
- creates two manifests (bank-payable + close-period) and backups

## Important Concepts
### Due Date
- each asset has its own due date pattern based on start date

### Billing Date
- one monthly batch invoicing date
- training example: 31 (month-end policy)
- weekend shifts to next working day

### Month Close
- separate control step
- once closed, the month is locked

### Asset Lifecycle States
- **Scheduled**: Asset procured but not yet delivered (`as_of < start_date`). Not invoiced. Shown in daily reports as upcoming assets.
- **Active**: Asset on lease and invoicing (`as_of >= start_date AND balance > 0`). Included in portfolio totals and billing.
- **Matured**: Lease term complete (`balance == 0 OR as_of >= final_month`). No more invoicing. Remains in workbook for history.

## Training Reminders
- always provide the operator user name on every wrapper run
- rate used is nominal monthly rate (`APR/12`), not monthly compounded
- invoice date and month-end date may be different by process timing; if you operate them on the same date, use the same billing day on all wrappers (`op_daily.sh`, `op_invoice.sh`, `op_month_end.sh`)
- close all CSV and XLS/XLSX files before running scripts so wrappers can overwrite outputs

## Never Do This
- do not edit `posted_invoices.csv` manually
- do not edit `closed_periods.csv` manually
- do not create fake production history after baseline
- do not rely on memory instead of the daily run
- do not keep workbook/CSV files open during wrapper runs

## Quick Validation
After invoice day:
- invoice CSV exists
- posted ledger updated
- workbook refreshed
- manifest created

Interpretation:
- `invoices_YYYY-MM.csv` = outbound monthly processing file
- `posted_invoices.csv` = internal posted history

After month-end:
- bank payable CSV updated
- closed month recorded
- manifest created
