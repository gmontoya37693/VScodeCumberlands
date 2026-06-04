# ALC Item Sheet Operating Manual

## Purpose
This manual defines the daily, weekly, month-end, and exception-handling workflow for Acento Leasing Company (ALC) asset invoicing and lease tracking.

The system is designed to be simple:
- One script
- One asset table
- One rate table
- One monthly ledger
- One daily report
- One monthly invoice run
- One bank-payable run

## Operating Model

The script should support two modes:

1. Guided mode
- The script asks the operator a short sequence of questions.
- It only asks questions when action is needed.
- It then routes the operator to the next required task.

2. Report-only mode
- If nothing changed and no billing action is due, the script skips questions.
- It generates the daily report, inventory snapshot, and maturity summary.
- It exits cleanly with a message such as: "No operational actions required today."

## Daily Behavior

Every day the operator runs the script once.

The script checks, in this order:
1. New assets added today?
2. Any asset status changes?
3. Any new bank rate effective today?
4. Any invoices due today?
5. Any bank payment due today?
6. Any exceptions or mismatches?

If the answer is "no" to all of the above, the script produces only the report package.

If the answer is "yes" to any item, the script prompts for the missing data and then continues.

## What the Script Should Ask

The script should only ask for facts the system cannot infer.

Typical questions:
- Did any new asset enter service today?
- Was any asset disposed, transferred, or deactivated?
- Did the bank publish a new annual rate?
- Are we closing the month today?
- Do you want to generate invoices now?
- Do you want to generate the bank payment summary now?
- Do you want to export CSV, PDF, or both?

The script should not ask repetitive questions if no action is needed.

## Core Workflows

### 1. Asset Entry Workflow
When a new asset is purchased or allocated:
- Create the asset record
- Assign property and allocation
- Set start date and lease term
- Set asset value and salvage value
- Assign bank rate and NIM assumption
- Assign GL account
- Mark asset as active

The system then calculates:
- monthly installment
- expected maturity date
- opening balance
- amortization schedule

### 2. Daily Operations Workflow
Each morning the operator runs the daily script.

The script outputs:
- active asset count
- assets starting soon
- assets maturing soon
- invoices due today
- bank payments due today
- assets with missing fields
- balance reconciliation totals

If no action is needed, the operator simply reviews the report and closes the task.

### 3. Monthly Invoicing Workflow
Use one monthly billing cycle for all active assets.

Recommended policy:
- Close billing on the last day of the month
- Invoice all active assets as of the close date
- Include line-item detail by asset or allocation
- Group by property for customer invoicing

This is easier than anniversary billing because it keeps accounting, collections, and bank reconciliation aligned.

### 4. Bank Payment Workflow
The system should also create a monthly bank-payable summary.

That summary shows:
- principal due
- interest due
- total debt service due
- outstanding balance by asset
- total portfolio balance

The operator uses this to pay the bank and cross-check the debt ledger.

### 5. Month-End Close Workflow
At month-end the operator should:
- confirm all asset changes are entered
- confirm the bank rate table is current
- run the month-end ledger close
- generate invoices
- generate bank payable report
- export reports to accounting
- save a locked month snapshot

After close, changes should be handled by adjustment entries, not by editing closed rows.

## Reports

### Daily Reports
1. Portfolio summary
2. Active asset inventory
3. Maturity schedule
4. Invoice due list
5. Bank payable due list
6. Exception list
7. Reconciliation summary

### Monthly Reports
1. Invoice register
2. Property-level billing summary
3. Asset-level amortization report
4. Bank debt summary
5. Closed-month snapshot
6. Variance report

### Exception Reports
1. Asset missing start date
2. Asset missing GL account
3. Asset with zero or negative balance unexpectedly
4. Rate change with no effective date
5. Invoice mismatch
6. Bank balance mismatch

## Duties by Role

### Operator / Agent
- Run the script daily
- Review exception list
- Enter new assets
- Confirm rate changes
- Launch invoicing on month-end
- Launch bank payable run
- Save or export reports

### Accounting / Finance
- Review invoices
- Post receivables
- Confirm bank payment
- Reconcile ledger vs bank debt
- Approve adjustment entries

### Management
- Review summary dashboards
- Watch active inventory
- Monitor maturity exposure
- Track revenue and debt trends

## Recommended System Behavior

The script should behave like a guided checklist:
- If a task is needed, prompt for it
- If a task is not needed, skip it
- If no action is needed at all, generate reports and exit

That means the operator always has a clear next step.

## Suggested Daily Decision Flow

1. Load master data.
2. Check for new assets.
3. Check for rate changes.
4. Check for invoices due.
5. Check for bank payment due.
6. If action exists, prompt the operator.
7. If no action exists, generate the daily report package.
8. Exit.

## Suggested Month-End Decision Flow

1. Are all assets entered?
2. Are all rates updated?
3. Has the month been closed?
4. Generate invoices.
5. Generate bank payable report.
6. Export closed-month package.
7. Lock the month.

## Operating Principle

The company should be run from the report, not from memory.

If the report is clean, no action is needed.
If the report shows an exception, the script should surface it immediately and tell the operator what to do next.
