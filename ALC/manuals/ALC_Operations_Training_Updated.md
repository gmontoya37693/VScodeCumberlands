# ALC Operations Training

## Slide 1: Title
- ALC Operations Training
- Running ALC safely from clean go-live through daily monitoring, invoice-day posting, and month-end close
- Internal training for Ana and Andres

Presenter note:
Set the tone: this is a no-code operator workflow. The goal is to use only the approved files and wrapper scripts.

## Slide 2: Class Goal
- Teach operators how ALC works from a zero-record baseline
- Show which files operators edit and which files the system controls
- Practice the 3-step recurring operating cycle
- Build confidence in validation and auditability

Presenter note:
Stress that production starts clean. History begins only when real operations begin.

## Slide 3: Business Story
- We receive or update assets and bank rates
- Each asset begins a lease life based on its activation date
- The system calculates expected installments across that life
- Daily we monitor the portfolio and upcoming billing activity
- On billing day we post invoices in one batch
- At month-end we record bank payable and close the month
- The system preserves history, locks closed months, and leaves an audit trail

Presenter note:
Repeat the message: users manage business inputs, the system manages controls and outputs.

## Slide 4: The 3-Step Operating Cycle
- Every day: run the daily script
- On invoice day: run the invoice-day script
- At month-end: run the month-end script

Presenter note:
Keep this simple. Routine operations use only three recurring commands after baseline.

## Slide 5: Due Date vs Billing Date vs Month Close
- Due date: each asset has its own anniversary-based installment date
- Billing date: one monthly batch invoicing date, default example 22nd
- Weekend rule: billing date shifts to the next working day
- Month close: a separate control step that locks the month

Presenter note:
This is the most important concept to avoid confusion. Asset due date is not the same as billing date.

## Slide 6: Learning Objectives
- Identify the core files and their purpose
- Explain due date, billing date, and month close
- Run baseline once before production
- Run daily, invoice-day, and month-end wrappers
- Validate outputs and audit traces
- Recognize common operating mistakes

Presenter note:
Make it clear they do not need to use developer commands or non-routine helper scripts.

## Slide 7: File Map
### User-managed inputs
- assets.csv
- rates.csv

### System / control files
- baseline_config.json
- posted_invoices.csv
- closed_periods.csv

### Operational outputs
- invoices_YYYY-MM.csv
- bank_payable.csv
- ALC - Asset Calculation Unit.xlsx

### Audit / traceability
- run_manifests/
- backups/

Presenter note:
Only assets and rates are operator-maintained business inputs.

## Slide 8: What Operators Edit vs Never Edit
### Operators edit
- assets.csv
- rates.csv

### Operators do not edit manually
- posted_invoices.csv
- closed_periods.csv
- bank_payable.csv
- files in run_manifests/
- files in backups/

Presenter note:
This slide is a guardrail. If they remember only one file rule, it should be this one.

## Slide 9: Clean Baseline and Go-Live
- Baseline is a one-time clean production start
- Posted invoice history starts empty
- Closed periods start empty
- Bank payable history starts empty
- No fake or sample history should remain in live production files
- Real history begins only after go-live activity

Presenter note:
Baseline is not an invoice run and not a month close.

## Slide 10: Baseline Command
- Command:
  - ./scripts/op_init_baseline.sh german 2026-07-01 "Production start"
- Expected result:
  - baseline config created
  - posted ledger reset to header only
  - closed periods reset to header only
  - manifest created

Presenter note:
This is a one-time operation before production use.

## Slide 11: Daily Run
- Command:
  - ./scripts/op_daily.sh german 2026-07-21 22
- Daily run monitors, but does not post accounting history
- Wrapper output includes visual separators for readability:
  - blank space before and after each run
  - line separator: `______________`
- Daily run prints:
  - portfolio totals
  - month to invoice
  - billing date
  - billing window due dates
  - invoice lines due
  - invoice amount due
  - reminder if invoice day is tomorrow or today

Presenter note:
Daily run keeps the process from depending on memory.

## Slide 12: Invoice-Day Run
- Command:
  - ./scripts/op_invoice.sh german 2026-07 22
- Invoice-day run:
  - reads current assets and rates
  - reads posted history first
  - determines the billing date for the month
  - posts invoice rows due in the billing window
  - creates the monthly invoice CSV handoff for a third person to process
  - updates the posted invoice ledger as internal invoicing history
  - refreshes the workbook automatically

Presenter note:
Invoice-day posts history. It does not close the month.

## Slide 13: Month-End Run
- Command:
  - ./scripts/op_month_end.sh german 2026-07 22
- Month-end run:
  - updates bank_payable.csv
  - stores one row per month
  - closes the month in closed_periods.csv
  - writes two manifests (bank-payable and close-period) and backups

Presenter note:
Month-end records payable and locks the month. It is the control step.

## Slide 14: Workbook and Outputs
- Workbook: ALC - Asset Calculation Unit.xlsx
- First tab: Inventory
- Remaining tabs: one tab per asset
- Inventory shows the portfolio as of the workbook snapshot date and displays that as-of date
- Asset tabs show lease inputs, lifecycle status, and schedule state
- Invoice-day refreshes the workbook automatically
- invoices_YYYY-MM.csv is the monthly outbound billing file for processing
- posted_invoices.csv is the internal posted invoice history log

Presenter note:
Operators should review the workbook, not maintain it by hand.

## Slide 15: Class Drill Sequence
- Start from a clean workspace: inputs ready, outputs empty, no history yet
- Use the three training assets: Windblower at WWE, Snow Blower at CHA, Golf Cart at LPA
- Use billing day 22 for all steps
- No manual `echo` blocks are needed in class; wrappers already print separators.
- Run the sequence in order:
  - ./scripts/op_daily.sh german 2026-07-09 22
  - ./scripts/op_daily.sh german 2026-07-21 22
  - ./scripts/op_invoice.sh german 2026-07 22
  - ./scripts/op_daily.sh german 2026-07-23 22
  - ./scripts/op_month_end.sh german 2026-07 22
  - ./scripts/op_invoice.sh german 2026-08 22
  - ./scripts/op_month_end.sh german 2026-08 22

Presenter note:
This drill proves preview, post, close, and carry-forward behavior across two billing cycles.

## Slide 16: Validation Checklist
- assets.csv and rates.csv are valid
- invoice CSV exists after invoice-day run
- posted_invoices.csv has new rows after invoice-day run
- workbook refreshed and Inventory tab is first
- bank_payable.csv has one row for the month after month-end
- closed_periods.csv includes the month after month-end
- manifest and backups exist after write operations
- month-end run adds two manifests (bank-payable and close-period)
- each run is clearly delimited in terminal by separator blocks

Presenter note:
Validation is part of the job, not an optional extra.

## Slide 17: Common Errors
- Running inside Python >>> instead of shell terminal
- Wrong date format in CSV
- Missing or wrong file path
- Confusing due date with billing date
- Using `.op_...` instead of `./scripts/op_...`
- Trying to post a closed month
- Trying to invoice before go-live
- Expecting weekend billing to stay on a weekend date

Presenter note:
Explain what each error means and what the safe next action should be.

## Slide 18: Roles and Guardrails
### Operator
- update assets.csv
- update rates.csv
- run wrappers
- validate outputs

### Admin / owner
- manage baseline and overrides
- handle exceptional corrections
- manage structure changes

### Guardrails
- never edit posted ledger manually in normal operations
- never edit closed periods manually in normal operations
- never create fake production history after baseline

Presenter note:
The workflow should stay boring and repeatable.

## Slide 19: Success Looks Like
- User can explain the 3-step operating cycle
- User can explain due date vs billing date vs month close
- User can identify what they edit and what the system controls
- User can run the routine wrappers without developer help
- User can validate outputs and locate audit evidence

Presenter note:
End by reinforcing simplicity and confidence: same files, same scripts, same checks every cycle.
