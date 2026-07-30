# ALC Week 1 SOW
## Python Commands Exposed as API Endpoints

Version: V1
Date: 2026-07-30
Owner: Finance Systems and Data Engineering
Status: Draft for approval

## 1) Purpose
Week 1 will package the existing deterministic ALC Python workflow into secure HTTP API endpoints so a Microsoft 365 agent can orchestrate runs without end users executing shell scripts.

## 2) Week 1 Goal
Expose core run actions as production-ready API endpoints with validation, run manifests, and consistent JSON responses.

## 3) In Scope (Week 1)
- API service skeleton and deployment-ready runtime.
- Endpoint wrappers for existing ALC operations.
- Deterministic execution of existing command workflow.
- Input validation and structured error handling.
- Artifact path return in JSON payloads.
- Run manifest passthrough and correlation IDs.
- Basic authentication pattern compatible with Microsoft 365 orchestration.
- Smoke tests and handoff notes.

## 4) Out of Scope (Week 1)
- Copilot Studio conversation design.
- Power Automate full orchestration.
- Teams cards and adaptive card UX.
- Multi-stage approvals.
- Dashboarding and analytics.

## 5) Command to Endpoint Mapping
The API will wrap current command behavior exactly. Finance logic remains in the existing Python engine.

1. Daily preview
- Current wrapper: scripts/op_daily.sh
- Endpoint: POST /api/v1/runs/daily-preview
- Behavior: Read-only summary for as-of date and billing day.

2. Invoice run
- Current wrapper: scripts/op_invoice.sh
- Endpoint: POST /api/v1/runs/invoice
- Behavior: Generates invoice file, updates posted ledger, refreshes workbook.

3. Month-end close
- Current wrapper: scripts/op_month_end.sh
- Endpoint: POST /api/v1/runs/month-end
- Behavior: Generates bank payable totals, closes period.

4. One-pager workbook refresh
- Current wrapper: scripts/op_one_pager.sh
- Endpoint: POST /api/v1/runs/one-pager
- Behavior: Rebuilds workbook tabs for inventory and asset sheets.

5. Health check
- Endpoint: GET /api/v1/health
- Behavior: Returns service status and script version.

## 6) Proposed Request Contracts
Common request fields for run endpoints:
- operator: string
- assets_path: optional string
- rates_path: optional string
- billing_day: integer
- correlation_id: optional string

### Daily preview request
{
  "operator": "ana",
  "as_of": "2026-04-29",
  "billing_day": 31,
  "assets_path": "assets.csv",
  "rates_path": "rates.csv"
}

### Invoice request
{
  "operator": "ana",
  "month": "2026-04",
  "billing_day": 31,
  "assets_path": "assets.csv",
  "rates_path": "rates.csv"
}

### Month-end request
{
  "operator": "ana",
  "month": "2026-04",
  "billing_day": 31,
  "assets_path": "assets.csv",
  "rates_path": "rates.csv"
}

## 7) Proposed Response Contract
All run endpoints return a standard envelope:
{
  "success": true,
  "command": "invoice",
  "run_id": "20260730T101500-ab12cd34",
  "summary": {
    "month": "2026-04",
    "invoice_lines": 23,
    "invoice_total": 1270.58,
    "bank_interest_total": 258.19,
    "bank_principal_total": 859.71,
    "bank_payable_total": 1117.90
  },
  "artifacts": {
    "manifest": "ALC/run_manifests/...json",
    "invoice_file": "ALC/invoices_2026-04.csv",
    "bank_payable_file": "ALC/bank_payable.csv",
    "workbook": "ALC/ALC - Asset Calculation Unit.xlsx"
  },
  "warnings": [],
  "errors": []
}

Error envelope:
{
  "success": false,
  "command": "invoice",
  "run_id": "20260730T101900-ef56ab78",
  "errors": [
    {
      "code": "INVALID_MONTH",
      "message": "month must be YYYY-MM"
    }
  ]
}

## 8) Technical Approach
- Keep existing engine file as single deterministic calculator.
- Add a thin API layer that:
  - validates payload,
  - invokes existing internal Python functions,
  - captures stdout and exceptions,
  - returns normalized JSON.
- No business math rewritten in API layer.

## 9) Security and Access (Week 1 baseline)
- API key or bearer token check at gateway.
- Request logging with operator and correlation ID.
- No secrets in response payload.
- Restrict writable output paths to ALC workspace directories.

## 10) Week 1 Workplan
Day 1
- Create API project skeleton.
- Add health endpoint and shared response model.

Day 2
- Implement daily-preview endpoint and validation.
- Add smoke test for deterministic output keys.

Day 3
- Implement invoice endpoint.
- Return artifact locations and totals.

Day 4
- Implement month-end endpoint and one-pager endpoint.
- Add error mapping and consistent HTTP statuses.

Day 5
- End-to-end smoke test sequence.
- Handoff package, runbook, and acceptance walkthrough.

## 11) Deliverables
- API source code in repository.
- Endpoint contract documentation.
- Environment configuration template.
- Smoke test script and sample payloads.
- Week 1 handoff note for M365 orchestration team.

## 12) Acceptance Criteria
1. Daily preview endpoint returns totals and due reminders for a valid date.
2. Invoice endpoint produces invoice file and posted ledger update.
3. Month-end endpoint produces bank payable totals and period-close record.
4. One-pager endpoint rebuilds workbook successfully.
5. Every endpoint returns run_id and manifest location.
6. Invalid payloads return structured errors without stack traces.
7. Deterministic rerun with same inputs produces matching financial totals.

## 13) Risks and Mitigations
- Risk: file lock conflicts on workbook.
  - Mitigation: detect lock and return actionable error code.
- Risk: accidental duplicate month-end posting.
  - Mitigation: guard check for closed_periods prior to write.
- Risk: user path mistakes.
  - Mitigation: default paths and strict validation.

## 14) Dependencies
- Python runtime aligned with current ALC scripts.
- Access to ALC working directory or mounted equivalent.
- Decision on hosting target: Azure Function or Azure Container Apps.
- Naming convention for operator IDs and correlation IDs.

## 15) Sign-off
Prepared by: __________________
Reviewed by: __________________
Approved by: __________________
Approval date: ________________
