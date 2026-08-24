# ALC Azure V1 Enablement SOW

**Owner:** German Montoya (`gmontoya@acentopartners.com`)
**Status:** Request for Azure tenant enablement
**Scope:** Azure backend for the validated ALC Python engine

## Objective

Make the existing ALC engine available through Microsoft Teams/Copilot Studio without requiring users to run Python or shell scripts. The agent will read protected inputs, run calculations, deliver dated output copies, and require explicit authenticated approval before changing asset or rate inputs.

## Current State

The validated engine is in `ALC_1/alc_item_sheet_tracker.py` and already contains the tested financial calculations and operating behaviors, including:

- daily portfolio reporting
- snapshots and schedules
- invoice calculation and posted-invoice tracking
- bank-payable calculation
- one-pager workbook generation
- date, rate, and asset logic

A local FastAPI boundary has been started in `ALC_1/api/` for validation. No Azure resources or production deployment are requested by this repository change.

## Azure V1 To Be Built

```text
Copilot Studio / Teams
        |
   Power Automate
        |
   Authenticated API
        |
   ALC Python engine
        |
   Azure Blob Storage
```

### Blob logical structure

```text
inputs/assets.csv
inputs/rates.csv
state/posted_invoices.csv
state/bank_payable.csv
state/closed_periods.csv
working/<run_id>/
outputs/YYYY/MM/DD/
manifests/
archives/
```

Original input files remain protected. Every generated report is a new dated copy with a unique run ID.

The administrator is not expected to create each logical path or upload application files. After the storage resources and permissions are enabled, the owner will create the Blob structure, upload the initial input files, configure the application, and perform the deployment work.

## V1 API Capabilities

The API will provide:

- read current assets and rates
- run daily, snapshot, schedule, invoice, bank-payable, and one-pager operations
- read operational histories and run manifests
- return calculation summaries and secure links to dated CSV/XLSX outputs
- propose new or changed assets and rates
- commit asset/rate changes only after authenticated user approval

The existing Python engine remains the single source of financial truth. Business calculations will not be duplicated in Copilot, Power Automate, or the API.

## V1 Guardrails

- The normal API/agent identity can read `inputs/` but cannot write it.
- The agent cannot write directly to Blob Storage.
- Asset and rate changes require a proposal, explicit approval, input-hash verification, archival of the prior file, and an audit manifest.
- Outputs are written as dated copies and are not overwritten.
- Invoice, bank-payable, closed-period, and workbook output approval gates are deferred to V2; V1 focuses on protected inputs and reliable delivery.

## Requested Azure Enablement

Please enable or provision the Azure foundation below. The owner will perform the remaining setup and deployment work.

1. A resource group and private Azure Storage Account with Blob Storage.
2. A private container, or an approved equivalent storage structure, for the ALC paths listed above.
3. Prefer Azure Container Apps using a consumption/scale-to-zero plan for the API and Python runtime. Container Apps is preferred for FastAPI, Python dependencies, and workbook generation. Azure Functions is an acceptable alternative if it is the organizational standard.
4. A managed identity for the API runtime.
5. Entra ID authentication for the API, with access assigned to:
   - `gmontoya@acentopartners.com`
6. Blob data access for `gmontoya@acentopartners.com`, scoped to the approved ALC storage resource, so the owner can create the logical paths and upload the initial input files.
7. API permissions/RBAC for the runtime identity:
   - read `inputs/`
   - read/write `working/`
   - read/write operational `state/`
   - write `outputs/` and `manifests/`
   - write `archives/`
   - no write permission to `inputs/`
8. A separately controlled permission path for approved asset/rate commits, or an approved service identity/API operation that can write `inputs/` only after authenticated approval.
9. Required networking, secret/configuration storage, logging, and monitoring under organizational policy.
10. Scoped deployment permission for `gmontoya@acentopartners.com`, or a CI/CD identity operated by that user, to deploy and update application code within the approved resource group without subscription-wide administration.
11. Blob versioning, soft delete, and lifecycle retention rules.
12. A small consumption budget and cost alert.

The Microsoft 365 environment is already available to the owner. The owner can create and deploy Copilot Studio agents and create/connect Power Automate flows. No additional Copilot Studio or Power Automate environment permission is requested for this SOW.

Runtime access and deployment access are separate requirements. The API managed identity needs only runtime permissions. The owner or CI/CD identity needs scoped deployment permission so application updates can be published after development and testing. The owner will create the logical Blob paths, upload the initial inputs, configure the API, connect the existing Microsoft 365 workflow, and perform the application deployment.

## V1 Acceptance Criteria

- User can ask Copilot for current assets or rates.
- User can request a daily, snapshot, schedule, invoice, bank-payable, or one-pager run.
- API runs the existing engine without rewriting financial logic.
- Outputs appear as dated, secure, Teams-accessible files.
- Original `assets.csv` and `rates.csv` cannot be modified by the normal agent/API identity.
- Approved asset/rate changes are archived, auditable, and rejected if inputs changed after proposal.
- Azure usage is consumption-based or scale-to-zero with cost alerts.

## Deferred to V2

- approval before invoice posting
- approval before bank-payable updates
- approval before closing periods
- approval before final workbook publication
- expanded dashboards and analytics
