# POC Proposal: Copilot Studio Agent with Digital Fingers for WTN Document Automation
 
## Executive Summary
 
Acento will pilot a Microsoft Teams-based Copilot Studio agent capable of assisting users with WTN document processes by automating interactions with a third-party web portal that does not expose APIs.
 
The solution will leverage Copilot Studio, Power Automate Cloud, and Power Automate Desktop running on the user's workstation through Microsoft Edge. The objective is to provide users with an AI assistant that not only answers questions but can also perform browser-based actions on their behalf, including navigating websites, generating reports, downloading documents, and storing deliverables in approved repositories.
 
---
 
# Business Challenge
 
Today, WTN-related processes require users to:
 
1. Access a web portal.
2. Authenticate using their own credentials.
3. Navigate multiple pages.
4. Generate and download documents.
5. Save, upload, or distribute resulting files.
 
These repetitive activities consume operational time, introduce manual errors, and reduce scalability of back-office processes.
 
Because the target platform does not currently provide an API, traditional system-to-system integration is not feasible.
 
---
 
# Proposed Solution
 
## User Experience
 
```text
Microsoft Teams
↓
Copilot Studio Agent
↓
"Download WTN Document"
↓
Power Automate Cloud Flow
↓
Power Automate Desktop
↓
Microsoft Edge
↓
WTN Portal
↓
Download Document
↓
SharePoint / Repository
↓
Confirmation to User
```
 
The user remains logged into the portal using their own credentials while the agent performs the required navigation and download tasks.
 
---
 
# Required Components
 
## 1. Copilot Studio Agent
 
### Purpose
 
- Entry point within Microsoft Teams.
- Conversational interface for users.
- Receives requests.
- Launches automation actions.
- Returns status updates and document links.
 
### Example Commands
 
- "Download WTN document."
- "Generate occupancy report."
- "Retrieve latest tenant document."
- "Export compliance report."
 
---
 
## 2. Power Automate Cloud Flow
 
### Purpose
 
- Orchestrates the automation process.
- Receives inputs from Copilot Studio.
- Triggers desktop automation.
- Captures status and completion messages.
- Delivers results back to the user.
 
---
 
## 3. Power Automate Desktop
 
### Purpose
 
Provides the agent's **digital fingers**.
 
### Responsibilities
 
- Launch Microsoft Edge.
- Navigate websites.
- Click buttons and links.
- Complete forms.
- Download documents.
- Save files to approved destinations.
 
This component performs browser actions that are not available through APIs.
 
---
 
## 4. Microsoft Edge
 
### Purpose
 
- Executes portal interactions.
- Leverages existing authenticated user sessions where possible.
- Allows users to see the automation in action.
- Reinforces trust and ownership throughout the process.
 
---
 
# Why We Selected This Approach
 
## 1. No API Available
 
The target website does not currently provide APIs for document retrieval.
 
As a result:
 
```text
API Integration ❌
Browser Automation ✅
```
 
Browser automation is the most practical integration method available.
 
---
 
## 2. Users Have Individual Credentials
 
Each employee accesses the portal using personal credentials.
 
### Benefits
 
- No shared service accounts.
- Better auditability.
- Better security governance.
- Existing permission structures remain unchanged.
- Access is performed on behalf of the authenticated user.
 
---
 
## 3. User Visibility Is Valuable
 
Users can observe the browser activity while automation is running.
 
### Benefits
 
- Greater transparency.
- Faster organizational adoption.
- Increased trust in automation.
- Easier troubleshooting during pilot phases.
- Reinforces the idea that the agent is assisting rather than replacing the user.
 
The automation behaves like a digital assistant sitting beside the employee.
 
---
 
## 4. Microsoft-First Environment
 
Acento already utilizes:
 
- Microsoft Teams
- Copilot Studio
- Power Automate
- Microsoft Edge
- SharePoint
 
This minimizes technology sprawl and accelerates implementation.
 
---
 
## Solution Implications
 
### Security
 
- Users maintain ownership of credentials.
- Authentication remains tied to existing business accounts.
- No centralized credential vault is required for the pilot.
 
### Operations
 
- Automation runs on the employee workstation.
- Browser execution is distributed across users instead of centralized servers.
- Infrastructure requirements remain minimal.
 
### User Experience
 
- Employees can watch the automation execute.
- Users remain in control of execution initiation.
- Results are returned directly within Teams.
 
---
 
# Alternatives Evaluated
 
## Alternative A: Azure Function + Playwright
 
### Advantages
 
- Highly scalable.
- Fully cloud-native.
- Supports unattended execution.
- Suitable for high-volume operations.
 
### Disadvantages
 
- Higher development complexity.
- Additional DevOps and monitoring requirements.
- More complex credential management.
- Reduced visibility for end users.
 
### Decision
 
Deferred to future phases.
 
This option becomes attractive once process volume and ROI justify a cloud-first automation architecture.
 
---
 
## Alternative B: Dedicated RPA Virtual Machines
 
### Advantages
 
- Centralized execution.
- Supports unattended processing.
- Easier centralized scheduling.
 
### Disadvantages
 
- Additional infrastructure.
- Additional licensing costs.
- Increased maintenance effort.
- Less aligned with user-owned credentials.
 
### Decision
 
Not required for the pilot phase.
 
---
 
# Success Criteria
 
## Operational Success
 
- 80% or greater reduction in manual navigation activities.
- 50% or greater reduction in report retrieval time.
- Greater than 95% successful document retrieval rate.
 
## User Adoption Success
 
- Positive feedback from pilot participants.
- Minimal training requirements.
- Consistent use through Microsoft Teams.
- Demonstrated reduction in repetitive administrative tasks.
 
## Technical Success
 
- Stable execution within supported portals.
- Successful document retrieval and storage.
- No sharing of credentials.
- No significant interruption to user workflows.
 
---
 
# Future Vision
 
## Phase 1: Assisted Automation
 
```text
User + Agent
```
 
The user requests a document and watches the process execute.
 
---
 
## Phase 2: Expanded Digital Assistant
 
```text
User + Agent + Multiple Portals
```
 
Additional workflows may include:
 
- Tenant insurance validation.
- Carrier portal navigation.
- Compliance documentation retrieval.
- Bulk report generation.
 
---
 
## Phase 3: Autonomous Agentic Processing
 
```text
Copilot Studio
↓
Azure Services
↓
Playwright Automation
↓
24/7 Autonomous Processing
```
 
High-volume workflows may eventually transition to cloud-hosted browser automation while preserving the same Teams-based user experience.
 
---
 
# Expected Outcome
 
Acento will deploy a Teams-based AI assistant capable of acting with **digital fingers** on behalf of employees, automating WTN document retrieval processes through browser interaction while preserving user ownership, security, visibility, and auditability.
 
The pilot minimizes implementation complexity, leverages existing Microsoft investments, and establishes a foundation for future agentic automation initiatives.