---
description: "Investigation worker: performs deep-dive analysis on a specific health check finding. Dispatched by repo-health-check for critical/high findings. Reports results back to the dashboard issue."

on:
  workflow_dispatch:
    inputs:
      finding_id:
        description: "Fingerprint of the finding to investigate"
        required: true
      category:
        description: "Finding category: issue, pr, or pipeline"
        required: true
      severity:
        description: "Finding severity: critical, high, or medium"
        required: true
      summary:
        description: "One-line description of the finding"
        required: true
      health_issue_number:
        description: "Dashboard issue number to report back to"
        required: true

permissions:
  contents: read
  issues: read
  pull-requests: read
  actions: read

tools:
  github:
    toolsets: [repos, issues, pull_requests, actions]
  bash: ["gh", "curl", "jq", "date", "echo", "sort", "uniq", "head", "tail", "grep", "wc"]

safe-outputs:
  add-comment:
    max: 1
    target: "*"
---

# Repo Health — Investigate Finding

Deep-dive investigation of a specific finding from the repo health check.

**Finding**: `${{ inputs.finding_id }}` — ${{ inputs.summary }}
**Category**: ${{ inputs.category }}
**Severity**: ${{ inputs.severity }}
**Dashboard**: #${{ inputs.health_issue_number }}

---

## Step 1 — Route to Category Playbook

Based on the `category` input, follow the appropriate investigation path.

### If category = `issue`

Deep-dive into the issue(s) referenced in the finding:

```bash
# Get full issue details
gh issue view ISSUE_NUMBER --repo dotnet/machinelearning --json title,body,comments,labels,assignees,milestone,createdAt,updatedAt

# Check for related PRs
gh pr list --repo dotnet/machinelearning --search "ISSUE_NUMBER" --json number,title,state,url

# Check if this is a regression — search for related closed issues
gh issue list --repo dotnet/machinelearning --state closed --search "KEYWORDS" --json number,title,closedAt --limit 10

# Check recent timeline
gh api repos/dotnet/machinelearning/issues/ISSUE_NUMBER/timeline --jq '.[].event' | sort | uniq -c
```

Analyze:
- Is this a regression of a previously fixed issue?
- Are there related PRs that might fix it?
- Has anyone been working on it (timeline events)?
- Is it properly triaged (labels, milestone, assignee)?

### If category = `pr`

Deep-dive into the PR(s) referenced in the finding:

```bash
# Get full PR details
gh pr view PR_NUMBER --repo dotnet/machinelearning --json title,body,author,reviewDecision,reviewRequests,statusCheckRollup,files,comments,createdAt,updatedAt

# Check CI status details
gh pr checks PR_NUMBER --repo dotnet/machinelearning

# Get CI failure logs (if failing)
FAILED_RUN=$(gh run list --repo dotnet/machinelearning --branch PR_BRANCH --json databaseId,conclusion --jq '[.[] | select(.conclusion == "failure")][0].databaseId')
if [ -n "$FAILED_RUN" ]; then
  gh run view "$FAILED_RUN" --repo dotnet/machinelearning --log-failed 2>/dev/null | tail -100
fi

# Check review comments for blockers
gh pr view PR_NUMBER --repo dotnet/machinelearning --json reviewRequests,latestReviews --jq '.latestReviews[] | {author: .author.login, state: .state, body: .body}'
```

Analyze:
- What's blocking the PR? (CI failure, missing review, author unresponsive)
- If CI failing — what's the failure? Is it related to PR changes or flaky?
- If waiting on review — are reviewers assigned? Have they been pinged?
- If stale — is there a reason noted in comments?

### If category = `pipeline`

Deep-dive into the pipeline/workflow failure:

```bash
# For Azure DevOps pipelines — get recent runs with details
curl -s -u ":$AZDO_PAT" \
  "https://dev.azure.com/dnceng/public/_apis/build/builds?definitions=PIPELINE_ID&\$top=10&api-version=7.0" \
  | jq '.value[] | {id, buildNumber, result, sourceBranch, startTime, finishTime}'

# Get failed build timeline for root cause
FAILED_BUILD_ID=$(curl -s -u ":$AZDO_PAT" \
  "https://dev.azure.com/dnceng/public/_apis/build/builds?definitions=PIPELINE_ID&resultFilter=failed&\$top=1&api-version=7.0" \
  | jq -r '.value[0].id')

curl -s -u ":$AZDO_PAT" \
  "https://dev.azure.com/dnceng/public/_apis/build/builds/$FAILED_BUILD_ID/timeline?api-version=7.0" \
  | jq '[.records[] | select(.result == "failed") | {name, result, errorCount, issues: [.issues[]? | {type, message}]}]'

# Compare with last green run
GREEN_BUILD_ID=$(curl -s -u ":$AZDO_PAT" \
  "https://dev.azure.com/dnceng/public/_apis/build/builds?definitions=PIPELINE_ID&resultFilter=succeeded&\$top=1&api-version=7.0" \
  | jq -r '.value[0].id')

# Check for infrastructure vs code issues
curl -s -u ":$AZDO_PAT" \
  "https://dev.azure.com/dnceng/public/_apis/build/builds/$FAILED_BUILD_ID/timeline?api-version=7.0" \
  | jq '[.records[] | select(.result == "failed") | .issues[]? | .message]' \
  | grep -iE "timeout|oom|disk.space|rate.limit|connection|pool|agent" | head -10
```

For GitHub Actions workflows (bot workflows):

```bash
# Get recent runs with details
gh run list --repo dotnet/machinelearning --workflow "WORKFLOW_NAME" --branch main --json databaseId,conclusion,createdAt,event --limit 10

# Get failed run logs
FAILED_RUN=$(gh run list --repo dotnet/machinelearning --workflow "WORKFLOW_NAME" --branch main --json databaseId,conclusion --jq '[.[] | select(.conclusion == "failure")][0].databaseId')
gh run view "$FAILED_RUN" --repo dotnet/machinelearning --log-failed 2>/dev/null | tail -200

# Check for infrastructure vs code issues
gh run view "$FAILED_RUN" --repo dotnet/machinelearning --log-failed 2>/dev/null | grep -iE "timeout|oom|disk.space|rate.limit|connection" | head -10
```

Analyze:
- Is this a code issue or infrastructure issue?
- When did it start failing? What commit introduced the failure?
- Is there a pattern (always fails, intermittent, specific platform)?
- Compare failure output with last green run

---

## Step 2 — Gather Evidence

Collect all relevant data points. Be thorough but focused — gather only what's needed for root cause analysis.

Build an evidence list:
```markdown
### Evidence

1. [What was observed] — [Source: API call / log line / issue comment]
2. [Pattern or data point] — [Source]
3. ...
```

---

## Step 3 — Root Cause Analysis

Based on evidence, determine the most likely root cause.

Classify confidence:
- **High confidence** — Clear evidence points to a single cause
- **Medium confidence** — Evidence suggests a cause but some ambiguity
- **Low confidence** — Multiple possible causes, insufficient data

---

## Step 4 — Remediation Recommendation

Provide actionable next steps:

1. **Immediate** — What should be done right now?
2. **Short-term** — What should be done this week?
3. **Long-term** — What structural change would prevent recurrence?

---

## Step 5 — Report Back

Post a single comment on the dashboard issue (#${{ inputs.health_issue_number }}).

```bash
gh issue comment ${{ inputs.health_issue_number }} --repo dotnet/machinelearning --body "$REPORT"
```

### Report Format

```markdown
## 🔍 Investigation: `FINDING_ID`

**Finding**: SUMMARY
**Severity**: SEVERITY
**Category**: CATEGORY

### Evidence

1. Evidence point 1
2. Evidence point 2
3. ...

### Root Cause

**Confidence**: High / Medium / Low

[Root cause analysis — 2-3 sentences]

### Recommendations

| Priority | Action | Who |
|----------|--------|-----|
| 🔴 Immediate | [Action] | [Team/Person] |
| 🟡 Short-term | [Action] | [Team/Person] |
| ℹ️ Long-term | [Action] | [Team/Person] |

### Related

- Issue/PR links
- Previous occurrences
- Related findings

---
*Investigated by repo-health-investigate • Finding: `FINDING_ID`*
```

---

## Rules

1. **Single-finding focus** — Investigate only the finding specified in inputs
2. **Evidence-based** — Every conclusion must cite evidence
3. **Non-destructive** — Read-only except for the final comment
4. **Budget: 1 comment** — Post exactly one comment on the dashboard issue
5. **Concise** — Keep report under 500 lines; use collapsible sections for long logs
