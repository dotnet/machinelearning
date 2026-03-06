---
description: "Daily repo health orchestrator: collects data on issues, PRs, and CI pipelines, diffs against previous run, updates a pinned dashboard issue, and dispatches investigators for critical findings."

on:
  schedule: "0 6 * * *"
  workflow_dispatch:

permissions:
  contents: read
  issues: read
  pull-requests: read
  actions: read

tools:
  github:
    toolsets: [repos, issues, pull_requests, actions]
  bash: ["gh", "curl", "jq", "date", "base64", "echo", "sort", "uniq", "head", "tail", "grep", "wc"]
  cache-memory: true

safe-outputs:
  create-issue:
    max: 1
  update-issue:
    max: 1
  add-comment:
    max: 1
  dispatch-workflow:
    max: 5
    workflows: ["repo-health-investigate"]
---

# Repo Health Check — Orchestrator

Collect health data for **dotnet/machinelearning**, diff against the previous run, update the dashboard issue, and dispatch investigators for critical findings.

## Configuration

| Setting | Value |
|---------|-------|
| Repository | `dotnet/machinelearning` |
| CI System | `azure-pipelines` |
| GitHub Actions Workflows | `backport.yml, locker.yml` |
| Known Baseline | `.github/health-baseline.md` |
| Priority Labels | `P0, P1, P2, P3, bug` |
| Review Labels | `api-ready-for-review, Awaiting User Input, needs-author-action` |
| Area Labels | *(none — use milestones and `untriaged` label instead)* |
| Investigate Workflow | `repo-health-investigate` |
| Max Dispatches | `5` |
| AzDO Org | `dnceng` |
| AzDO Project | `public` |
| AzDO Pipelines | `vsts-ci, codecoverage-ci, night-build, outer-loop-build` |

---

## Step 1 — Data Collection

Collect all health data using `gh` CLI and GitHub API. Store raw results for analysis.

### Issues

**I1. Open issues by priority/severity label**

```bash
# Count and age distribution by priority label
for label in P0 P1 P2 P3 bug; do
  echo "=== $label ==="
  gh issue list --repo dotnet/machinelearning \
    --label "$label" --state open \
    --json number,title,createdAt,assignees,labels,updatedAt \
    --limit 100
done
```

**I2. Issue velocity — opened vs closed in last 24h**

```bash
# Opened in last 24h
gh issue list --repo dotnet/machinelearning \
  --state all --json number,title,state,createdAt \
  --limit 200 | jq '[.[] | select(.createdAt > (now - 86400 | todate))]'

# Closed in last 24h
gh issue list --repo dotnet/machinelearning \
  --state closed --json number,title,closedAt \
  --limit 200 | jq '[.[] | select(.closedAt > (now - 86400 | todate))]'
```

**I3. Issues without triage (no milestone, untriaged label)**

```bash
gh issue list --repo dotnet/machinelearning \
  --state open --label "untriaged" \
  --json number,title,labels,createdAt,author \
  --limit 100

# Also check for issues with no milestone
gh issue list --repo dotnet/machinelearning \
  --state open --search "no:milestone" \
  --json number,title,labels,createdAt,author \
  --limit 100
```

**I4. Needs-info with no response > 14 days**

```bash
gh issue list --repo dotnet/machinelearning \
  --state open --label "need info,Awaiting User Input,needs-author-action" \
  --json number,title,updatedAt,comments \
  --limit 100
# Filter: updatedAt older than 14 days
```

**I5. Recent activity on old items (potential escalations)**

```bash
gh issue list --repo dotnet/machinelearning \
  --state open --json number,title,createdAt,updatedAt,comments \
  --limit 200
# Filter: createdAt > 90 days ago AND updatedAt < 3 days ago — old issue with new activity
```

### Pull Requests

**P1. Open PRs — count, age, review status**

```bash
gh pr list --repo dotnet/machinelearning \
  --state open \
  --json number,title,createdAt,author,reviewDecision,reviewRequests,isDraft,labels \
  --limit 200
```

**P2. PRs without reviewers assigned**

```bash
gh pr list --repo dotnet/machinelearning \
  --state open \
  --json number,title,createdAt,reviewRequests,reviewDecision \
  --limit 200
# Filter: reviewRequests is empty AND reviewDecision is not APPROVED
```

**P3. PRs with failing CI**

```bash
gh pr list --repo dotnet/machinelearning \
  --state open \
  --json number,title,statusCheckRollup \
  --limit 200
# Filter: statusCheckRollup contains FAILURE or ERROR
```

**P4. PRs waiting on author > 7 days**

```bash
gh pr list --repo dotnet/machinelearning \
  --state open --label "needs-author-action,Awaiting User Input" \
  --json number,title,updatedAt,author,labels \
  --limit 100
# Filter: updatedAt > 7 days ago
```

**P5. Stale PRs (no activity > 30 days)**

```bash
gh pr list --repo dotnet/machinelearning \
  --state open \
  --json number,title,updatedAt,author \
  --limit 200
# Filter: updatedAt > 30 days ago
```

**P6. PRs merged in last 24h**

```bash
gh pr list --repo dotnet/machinelearning \
  --state merged \
  --json number,title,mergedAt \
  --limit 100 | jq '[.[] | select(.mergedAt > (now - 86400 | todate))]'
```

### Pipelines / CI

**C1. Recent workflow run success/failure rate (GitHub Actions — bot workflows)**

```bash
for workflow in backport.yml locker.yml; do
  echo "=== $workflow ==="
  gh run list --repo dotnet/machinelearning \
    --workflow "$workflow" \
    --json status,conclusion,createdAt,event \
    --limit 30
done
```

**C2. Currently failing workflows on default branch**

```bash
for workflow in backport.yml locker.yml; do
  gh run list --repo dotnet/machinelearning \
    --workflow "$workflow" --branch main \
    --json conclusion,createdAt,url \
    --limit 5
done
```

**C3. Flaky tests (pass/fail alternation in last 10 runs)**

```bash
for workflow in backport.yml locker.yml; do
  gh run list --repo dotnet/machinelearning \
    --workflow "$workflow" --branch main \
    --json conclusion \
    --limit 10
done
# Detect alternating success/failure pattern — 3+ alternations = flaky
```

**C4. Average CI duration trend**

```bash
for workflow in backport.yml locker.yml; do
  gh run list --repo dotnet/machinelearning \
    --workflow "$workflow" --branch main \
    --json createdAt,updatedAt \
    --limit 20
done
# Calculate average duration and compare recent (5) vs older (5)
```

### Azure DevOps Pipelines

> **Note**: AzDO monitoring requires `AZDO_PAT` secret. The `dnceng` org requires authentication even for the `public` project. If `AZDO_PAT` is not set, skip all A1-A3 checks and note "AzDO pipeline monitoring disabled — no AZDO_PAT configured" in the dashboard.

**A1. Pipeline status — last run result**

```bash
# Check if AZDO_PAT is available; skip AzDO checks if not
if [ -z "$AZDO_PAT" ]; then
  echo "AZDO_PAT not set — skipping Azure DevOps pipeline checks"
else
for pipeline in vsts-ci codecoverage-ci night-build outer-loop-build; do
  curl -s -u ":$AZDO_PAT" \
    "https://dev.azure.com/dnceng/public/_apis/build/builds?definitions=$pipeline&\$top=1&api-version=7.0" \
    | jq '.value[0] | {id, buildNumber, status, result, queueTime, finishTime}'
done
```

**A2. Pipeline failure rate (last 7 days)**

```bash
SINCE=$(date -u -d '7 days ago' +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -v-7d +%Y-%m-%dT%H:%M:%SZ)
for pipeline in vsts-ci codecoverage-ci night-build outer-loop-build; do
  curl -s -u ":$AZDO_PAT" \
    "https://dev.azure.com/dnceng/public/_apis/build/builds?definitions=$pipeline&minTime=$SINCE&api-version=7.0" \
    | jq '[.value[] | .result] | group_by(.) | map({result: .[0], count: length})'
done
```

**A3. Queue times**

```bash
for pipeline in vsts-ci codecoverage-ci night-build outer-loop-build; do
  curl -s -u ":$AZDO_PAT" \
    "https://dev.azure.com/dnceng/public/_apis/build/builds?definitions=$pipeline&\$top=10&api-version=7.0" \
    | jq '[.value[] | {queueTime, startTime} | {wait: ((.startTime | fromdateiso8601) - (.queueTime | fromdateiso8601))}] | {avg_wait_seconds: (map(.wait) | add / length)}'
done
fi
```

---

## Step 2 — Fingerprint & Diff

Load previous findings from cache-memory at `/tmp/gh-aw/cache-memory/findings.json`. Generate deterministic fingerprints for current findings and classify each.

### Fingerprint Format

Each finding gets an ID: `{CHECK_ID}-{hash}` where hash is derived from the key attributes.

| Category | Fingerprint Components |
|----------|-----------------------|
| Issue finding | Check ID + issue numbers involved |
| PR finding | Check ID + PR numbers involved |
| Pipeline finding | Check ID + workflow name + failure type |

### Classification

```
For each current finding:
  fingerprint = generate_fingerprint(finding)
  if fingerprint in baseline_file:
    status = "📋 BASELINED"
  elif fingerprint in previous_run:
    days = (today - previous_run[fingerprint].first_seen).days
    status = "📌 EXISTING (Day {days})"
  else:
    status = "🆕 NEW"

For each previous finding NOT in current findings:
  status = "✅ RESOLVED"
  resolved_date = today
```

### Load Baseline

```bash
# Read baseline file if it exists
gh api repos/dotnet/machinelearning/contents/.github/health-baseline.md \
  --jq '.content' | base64 -d 2>/dev/null || echo "No baseline file"
```

### Save to Cache-Memory

Store the full findings list with fingerprints, statuses, and first-seen dates in `/tmp/gh-aw/cache-memory/findings.json` for the next run.

---

## Step 3 — Analysis

### Executive Summary

Write 1–2 sentences covering:
- Overall health status (Healthy / Warning / Critical)
- Most important change since last run
- Key trend direction

### Severity Classification

| Severity | Criteria |
|----------|----------|
| 🔴 Critical | AzDO pipelines failing on default branch (A1), P0 issues without assignee (I1), untriaged security issues (I3) |
| 🟡 Warning | Stale PRs > 5 (P5), issue backlog growing (I2 negative velocity), flaky tests (C3), CI slowing > 20% (C4) |
| ℹ️ Info | Normal velocity, resolved items, stable metrics |

### Correlations

Look for connections between findings:
- CI failures ↔ stale PRs (PRs may be blocked by CI)
- Rising untriaged issues ↔ missing milestones (triage process gap)
- Slow CI ↔ large PR count (resource contention)

---

## Step 4 — Dashboard Output

### Find or Create Dashboard Issue

```bash
# Find existing dashboard issue
ISSUE=$(gh issue list --repo dotnet/machinelearning \
  --label "repo-health" --state open \
  --json number --jq '.[0].number')

if [ -z "$ISSUE" ]; then
  # Create new dashboard issue
  ISSUE=$(gh issue create --repo dotnet/machinelearning \
    --title "🏥 Repo Health Dashboard" \
    --label "repo-health" \
    --body "$DASHBOARD_BODY")
  # Pin the issue
  gh issue pin "$ISSUE" --repo dotnet/machinelearning
fi
```

### Update Issue Body

Replace the entire issue body with the current state using the dashboard format. Include:

1. **Header** — Last updated timestamp, overall status emoji and counts
2. **Summary** — Executive summary (1-2 sentences)
3. **Findings tables** — Critical, Warning, Recently Resolved, Baselined
4. **Trends (7-day)** — Key metrics with directional arrows
5. **Footer** — Link to workflow run and baseline file

### Post Daily Comment

```bash
gh issue comment "$ISSUE" --repo dotnet/machinelearning \
  --body "$DELTA_SUMMARY"
```

The delta comment should include:
- Number of new/resolved/existing findings
- Any severity changes (warning → critical, etc.)
- Key metric changes with direction

---

## Step 5 — Triage Dispatch

For findings classified as 🔴 Critical or 🟡 Warning (high confidence):

```bash
# Budget: max 5 dispatches
DISPATCHED=0

for finding in critical_and_high_findings; do
  if [ $DISPATCHED -ge 5 ]; then
    break
  fi

  gh workflow run repo-health-investigate.lock.yml \
    --repo dotnet/machinelearning \
    -f finding_id="$FINDING_ID" \
    -f category="$CATEGORY" \
    -f severity="$SEVERITY" \
    -f summary="$SUMMARY" \
    -f health_issue_number="$ISSUE"

  DISPATCHED=$((DISPATCHED + 1))
done
```

Prioritize dispatches:
1. 🔴 Critical findings — always dispatch
2. 🟡 Warning findings — dispatch if budget remains, prefer NEW over EXISTING

---

## Rules

1. **Idempotent** — Running twice in the same state produces the same dashboard
2. **Budget-aware** — Never exceed safe-output limits
3. **Baseline-respecting** — Never flag baselined items as NEW
4. **Cache-dependent** — First run classifies everything as NEW (no previous data)
5. **Non-destructive** — Only creates/updates the dashboard issue, never modifies source issues or PRs
