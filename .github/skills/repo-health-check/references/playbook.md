# Repo health check playbook

This playbook contains the detailed ML.NET repository-health methodology. The local execution and approval rules in [`../SKILL.md`](../SKILL.md) override any historical workflow-oriented wording or mutating command examples below.

# Repo Health Check — Orchestrator

Collect health data for **dotnet/machinelearning**, diff against the previous run, draft a dashboard update, and investigate critical findings locally.

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
| Investigation Skill | `repo-health-investigate` |
| Max Investigations | `5` |
| AzDO Org | `dnceng-public` |
| AzDO Project | `public` |
| Enabled AzDO Definitions | `167` (`MachineLearning-CI`), `168` (`MachineLearning-CodeCoverage`) |
| Disabled AzDO Definition | `169` (`MachineLearning-NightlyBuild`) — report as a coverage gap |

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
  --state open --search 'label:"need info","Awaiting User Input","needs-author-action"' \
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
  --state open --search 'label:"needs-author-action","Awaiting User Input"' \
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

> **Note**: Prefer an authenticated Azure DevOps tool or `AZDO_PAT` when available. The `dnceng-public/public` REST API is publicly readable, so a missing PAT alone is not a coverage gap. If both authenticated access and the public API fail, skip A1-A3 and report the HTTP or tool error. Definition `169` is disabled, and there is no standalone outer-loop definition; report those coverage gaps rather than querying nonexistent pipelines.

```bash
AZDO_AUTH=()
if [ -n "${AZDO_PAT:-}" ]; then
  AZDO_AUTH=(-u ":$AZDO_PAT")
fi

AZDO_BUILD_URL="https://dev.azure.com/dnceng-public/public/_apis/build"
ENABLED_DEFINITIONS="167 168"
```

**A1. Pipeline status — last run result**

```bash
for definition_id in $ENABLED_DEFINITIONS; do
  curl -fSs "${AZDO_AUTH[@]}" \
    "$AZDO_BUILD_URL/builds?definitions=$definition_id&branchName=refs%2Fheads%2Fmain&queryOrder=finishTimeDescending&\$top=1&api-version=7.0" \
    | jq -e 'if (.value | length) == 0 then error("No main-branch builds returned") else .value[0] | {definition: .definition.name, id, buildNumber, status, result, queueTime, finishTime} end'
done

# Surface disabled scheduled coverage explicitly.
curl -fSs "${AZDO_AUTH[@]}" \
  "$AZDO_BUILD_URL/definitions/169?api-version=7.0" \
  | jq '{id, name, queueStatus}'
```

**A2. Pipeline failure rate (last 7 days)**

```bash
SINCE=$(date -u -d '7 days ago' +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -v-7d +%Y-%m-%dT%H:%M:%SZ)
for definition_id in $ENABLED_DEFINITIONS; do
  curl -fSs "${AZDO_AUTH[@]}" \
    "$AZDO_BUILD_URL/builds?definitions=$definition_id&branchName=refs%2Fheads%2Fmain&minTime=$SINCE&api-version=7.0" \
    | jq '[.value[] | .result] | group_by(.) | map({result: .[0], count: length})'
done
```

**A3. Queue times**

```bash
for definition_id in $ENABLED_DEFINITIONS; do
  curl -fSs "${AZDO_AUTH[@]}" \
    "$AZDO_BUILD_URL/builds?definitions=$definition_id&branchName=refs%2Fheads%2Fmain&queryOrder=finishTimeDescending&\$top=10&api-version=7.0" \
    | jq '
        def epoch: sub("\\.[0-9]+Z$"; "Z") | fromdateiso8601;
        [.value[]
          | select(.queueTime != null and .startTime != null)
          | {wait: ((.startTime | epoch) - (.queueTime | epoch))}]
        | if length == 0
          then {samples: 0, avg_wait_seconds: null}
          else {samples: length, avg_wait_seconds: (map(.wait) | add / length)}
          end'
done
```

---

## Step 2 — Fingerprint & Diff

Load previous findings from `/tmp/mlnet-repo-health/findings.json`. Generate deterministic fingerprints for current findings and classify each.

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

### Save Local State

Store the full findings list with fingerprints, statuses, and first-seen dates in `/tmp/mlnet-repo-health/findings.json` for the next run.

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

### Find or Draft Dashboard Issue

```bash
# Find existing dashboard issue
ISSUE=$(gh issue list --repo dotnet/machinelearning \
  --label "repo-health" --state open \
  --json number --jq '.[0].number')

if [ -z "$ISSUE" ]; then
  echo "No dashboard issue found. Draft a new issue and request approval before creating or pinning it."
fi
```

### Update Issue Body

Replace the entire issue body with the current state using the dashboard format. Include:

1. **Header** — Last updated timestamp, overall status emoji and counts
2. **Summary** — Executive summary (1-2 sentences)
3. **Findings tables** — Critical, Warning, Recently Resolved, Baselined
4. **Trends (7-day)** — Key metrics with directional arrows
5. **Footer** — Link to the local skill and baseline file

### Draft Daily Comment

```bash
printf '%s\n' "$DELTA_SUMMARY"
```

The delta comment should include:
- Number of new/resolved/existing findings
- Any severity changes (warning → critical, etc.)
- Key metric changes with direction

---

## Step 5 — Local Investigations

For findings classified as 🔴 Critical or 🟡 Warning with high confidence, investigate up to five
locally by following [`../../repo-health-investigate/SKILL.md`](../../repo-health-investigate/SKILL.md).
If the user requested check-only mode, produce an investigation queue instead.

Prioritize investigations:
1. 🔴 Critical findings — always investigate first
2. 🟡 Warning findings — use remaining budget and prefer NEW over EXISTING

---

## Rules

1. **Idempotent** — Running twice in the same state produces the same dashboard
2. **Budget-aware** — Draft at most one dashboard update, one delta comment, and five investigations
3. **Baseline-respecting** — Never flag baselined items as NEW
4. **State-dependent** — First run classifies everything as NEW when no local state or dashboard history exists
5. **Non-destructive** — GitHub writes require explicit approval and never modify source issues or PRs
