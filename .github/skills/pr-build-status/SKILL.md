---
name: pr-build-status
description: "Read CI build results for a PR — which jobs failed, why, and what the error messages say. Use when asked about build status, CI failures, or why a PR is red. This is the 'eyes' of the CI feedback loop."
---

# PR Build Status

The agent's ability to see CI results. Without this, agents push code blindly.

## CI System: Azure Pipelines (primary) + GitHub Actions (secondary)

ML.NET uses Azure DevOps Pipelines (`build/vsts-ci.yml`) for official CI and GitHub Actions for auxiliary workflows (backport, locker, copilot-setup-steps).

## For Azure Pipelines

```bash
# List builds for a PR (requires AZDO_PAT env var)
curl -s "https://dev.azure.com/dnceng-public/public/_apis/build/builds?branchName=refs/pull/PR_NUM/merge&api-version=7.0" \
  -H "Authorization: Basic $(echo -n :$AZDO_PAT | base64)" | jq '.value[] | {id, status, result}'

# Get build timeline (shows failed tasks)
curl -s "https://dev.azure.com/dnceng-public/public/_apis/build/builds/BUILD_ID/timeline?api-version=7.0" \
  -H "Authorization: Basic $(echo -n :$AZDO_PAT | base64)" | jq '.records[] | select(.result == "failed") | {name, issues}'

# Get task log (actual error output)
curl -s "https://dev.azure.com/dnceng-public/public/_apis/build/builds/BUILD_ID/logs/LOG_ID?api-version=7.0" \
  -H "Authorization: Basic $(echo -n :$AZDO_PAT | base64)" | tail -100
```

## For GitHub Actions

```bash
# Find the latest run for a PR branch
gh run list --branch BRANCH_NAME --limit 3 --json databaseId,status,conclusion,name,createdAt

# Get failed jobs from a run
gh run view RUN_ID --json jobs --jq '.jobs[] | select(.conclusion == "failure") | {name, conclusion, startedAt}'

# Get failure logs (most important command)
gh run view RUN_ID --log-failed 2>&1 | tail -200

# Get specific job log
gh run view RUN_ID --log --job JOB_ID 2>&1 | tail -100
```

## Output Format

```markdown
## CI Status for PR #XXXXX

| Job | Status | Duration |
|-----|--------|----------|
| Build | ✅ / ❌ | Nm |
| Tests | ✅ / ❌ | Nm |

### Failures
**[Job Name]**: [Error summary]
```
[Key error lines]
```

### Diagnosis
[What failed and why — extracted from logs]

### Suggested Fix
[What to change based on the error]
```

## Pipeline Names

- **Azure Pipelines**: `ML.NET Official Build` (`build/vsts-ci.yml`)
- **GitHub Actions**: `Copilot Setup Steps`, `Backport`, `Lock`
