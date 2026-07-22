---
name: repo-health-check
description: Run the dotnet/machinelearning repository health check locally, covering issue backlog, pull requests, GitHub Actions, and Azure DevOps pipelines, then draft a dashboard update and prioritized investigations. Use when asked for ML.NET repository health, maintenance status, stale work, or CI trends.
---

# ML.NET repository health check

Collect and analyze repository-health data from a local Copilot CLI session. This skill replaces the scheduled orchestrator for now.

## Workflow

1. Confirm local GitHub access and the `dotnet/machinelearning` repository.
2. Create state under `/tmp/mlnet-repo-health/`.
3. Read [`references/playbook.md`](references/playbook.md) and follow its data collection, fingerprinting, severity, and dashboard-format rules.
4. Prefer available authenticated GitHub and Azure DevOps tools. If those are unavailable, use the public `dnceng-public/public` REST API; skip AzDO checks only when both access paths fail, and state the coverage gap.
5. Treat dashboard creation, body updates, comments, and investigation requests as drafts during analysis.
6. For up to five critical or high-confidence warning findings, either:
   - investigate inline by following [`../repo-health-investigate/SKILL.md`](../repo-health-investigate/SKILL.md), or
   - produce a prioritized investigation queue when the user requested check-only mode.
7. Present the proposed dashboard body, daily delta, and any investigation reports.
8. Create or update the dashboard and post comments only after explicit approval for the exact writes.

## Local substitutions

- Store local state under `/tmp/mlnet-repo-health/`.
- Perform investigations locally rather than dispatching a workflow.
- Use the authenticated local user instead of workflow tokens.
- Preserve the one-dashboard-update, one-delta-comment, and five-investigation budgets.

## Validation

- Every finding has a deterministic fingerprint and status.
- Baselined findings are not marked new.
- Required dashboard sections remain present.
- No source issue or PR is modified.
