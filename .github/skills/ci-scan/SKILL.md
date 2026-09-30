---
name: ci-scan
description: Analyze recent dotnet/machinelearning MachineLearning-CI failures locally, identify stable failure signatures, deduplicate them against Known Build Error issues, and draft up to three actionable KBE issues. Use when asked to scan ML.NET CI, investigate recurring main-branch failures, or run the former ci-scan agent locally.
---

# ML.NET CI scan

Run the ML.NET CI failure scanner from a local Copilot CLI session. This skill replaces scheduled GitHub Actions execution for now.

## Inputs

| Input | Required | Description |
|---|---|---|
| Build ID | No | Specific AzDO build to analyze. Otherwise select the source build using the playbook. |
| Apply approved drafts | No | Defaults to false. GitHub mutations require explicit approval in the current conversation. |

## Workflow

1. Confirm the working repository is `dotnet/machinelearning`.
2. Verify local GitHub access with `gh auth status`. Use anonymous `dnceng-public/public` AzDO and Helix endpoints from the references.
3. Create local state under `/tmp/mlnet-ci-scan/`.
4. Read [`references/playbook.md`](references/playbook.md) and [`references/ci-scan.instructions.md`](references/ci-scan.instructions.md).
5. Follow the playbook once for the selected build window.
6. Treat every issue-writing instruction as a request to prepare a draft. Do not create issues, add labels, or post comments while analyzing.
7. Present each proposed issue with its exact title, body, labels, evidence links, and dedup result. Cap the run at three drafts.
8. Apply drafts only after the user explicitly approves the exact GitHub writes. Never close or modify unrelated issues.

## Local substitutions

- Store all intermediate state under `/tmp/mlnet-ci-scan/`.
- Replace `aw_<id>` placeholders with `draft-<n>` until an issue is created.
- Report no-op and skip outcomes directly in chat. Do not create status issues.
- Use the authenticated local user rather than `COPILOT_GITHUB_TOKEN` or a PAT pool.

## Validation

- Every failure signature has exactly one outcome.
- Every draft passes the occurrence, follow-up, specificity, sanitization, match-count, and dedup gates.
- The final response includes the full tally table and all skipped reasons.
