---
name: repo-health-groom
description: Groom the dotnet/machinelearning repo-health dashboard locally by linking investigation results, marking resolved findings, archiving stale entries, and drafting conservative comment minimization. Use when asked to clean, update, or maintain the ML.NET health dashboard.
---

# ML.NET repository health groom

Prepare a safe, idempotent dashboard cleanup from a local Copilot CLI session.

## Workflow

1. Find the open `dotnet/machinelearning` issue labeled `repo-health`.
2. Read [`references/playbook.md`](references/playbook.md).
3. Classify investigation reports, daily overviews, and manual comments.
4. Draft the updated dashboard body and the ordered list of comments eligible for minimization.
5. Enforce the 80 percent length floor and all required dashboard sections.
6. Show the body diff and comment IDs before any mutation.
7. Update the issue or minimize comments only after explicit approval.

## Rules

- Modify only the dashboard issue.
- Keep manual team comments visible.
- Minimize at most 50 comments and prefer conservative decisions.
- Running twice against the same state must produce the same result.
- Never shrink or rewrite the body when validation fails.
