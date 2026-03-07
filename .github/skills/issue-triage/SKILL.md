---
name: issue-triage
description: Queries and triages open GitHub issues that need attention. Helps identify issues needing milestones, labels, or investigation. Use when asked to "triage issues", "find issues without milestones", or "what needs attention".
---

# Issue Triage

Present issues ONE AT A TIME for human triage decisions.

## Workflow

### 1. Query issues without milestones

```bash
gh issue list --repo OWNER/REPO \
  --search "no:milestone -label:needs-info -label:needs-repro" \
  --limit 50 --json number,title,labels,createdAt,author,comments,url
```

### 2. Present one issue

```markdown
## Issue #XXXXX — [Title]
🔗 [URL]

| Field | Value |
|-------|-------|
| Author | username |
| Labels | labels |
| Age | N days |
| Comments | N |

**Suggestion**: `Milestone` — Reason
```

### 3. Wait for user decision
- Milestone name → apply it
- "skip" → next issue
- "yes" → accept suggestion

### 4. Apply and move to next

```bash
gh issue edit NUMBER --repo OWNER/REPO --milestone "MILESTONE"
```

Auto-reload when batch is exhausted. Don't ask "Load more?" — just do it.
