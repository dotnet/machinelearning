---
name: pr-finalize
description: Finalizes any PR for merge by verifying title/description match implementation AND performing code review. Use when asked to "finalize PR", "check PR description", "review commit message", before merging any PR.
---

# PR Finalize

Verifies PR title and description accurately reflect the implementation, then reviews code for best practices.

## Rules

- **NEVER** use `gh pr review --approve` or `--request-changes` — approval is a human decision
- **NEVER** post comments directly — this skill is analysis only. Use ai-summary-comment to post.

## Workflow

### Phase 1: Title & Description

```bash
gh pr view XXXXX --json title,body,files,commits
gh pr diff XXXXX
```

1. **Evaluate existing description** — Is it good? Don't replace quality with a template.
2. **Check title** — Should describe behavior, not just "Fix #123". Format: `[Scope] What changed`
3. **Check description** — Should explain what changed and why, link to issues, note breaking changes.

### Phase 2: Code Review

Focus on: code quality, error handling, performance, breaking changes, test coverage.

### Output

```markdown
## PR #XXXXX Review

### Title: ✅ Good / ⚠️ Needs Update
**Current**: "existing"
**Suggested**: "improved" (if needed)

### Description: ✅ Good / ⚠️ Needs Update

### Code Review
#### 🔴 Critical: [issue in path/to/file]
#### 🟡 Suggestion: [improvement]
#### ✅ Looks Good: [positive observation]
```
