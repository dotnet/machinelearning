---
name: ai-summary-comment
description: Posts or updates automated progress comments on GitHub PRs. Creates single unified comment with collapsible sections. Use after completing any agent phase to post results, or when asked to "post comment to PR", "update PR progress".
---

# AI Summary Comment

Posts a single unified comment on a PR with collapsible sections for each phase of work.

## Architecture

One comment per PR, identified by `<!-- AI Summary -->`. Each update modifies only its section.

```markdown
<!-- AI Summary -->
## 🤖 AI Summary

<!-- SECTION:REVIEW -->
<details><summary>📋 PR Review — ✅ Complete</summary>
[Review findings]
</details>
<!-- /SECTION:REVIEW -->

<!-- SECTION:TESTS -->
<details><summary>🧪 Tests — ✅ Pass</summary>
[Test results]
</details>
<!-- /SECTION:TESTS -->
```

## Usage

```bash
# Find existing comment
COMMENT_ID=$(gh pr view PR --json comments --jq '.comments[] | select(.body | contains("<!-- AI Summary -->")) | .databaseId')

# Update or create
if [ -n "$COMMENT_ID" ]; then
  gh api repos/OWNER/REPO/issues/comments/$COMMENT_ID --method PATCH -f body="$BODY"
else
  gh pr comment PR --body "$BODY"
fi
```

## Rules

1. **Self-contained** — Never reference local files in comments
2. **Idempotent** — Running twice produces same result
3. **Section isolation** — Updating one section preserves others
4. **Collapsible** — Use `<details>` tags to keep comments compact
5. **No approvals** — Never use `--approve` or `--request-changes`
