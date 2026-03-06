---
name: find-reviewable-pr
description: Finds open PRs that are good candidates for review, prioritizing by milestone, priority labels, and community status. Use when asked to "find PRs to review", "what needs review", or "show me open PRs".
---

# Find Reviewable PR

Searches for open PRs prioritized by importance.

## Priority Order

1. 🔴 **P/0** — Critical priority, review first
2. ✅ **Approved (not merged)** — Ready to merge
3. 📅 **Milestoned** — Has a deadline
4. ✨ **Community** — External contributions
5. 🕐 **Recent** — Created in last 2 weeks, no review yet

## Commands

```bash
# Priority PRs
gh pr list --search "label:p/0" --json number,title,author,labels,createdAt

# Milestoned
gh pr list --search "milestone:*" --json number,title,author,milestone,createdAt

# Recent needing review
gh pr list --search "review:none created:>=$(date -v-14d +%Y-%m-%d)" --json number,title,author,createdAt
```

## Output

Group results by category. Include: PR number, title, author, age, complexity (Easy/Medium/Complex based on file count and additions).
