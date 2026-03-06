---
name: learn-from-pr
description: Analyzes a completed PR to extract lessons learned from agent behavior. Use after any PR with agent involvement to identify what worked, what failed, and what to improve in instruction files, skills, or documentation.
---

# Learn From PR

Extracts lessons from completed PRs and produces actionable recommendations.

## Workflow

### 1. Gather data

```bash
gh pr view XXXXX --json title,body,files,commits,comments,reviews
gh pr diff XXXXX
```

### 2. Analyze

- **Fix location**: Which files were changed? Which module/layer?
- **Failure modes** (if agent struggled): Wrong files targeted? Missing domain knowledge? Bad test command?
- **What worked**: What led to the successful fix?

### 3. Generate recommendations

Each recommendation:

| Field | Description |
|-------|-------------|
| Priority | high / medium / low |
| Category | instruction-file, skill, code-comment, documentation |
| Location | Which file to update |
| Change | Specific text to add/modify |
| Why | How this prevents future failures |

### 4. Present to user

```markdown
## Lessons from PR #XXXXX

### What Happened
[Problem → Attempts → Solution]

### Recommendations
| # | Priority | Where | What to Change |
|---|----------|-------|----------------|
| 1 | High | .github/instructions/X.md | Add guidance about Y |
```

The learn-from-pr **agent** (separate from this skill) takes it further by actually applying the recommendations to files.
