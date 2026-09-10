# Repo health groom playbook

This playbook contains the detailed ML.NET dashboard-grooming methodology. The local execution and approval rules in [`../SKILL.md`](../SKILL.md) override any mutating command examples below.

# Repo Health — Dashboard Groomer

Maintain the health dashboard issue for **dotnet/machinelearning** — link investigation results, mark resolved findings, hide stale comments, and keep the dashboard clean.

## Configuration

| Setting | Value |
|---------|-------|
| Repository | `dotnet/machinelearning` |
| Dashboard Label | `repo-health` |

---

## Step 1 — Find Dashboard Issue

```bash
ISSUE=$(gh issue list --repo dotnet/machinelearning \
  --label "repo-health" --state open \
  --json number --jq '.[0].number')

if [ -z "$ISSUE" ]; then
  echo "No dashboard issue found. Nothing to groom."
  exit 0
fi

# Get issue body
BODY=$(gh issue view "$ISSUE" --repo dotnet/machinelearning --json body --jq '.body')

# Get all comments
COMMENTS=$(gh issue view "$ISSUE" --repo dotnet/machinelearning \
  --json comments --jq '.comments[] | {id: .databaseId, author: .author.login, createdAt: .createdAt, body: .body}')
```

---

## Step 2 — Classify Comments

Parse all comments on the dashboard issue and classify them:

### Investigation Reports

Identify by the marker: `*Investigated by repo-health-investigate • Finding:*`

Extract:
- Finding ID (from the `Finding: \`ID\`` marker)
- Root cause confidence level
- Recommendations
- Date posted

### Daily Overviews

Identify by the marker: `## 📊 Daily Delta` or similar daily summary pattern.

Extract:
- Date
- Findings summary

### Other Comments

Any comment that doesn't match the above patterns — likely manual team comments.

---

## Step 3 — Link Investigation Results into Issue Body

For each investigation report found in comments:

1. Find the corresponding finding in the dashboard issue body (match by Finding ID)
2. Update the finding's row to include a link to the investigation comment
3. If the investigation found a resolution, note it in the status column

### Update Format

In the findings tables, add investigation links:

Before:
```
| `C1-abc123` | Pipeline | CI failing on main | 🆕 NEW | 2026-03-06 |
```

After:
```
| `C1-abc123` | Pipeline | CI failing on main ([investigated](#comment-link)) | 🆕 NEW | 2026-03-06 |
```

---

## Step 4 — Mark Resolved Findings

Compare the current findings in the dashboard with the latest data:

1. Findings that are now in the "Recently Resolved" section — keep for 7 days, then remove
2. Resolved findings older than 7 days — move to a collapsed "Archive" section or remove

```markdown
<details>
<summary>📁 Archived Resolved Findings (last 30 days)</summary>

| ID | Category | Finding | Resolved |
|----|----------|---------|----------|
| ... | ... | ... | ... |

</details>
```

---

## Step 5 — Hide Stale Comments

Hide comments that are no longer useful to reduce noise. Apply in priority order, respecting the budget of 50 hide operations:

### Hide Criteria (in order)

1. **Resolved investigation reports** — Finding is now resolved AND report is > 7 days old
2. **Old daily overviews** — Daily delta comments > 7 days old (keep the last 7)
3. **Superseded investigations** — If a finding was investigated multiple times, hide older reports

### Approved Hide Operation

```bash
# Run only after the user approves this exact comment ID.
gh api graphql -f query='
  mutation {
    minimizeComment(input: {
      subjectId: "COMMENT_NODE_ID",
      classifier: OUTDATED
    }) {
      minimizedComment { isMinimized }
    }
  }
'
```

### Budget Tracking

```
Hidden: 0 / 50
- Resolved investigations: X hidden
- Old daily overviews: Y hidden
- Superseded: Z hidden
Total: X + Y + Z (must be ≤ 50)
```

---

## Step 6 — Enforce Issue Body Sanity

Before writing the updated issue body, validate:

### Length Check

```
previous_length = len(previous_body)
new_length = len(new_body)
ratio = new_length / previous_length

if ratio < 0.80:
  # Something went wrong — body shrank too much
  # Abort update and log warning
  echo "WARNING: New body is ${ratio}% of previous body. Aborting to prevent data loss."
  exit 1
fi
```

### Structure Check

Verify the updated body still contains required sections:
- `# 🏥 Repo Health Dashboard` header
- `## Summary` section
- `## Findings` section
- At least one findings table
- `## Trends` section
- Footer with local skill link

If any required section is missing, abort the update.

### Apply Approved Update

```bash
# Run only if validation passes and the user approves the exact body.
gh issue edit "$ISSUE" --repo dotnet/machinelearning --body "$UPDATED_BODY"
```

---

## Rules

1. **Dashboard-only** — Approved writes modify only the dashboard issue, never other issues or PRs
2. **Conservative** — When in doubt, don't hide a comment or remove a finding
3. **Length-preserving** — Never shrink the issue body below 80% of its previous length
4. **Budget-aware** — Max 1 issue update, max 50 comment hides
5. **Idempotent** — Running twice produces the same result
6. **Structure-preserving** — All required dashboard sections must survive grooming
