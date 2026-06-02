---
description: |
  Runs every 2 days. Sweeps open issues labeled `need info`. Removes the
  label when anyone other than the labeler has commented since the label
  was applied. Nudges silent issues at 14 days. Closes silent issues at
  30 days, but only if a previous nudge from this workflow is on the
  thread. Fixed wording, capped output.

on:
  schedule: every 2d
  workflow_dispatch:
  roles: [admin, maintain, write]

if: github.repository == 'dotnet/machinelearning'

timeout-minutes: 30

permissions: read-all

concurrency:
  group: needs-info-sweeper
  cancel-in-progress: false

network:
  allowed:
    - defaults
    - github

tools:
  github:
    toolsets: [repos, issues, search]
  bash: ["git", "find", "ls", "cat", "grep", "head", "tail", "wc", "jq", "tee", "sed", "awk", "tr", "cut", "sort", "uniq", "xargs", "echo", "date", "test", "gh"]

safe-outputs:
  noop:
    report-as-issue: false
  remove-labels:
    allowed: ["need info"]
    max: 30
  add-comment:
    target: "*"
    max: 50
    hide-older-comments: false
  close-issue:
    max: 15
---

# Needs-Info Sweeper (machinelearning)

Sweep open issues labeled `need info`. For each issue, decide one of:
- **reply received** (anyone other than the labeler commented since the label was applied) -> remove `need info`.
- **silent 14d** -> post the nudge comment.
- **silent 30d AND previously nudged by this workflow at least 14 days ago** -> post the close comment, then close the issue.
- **noop** otherwise.

## Hard rules

1. **Only `need info`-labeled open issues.** Never touch other labels.
2. **Fixed wording.** Use the exact comment text below; do not paraphrase. Do not personalize.
3. **Caps per run: 30 nudges, 15 closes, 30 label removals. Total comment budget = 50 (nudges + closes).** On any cap, stop that action and continue the others.
4. **Idempotency.** Every comment includes `<!-- needs-info-sweeper:<event> -->` where `<event>` is `nudge` or `close`. Skip a comment if the most recent bot comment on the issue carries the same marker.
5. **Reply check definition.** "Reply received" means at least one comment on the issue, created after `label_applied_at`, from a user whose login is not the labeler and is not a bot (`[bot]` suffix). Bot comments and the labeler's own comments do not count as replies.
6. **Skip protected labels: `bug`, `Known Build Error`, `blocking-clean-ci`, `needs-author-action`.** These deserve different treatment.
7. **Skip issues whose `need info` label was applied less than 14 days ago.** The clock starts at the label event.
8. **Never close without prior nudge from this workflow.** If `age >= 30d` but no prior bot comment carries `<!-- needs-info-sweeper:nudge -->`, downgrade to a nudge (post the nudge comment, do NOT close). Closure requires the issue to have been nudged at least 14 days earlier by this workflow, i.e. closure happens at `age >= 30d` AND `nudged_at` exists AND `(now - nudged_at) >= 14d`.

## Process

For each open issue with label `need info`:

1. Determine `label_applied_at`:
   ```bash
   gh api "/repos/dotnet/machinelearning/issues/<N>/timeline" --paginate \
     --jq '[.[] | select(.event == "labeled" and .label.name == "need info")] | last | .created_at'
   ```
2. **Reply check (rule 5).** Find comments created after `label_applied_at`:
   ```bash
   gh api "/repos/dotnet/machinelearning/issues/<N>/comments" --paginate \
     --jq '[.[] | select(.created_at > "<label_applied_at>") | .user.login]'
   ```
   If any login in the result is **not** the labeler AND does **not** end in `[bot]`: remove `need info`, stop.
3. **Age check.** `age = now - label_applied_at`.
4. **Locate last bot marker.** Find the most recent bot comment carrying `<!-- needs-info-sweeper:nudge -->`; record its `created_at` as `nudged_at` (or null if absent).
5. If `14d <= age < 30d` AND `nudged_at` is null: post the nudge comment.
6. If `age >= 30d`:
   - If `nudged_at` is null: post the nudge comment (rule 8 downgrade). Do not close.
   - If `nudged_at` is non-null AND `(now - nudged_at) >= 14d`: post the close comment, then close as `not planned`.
   - If `nudged_at` is non-null AND `(now - nudged_at) < 14d`: skip (give the nudge time to land).
7. Otherwise: skip.

## Fixed wording

**Nudge (day 14):**

```
<!-- needs-info-sweeper:nudge -->
Friendly nudge. Please share the missing details (minimal repro, environment, exact error) so we can investigate. This issue will be closed if there is no response in 14 days.

Posted by [`needs-info-sweeper`](https://github.com/dotnet/machinelearning/blob/main/.github/workflows/needs-info-sweeper.agent.md).
```

**Close (≥14 days after the nudge, and ≥30 days after the label was applied):**

```
<!-- needs-info-sweeper:close -->
Closing for inactivity. Reopen with the requested details and we will take another look.

Posted by [`needs-info-sweeper`](https://github.com/dotnet/machinelearning/blob/main/.github/workflows/needs-info-sweeper.agent.md).
```

Note: the `close_issue` safe-output transitions the issue to `closed` (state `completed`). Setting the close-reason to `not planned` is not supported by the safe-outputs schema; that is fine, the close comment makes the inactivity reason explicit.

## Tally

Append per-issue outcome to `/tmp/gh-aw/agent/sweep.txt`:

```
<issue#>  <outcome>
```

Outcomes: `reply-received`, `nudged`, `closed`, `skipped:<reason>`.

At end of run, print this table to the agent log:

```
| total-need-info | reply-received | nudged | closed | skipped |
```
