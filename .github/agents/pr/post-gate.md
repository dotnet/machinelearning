# PR Agent: Post-Gate Phases (3-4)

**⚠️ PREREQUISITE: Only read this file after 🚦 Gate shows `✅ PASSED`.**

If Gate is not passed, go back to `.github/agents/pr.md` and complete phases 1-2 first.

---

## Workflow Overview

| Phase | Name | What Happens |
|-------|------|--------------|
| 3 | **Fix** | Invoke `try-fix` skill with multiple models to explore independent fix alternatives, then compare with PR's fix |
| 4 | **Report** | Deliver result (approve PR, request changes, or create new PR) |

**All rules from `.github/agents/pr/SHARED-RULES.md` apply**, including multi-model configuration.

---

## 🔧 FIX: Multi-Model Exploration (Phase 3)

> **SCOPE**: Explore independent fix alternatives using `try-fix` skill across multiple AI models, compare with PR's fix, select the best approach.

### Why Multi-Model?

Each AI model has different strengths — one may spot a root cause another misses, or propose a simpler fix. By running try-fix with 3 models sequentially, you maximize fix diversity and increase the chance of finding the optimal solution.

### 🚨 CRITICAL: try-fix is Independent of PR's Fix

**The PR's fix has already been validated by Gate.** Phase 3 is NOT re-testing the PR's fix — it's exploring whether a better alternative exists.

**Do NOT let the PR's fix influence your thinking.** Generate ideas as if you hadn't seen the PR.

### Step 1: Run try-fix with Each Model (Round 1)

Run the `try-fix` skill **3 times sequentially**, once with each model (see `SHARED-RULES.md` for model list).

**⚠️ SEQUENTIAL ONLY**: try-fix runs modify the same files and use the same build/test environment. Never run in parallel.

**For each model**, invoke as a task agent with the specified model:

```
Invoke the try-fix skill for PR #XXXXX:
- problem: [Description of the bug — what's broken and expected behavior]
- test_command: dotnet test test/Microsoft.ML.Tests/Microsoft.ML.Tests.csproj
- target_files: [files likely affected]

Generate ONE independent fix idea. Review the PR's fix first to ensure your approach is DIFFERENT.
```

**Wait for each to complete before starting the next.**

**🧹 MANDATORY: Clean up between attempts.** After each try-fix completes (pass or fail):

```bash
# Restore all tracked files to HEAD
git checkout HEAD -- .

# Remove untracked files added by the previous attempt
git clean -fd
```

### Step 2: Cross-Pollination (Round 2+)

After Round 1, share each model's results with the others and ask for new ideas.

**For each model**, invoke again with this context:

```
Here are the fix attempts from Round 1:
[List each model's approach and result]

Given what worked and what didn't, propose a NEW fix idea that:
- Is DIFFERENT from all attempts above
- Learns from the failures (avoid the same mistakes)
- Combines insights from passing fixes if applicable

If you genuinely have no new idea, respond "NO NEW IDEAS" — don't force a bad attempt.
```

**Exhaustion criteria**: Cross-pollination is exhausted when ALL models respond "NO NEW IDEAS" via actual invocation (not assumed).

### Step 3: Select Best Fix

Build a comparison table of all candidates:

```markdown
### Fix Candidates
| # | Model | Approach | Result | Files Changed | Notes |
|---|-------|----------|--------|---------------|-------|
| 1 | claude-sonnet-4-5 | [approach] | ✅/❌ | `file.cs` | [why] |
| 2 | gpt-4.1 | [approach] | ✅/❌ | `file.cs` | [why] |
| PR | PR author | [approach] | ✅ (Gate) | `file.cs` | Original |
```

**Selection criteria** (in order):
1. Tests pass
2. Minimal changes (fewer files, fewer lines)
3. Root cause fix (not symptom suppression)
4. Code quality and maintainability

---

## 📋 REPORT: Deliver Result (Phase 4)

### If Starting from PR — Write Review

| Scenario | Recommendation |
|----------|---------------|
| PR's fix was selected | ✅ **APPROVE** — PR's approach is correct/optimal |
| Alternative fix was better | ⚠️ **REQUEST CHANGES** — suggest the better approach |
| PR's fix failed tests | ⚠️ **REQUEST CHANGES** — fix doesn't work |

Run `pr-finalize` skill to verify PR title/description match implementation.

### If Starting from Issue — Create PR

Present the selected fix to the user:

```markdown
I've implemented the fix for issue #XXXXX:
- **Selected fix**: Candidate #N — [approach]
- **Files changed**: [list]
- **Other candidates considered**: [brief summary]

Please review the changes and create a PR when ready.
```

### Report Format

```markdown
## Final Recommendation: APPROVE / REQUEST CHANGES

### Summary
[Brief summary of the review]

### Fix Exploration
[How many models tried, how many passed, which was selected and why]

### Root Cause
[Root cause analysis]

### Fix Quality
[Assessment of the selected fix]
```

---

## Common Mistakes

- ❌ **Looking at PR's fix before generating ideas** — Generate independently first
- ❌ **Re-testing the PR's fix in try-fix** — Gate already validated it
- ❌ **Skipping models in Round 1** — All models must run before cross-pollination
- ❌ **Running try-fix in parallel** — SEQUENTIAL ONLY
- ❌ **Declaring exhaustion prematurely** — All models must confirm "NO NEW IDEAS"
- ❌ **Not cleaning up between attempts** — Always restore working directory
- ❌ **Selecting a failing fix** — Only select from passing candidates
