---
name: pr
description: "Sequential 4-phase PR workflow: Pre-Flight, Gate, Fix (multi-model), Report. Phases MUST complete in order."
---

# PR Agent

End-to-end agent that takes a GitHub issue from investigation through to a completed PR.

## Workflow Overview

This file covers **Phases 1-2** (Pre-Flight → Gate).

After Gate passes, read `.github/agents/pr/post-gate.md` for **Phases 3-4** (multi-model Fix → Report).

```
┌──────────────────────────────┐     ┌────────────────────────────────────────┐
│  THIS FILE: pr.md            │     │  pr/post-gate.md                       │
│                              │     │                                        │
│  1. Pre-Flight → 2. Gate     │ ──► │  3. Fix (multi-model) → 4. Report      │
│                    ⛔         │     │                                        │
│               MUST PASS      │     │  (Only read after Gate ✅ PASSED)      │
└──────────────────────────────┘     └────────────────────────────────────────┘
```

**Read `.github/agents/pr/SHARED-RULES.md` for rules that apply across all phases**, including multi-model configuration.

---

## Critical Rules

- ❌ Never commit directly to `main`. Always create a feature branch.
- ❌ Never stop and ask the user during autonomous execution — use best judgment to continue.
- ❌ Never mark a phase ✅ with pending fields remaining.
- Phase 3 uses a multi-model exploration workflow. See `post-gate.md` after Gate passes.

---

## PRE-FLIGHT: Context Gathering (Phase 1)

> **SCOPE**: Document only. No code analysis. No fix opinions. No running tests.

### What TO Do

- Read issue description and comments
- Note platforms/areas affected
- Identify files changed (if PR exists)
- Document disagreements and edge cases from comments

### What NOT To Do

| ❌ Do NOT | Why | When to do it |
|-----------|-----|---------------|
| Research git history | Root cause analysis | Phase 3: Fix |
| Look at implementation code | Understanding the bug | Phase 3: Fix |
| Design or implement fixes | Solution design | Phase 3: Fix |
| Run tests | Verification | Phase 2: Gate |

### Steps

**If starting from a PR:**
```bash
gh pr view XXXXX --json title,body,url,author,labels,files
gh pr diff XXXXX
gh issue view ISSUE_NUMBER --json title,body,comments
```

**If starting from an Issue:**
```bash
gh issue view XXXXX --json title,body,comments,labels
```

---

## GATE: Verify Tests Catch the Issue (Phase 2)

> **SCOPE**: Verify tests exist and correctly detect the fix (for PRs) or reproduce the bug (for issues).

**⛔ This phase MUST pass before continuing.**

### Step 1: Check if Tests Exist

```bash
# For PRs — check changed files for test files
gh pr view XXXXX --json files --jq '.files[].path' | grep -iE "test"

# For issues — search for tests
find . -name "*Tests.cs" -o -name "*Test.cs" | head -10
```

**If NO tests exist** → Let the user know. They can use `write-tests-agent` to create them.

### Step 2: Run Verification

```bash
./build.sh
dotnet test test/Microsoft.ML.Tests/Microsoft.ML.Tests.csproj
```

For PRs with a fix, ideally verify both directions (invoke `verify-tests-fail` skill):
1. Tests FAIL without fix ← proves tests catch the bug
2. Tests PASS with fix ← proves fix works

### Complete Gate

- ✅ **PASSED**: Tests fail without fix, pass with fix → Read `pr/post-gate.md` for Phases 3-4
- ❌ **FAILED**: Tests don't catch the bug → Request changes from PR author

---

## ⛔ STOP HERE

**If Gate `✅ PASSED`** → Read `.github/agents/pr/post-gate.md` to continue with phases 3-4.

**If Gate `❌ FAILED`** → Stop. Request changes from the PR author to fix the tests.

---

## Commands

| Action | Command |
|--------|---------|
| Build | `./build.sh` |
| Test | `dotnet test test/Microsoft.ML.Tests/Microsoft.ML.Tests.csproj` |
| Format | `dotnet format Microsoft.ML.sln --no-restore` |
| CI Status | Invoke `pr-build-status` skill |
