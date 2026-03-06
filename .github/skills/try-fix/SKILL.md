---
name: try-fix
description: "Attempts ONE alternative fix for a bug, tests it empirically, and reports results. Always explores a DIFFERENT approach from existing fixes."
---

# Try Fix

Single-shot: receive context → try ONE fix → test → report → revert.

## Principles
1. **Single-shot** — One fix idea per invocation
2. **Alternative** — Always different from existing fixes
3. **Empirical** — Implement and test, don't theorize
4. **Clean** — Always revert after, leave repo clean

## Workflow

### 1. Baseline
```bash
git stash
dotnet test test/Microsoft.ML.Tests/Microsoft.ML.Tests.csproj  # Confirm failure
```

### 2. Implement one fix
Minimal changes. Read prior attempts (if any) and do something different.

### 3. Test
```bash
./build.sh
dotnet test test/Microsoft.ML.Tests/Microsoft.ML.Tests.csproj
```

### 4. Report
```markdown
## Try-Fix Attempt #N
**Approach**: [what and why]
**Changes**: `path/file` — [what changed]
**Build**: ✅/❌
**Tests**: ✅/❌
**Verdict**: ✅ FIX WORKS / ❌ FAILED — [reason]
```

### 5. Revert
```bash
git checkout -- .
git stash pop
```

## Rules
- Sequential only — never parallel
- Max 5 attempts per session
- Always revert — leave repo clean
