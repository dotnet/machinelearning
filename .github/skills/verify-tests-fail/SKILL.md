---
name: verify-tests-fail-without-fix
description: "Verifies tests catch the bug — fail without fix, pass with fix. Use after writing tests for a bug fix, or when asked to prove tests are valid."
---

# Verify Tests Fail Without Fix

Proves tests actually catch the bug.

## Full Verification

```bash
# 1. Remove the fix, keep the tests
git stash push -m "fix" -- <FIX_FILES>

# 2. Build and run — should FAIL
./build.sh
dotnet test test/Microsoft.ML.Tests/Microsoft.ML.Tests.csproj --filter "FullyQualifiedName~RelevantTests"
# Expected: ❌ FAIL (proves tests catch the bug)

# 3. Restore fix
git stash pop

# 4. Build and run — should PASS
./build.sh
dotnet test test/Microsoft.ML.Tests/Microsoft.ML.Tests.csproj --filter "FullyQualifiedName~RelevantTests"
# Expected: ✅ PASS (proves fix works)
```

## Output

```markdown
## Verification

| State | Build | Tests | Expected |
|-------|-------|-------|----------|
| Without fix | ✅/❌ | ❌ FAIL | ❌ (good — catches bug) |
| With fix | ✅ | ✅ PASS | ✅ (good — fix works) |

**Verdict**: ✅ Tests properly validate the fix / ❌ Tests don't catch the bug
```

## Rules
- Always restore working state after verification
- If tests pass without the fix → they don't catch the bug, report this
- If tests fail with the fix → fix is incomplete, report this
