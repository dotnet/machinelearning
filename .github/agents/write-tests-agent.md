---
name: write-tests-agent
description: "Determines what type of tests to write and creates them following repo conventions."
---

# Write Tests Agent

Determines what tests are needed, finds the right test project, and writes tests following existing conventions.

## When to Use
- "Write tests for issue #XXXXX"
- "Add test coverage for..."

## Workflow

### 1. Determine test type
| Scenario | Type |
|----------|------|
| Data pipeline behavior | Unit test with `TestDataPipeBase` |
| Trainer functionality | Unit test with model training |
| API surface | Unit test verifying public API |
| End-to-end scenarios | Integration test |
| Tokenizer behavior | Unit test with known token sequences |

### 2. Find test project and conventions

```bash
# List test projects
find . -name "*Tests.csproj" | head -20

# Read existing test patterns
head -50 $(find . -name "*Tests.cs" | head -3)
```

Mirror source project naming: `Microsoft.ML.Foo` → `test/Microsoft.ML.Foo.Tests/`

### 3. Write tests following repo conventions

Test framework: **xUnit**

All tests must:
- Inherit from `TestDataPipeBase` (or `BaseTestClass` for simpler tests)
- Accept `ITestOutputHelper output` in constructor and pass to base
- Use PascalCase descriptive method names
- Include the 3-line .NET Foundation MIT license header
- Use `[Fact]` for single tests, `[Theory]` with `[InlineData]` for parameterized tests
- Use `Assert.*` (xUnit) or `AwesomeAssertions` for assertions

Test projects: `test/Microsoft.ML.Tests/`, `test/Microsoft.ML.Core.Tests/`, `test/Microsoft.ML.Tokenizers.Tests/`, `test/Microsoft.ML.GenAI.Core.Tests/`, and 30+ others

### 4. Run tests

```bash
dotnet test test/Microsoft.ML.Tests/Microsoft.ML.Tests.csproj --filter "FullyQualifiedName~NewTestClassName"
```

### 5. Verify tests catch the bug
If testing a fix: tests should fail without fix, pass with fix. Use verify-tests-fail skill.
