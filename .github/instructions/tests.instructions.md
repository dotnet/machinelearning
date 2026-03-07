---
applyTo:
  - "test/**"
  - "**/*Tests.cs"
  - "**/*Test.cs"
---

# Test Guidelines for ML.NET

## Framework: xUnit

### Base Class

All test classes inherit from `TestDataPipeBase` (or `BaseTestClass` for simpler tests):

```csharp
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public class MyFeatureTests : TestDataPipeBase
    {
        public MyFeatureTests(ITestOutputHelper output) : base(output) { }

        [Fact]
        public void MyFeatureBasicTest()
        {
            // ...
        }
    }
}
```

### Naming

- Test classes: `{Feature}Tests` (e.g., `AnomalyDetectionTests`, `CachingTests`)
- Test methods: PascalCase descriptive names (e.g., `RandomizedPcaTrainerBaselineTest`)
- No `Test_` prefix or `_Should_` patterns — use direct descriptive names

### Assertions

- Primary: `Assert.*` (xUnit) — `Assert.Equal`, `Assert.Throws<T>`, `Assert.Contains`, `Assert.True`
- Fluent: `AwesomeAssertions` is available for more expressive assertions
- Never use `Assert.That` (NUnit style) — this is an xUnit repo

### Test Data

- Use `Microsoft.ML.TestDatabases` package for standard datasets
- Test-specific data goes in `test/data/`
- Reference data path via `GetDataPath("filename")` from the base class

### Baseline Testing

- Expected output files in `test/BaselineOutput/`
- Use `BaseTestClass` methods to compare actual vs baseline output
- Update baselines carefully — they are the source of truth for output format stability

### Running Tests

```bash
# Specific test project
dotnet test test/Microsoft.ML.Tests/Microsoft.ML.Tests.csproj

# Filter by test name
dotnet test test/Microsoft.ML.Tests/Microsoft.ML.Tests.csproj --filter "FullyQualifiedName~AnomalyDetectionTests"

# Filter by single test
dotnet test test/Microsoft.ML.Tests/Microsoft.ML.Tests.csproj --filter "FullyQualifiedName~RandomizedPcaTrainerBaselineTest"
```

### Target Frameworks

Tests multi-target `net8.0;net48;net9.0` on Windows, `net8.0` only on Linux/macOS/arm64. Make sure tests pass on all targeted frameworks.

### Common Gotchas

- **Locale**: Base class pins `Thread.CurrentThread.CurrentCulture` to `en-US` — don't change it
- **License header**: Every `.cs` file needs the 3-line .NET Foundation MIT header
- **Unsafe code**: `AllowUnsafeBlocks` is enabled in test projects — this is intentional for testing native interop
- **XML doc warnings**: Suppressed (CS1573, CS1591, CS1712) in test code — no need to add XML docs to tests
