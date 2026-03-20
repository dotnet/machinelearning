---
description: "Guidance for GitHub Copilot when working on ML.NET (dotnet/machinelearning). Use for any task in this repo: code changes, test writing, PR reviews, issue investigation, build troubleshooting, or documentation."
---

# ML.NET Development Guide

## Repository Overview

ML.NET is a cross-platform, open-source machine learning framework for .NET. It provides APIs for training, evaluating, and deploying ML models across classification, regression, clustering, ranking, anomaly detection, time series, recommendation, and generative AI (LLaMA, Phi, Mistral via TorchSharp).

### Key Technologies

- .NET SDK 10.0.100 (see `global.json`)
- Build system: Microsoft Arcade SDK (`eng/common/`)
- Test framework: xUnit (with `AwesomeAssertions`, `Xunit.Combinatorial`)
- Native dependencies: MKL, OpenMP, libmf, oneDNN
- Major dependencies: TorchSharp, ONNX Runtime, TensorFlow, LightGBM, Semantic Kernel
- Central package management: `Directory.Packages.props`

## Build & Test

### Build

```bash
# Linux/macOS
./build.sh

# Windows
build.cmd

# Build specific project
dotnet build src/Microsoft.ML.Core/Microsoft.ML.Core.csproj
```

The repo uses Arcade SDK. `build.sh`/`build.cmd` wraps `eng/common/build.sh`/`eng/common/build.ps1` with `--restore --build`. On Linux, native dependencies require `eng/common/native/install-dependencies.sh`.

### Test

```bash
# Run tests for a specific project
dotnet test test/Microsoft.ML.Tests/Microsoft.ML.Tests.csproj

# Run tests with filter
dotnet test test/Microsoft.ML.Tests/Microsoft.ML.Tests.csproj --filter "FullyQualifiedName~ClassName.MethodName"

# Run all tests (slow, prefer specific projects)
dotnet test Microsoft.ML.sln
```

Test projects multi-target `net8.0;net48;net9.0` on Windows, `net8.0` only on Linux/macOS/arm64.

### Format

```bash
dotnet format Microsoft.ML.sln --no-restore
```

The repo has `.editorconfig` and `EnforceCodeStyleInBuild=true`.

## Project Structure

```
src/
├── Microsoft.ML.Core/              # Core types, contracts, host environment
├── Microsoft.ML.Data/              # Data pipeline, DataView, schema
├── Microsoft.ML/                   # MLContext, public API surface
├── Microsoft.ML.StandardTrainers/  # Built-in trainers (logistic regression, SVM, etc.)
├── Microsoft.ML.Transforms/        # Data transforms (normalize, featurize, etc.)
├── Microsoft.ML.AutoML/            # Automated ML pipeline selection
├── Microsoft.ML.FastTree/          # Tree-based trainers
├── Microsoft.ML.LightGbm/          # LightGBM integration
├── Microsoft.ML.Recommender/       # Matrix factorization recommenders
├── Microsoft.ML.TimeSeries/        # Time series analysis
├── Microsoft.ML.Tokenizers/        # BPE/WordPiece/SentencePiece tokenizers
├── Microsoft.ML.GenAI.Core/        # GenAI base types (CausalLM pipeline)
├── Microsoft.ML.GenAI.LLaMA/       # LLaMA model support
├── Microsoft.ML.GenAI.Phi/         # Phi model support
├── Microsoft.ML.GenAI.Mistral/     # Mistral model support
├── Microsoft.ML.TorchSharp/        # TorchSharp-based trainers
├── Microsoft.ML.OnnxTransformer/   # ONNX model inference
├── Microsoft.ML.TensorFlow/        # TensorFlow model inference
├── Microsoft.ML.Vision/            # Image classification
├── Microsoft.ML.ImageAnalytics/    # Image transforms
├── Microsoft.ML.CpuMath/           # SIMD-optimized math operations
├── Microsoft.Data.Analysis/        # DataFrame API
├── Native/                         # C/C++ native library sources
└── Common/                         # Shared internal code
test/
├── Microsoft.ML.TestFramework/      # Base test classes and helpers
├── Microsoft.ML.TestFrameworkCommon/ # Shared test utilities
├── Microsoft.ML.Tests/              # Main functional tests
├── Microsoft.ML.Core.Tests/         # Core unit tests
├── Microsoft.ML.IntegrationTests/   # End-to-end integration tests
├── Microsoft.ML.Tokenizers.Tests/   # Tokenizer tests
├── Microsoft.ML.GenAI.*.Tests/      # GenAI component tests
└── ... (30+ test projects)
```

## Conventions

### Code Style

Every `.cs` file starts with the 3-line .NET Foundation MIT license header. This is enforced across the codebase and must not be omitted.

Namespaces match assembly name (`Microsoft.ML`, `Microsoft.ML.Data`, `Microsoft.ML.Trainers`). Order usings as `System.*` first, then `Microsoft.*`, then others.

Use `[BestFriend]` attribute for internal members shared across assemblies. The repo has many assemblies that need to share types without making them public; `[BestFriend]` provides controlled cross-assembly visibility for this.

Use `Contracts.Check*` / `Contracts.Except*` for argument and state validation rather than raw `throw` statements. This ensures consistent error messages and lets the ML.NET host environment intercept validation failures.

XML docs with `<summary>` tags are required on all public types and members.

When editing an existing file, match its style even if it differs from general guidelines. Consistency within a file matters more than global uniformity.

Follow [dotnet/runtime coding-style](https://github.com/dotnet/runtime/blob/main/docs/coding-guidelines/coding-style.md).

### Test Conventions

Framework: xUnit (`[Fact]`, `[Theory]`, `[InlineData]`).

Inherit from `TestDataPipeBase` (for data pipeline tests) or `BaseTestClass` (for simpler tests). Both provide `ITestOutputHelper`, test data paths, and locale pinning to `en-US`.

```csharp
public class MyFeatureTests : TestDataPipeBase
{
    public MyFeatureTests(ITestOutputHelper output) : base(output) { }

    [Fact]
    public void MyFeatureBasicTest()
    {
        // ...
    }
}
```

Name test classes as `{Feature}Tests`, test methods as PascalCase descriptive names (e.g., `RandomizedPcaTrainerBaselineTest`). Do not use `Test_` prefixes or `_Should_` patterns.

Use `Assert.*` (xUnit) or `AwesomeAssertions` for fluent assertions. Do not use `Assert.That` (NUnit style).

Test data: use `Microsoft.ML.TestDatabases` package or files in `test/data/`, referenced via `GetDataPath("filename")` from the base class. Baseline output comparison uses files in `test/BaselineOutput/`. Update baselines carefully since they are the source of truth for output format stability.

Gotchas: the base class pins locale to `en-US` (don't override). `AllowUnsafeBlocks` is enabled in test projects for native interop testing. XML doc warnings (CS1573, CS1591, CS1712) are suppressed in test code.

### Architecture

`MLContext` is the main entry point, exposing catalogs for each ML task (classification, regression, etc.).

Data flows through `IDataView`, a lazy, columnar, cursor-based data pipeline. This design avoids loading entire datasets into memory, which matters for ML workloads.

Trainers implement the `IEstimator<T>` to `ITransformer` pattern: call `Fit()` to train, then `Transform()` to apply. New trainers go in their own project under `src/`. New test projects mirror source naming: `Microsoft.ML.Foo` to `Microsoft.ML.Foo.Tests`.

## Git Workflow

- Default branch: `main`
- Never commit directly to `main`, always create a feature branch
- Branch naming: `feature/description`, `fix/description`
- PRs are squash-merged
- Reference a filed issue in PR description
- Address review feedback in additional commits (don't amend/force-push)
- Use `git rebase` for conflict resolution, not merge commits

## CI

Primary CI: Azure DevOps Pipelines (`build/vsts-ci.yml`), the official signed build. Builds run on Windows, Linux (Ubuntu 22.04), and macOS, covering both managed (.NET) and native components. Code coverage uses `coverlet.collector`. A custom internal Roslyn analyzer (`Microsoft.ML.InternalCodeAnalyzer`) runs on all test projects.

## AI Infrastructure

### Workflows

GitHub Actions in `.github/workflows/`:

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `copilot-setup-steps.yml` | Manual | Remote Copilot Coding Agent build environment |
| `find-similar-issues.yml` | Issue opened | AI-powered duplicate detection for new issues |
| `inclusive-heat-sensor.yml` | Comments | Detect heated language in issue/PR comments |

### Prompts

Reusable prompt templates in `.github/prompts/`:

| Prompt | Purpose |
|--------|---------|
| `release-notes.prompt.md` | Generate classified release notes between commits |

### Issue Triage

For issue triage workflows (automated milestone assignment, priority labeling, investigation), use [GitHub Agentic Workflows](https://github.github.com/gh-aw/). Define triage automation as natural-language workflow files rather than custom scripts.
