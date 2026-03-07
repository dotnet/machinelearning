---
description: "Guidance for GitHub Copilot when working on ML.NET (dotnet/machinelearning)."
---

# Development Instructions

## Repository Overview

ML.NET is a cross-platform, open-source machine learning framework for .NET. It provides APIs for training, evaluating, and deploying ML models including classification, regression, clustering, ranking, anomaly detection, time series, recommendation, and generative AI (LLaMA, Phi, Mistral via TorchSharp).

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

The repo uses Arcade SDK — `build.sh`/`build.cmd` wraps `eng/common/build.sh`/`eng/common/build.ps1` with `--restore --build`. Native dependencies require `eng/common/native/install-dependencies.sh` on Linux.

### Test

```bash
# Run tests for a specific project
dotnet test test/Microsoft.ML.Tests/Microsoft.ML.Tests.csproj

# Run tests with filter
dotnet test test/Microsoft.ML.Tests/Microsoft.ML.Tests.csproj --filter "FullyQualifiedName~ClassName.MethodName"

# Run all tests (slow — use specific projects)
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
├── Microsoft.ML.Core/              ← Core types, contracts, host environment
├── Microsoft.ML.Data/              ← Data pipeline, DataView, schema
├── Microsoft.ML/                   ← MLContext, public API surface
├── Microsoft.ML.StandardTrainers/  ← Built-in trainers (logistic regression, SVM, etc.)
├── Microsoft.ML.Transforms/        ← Data transforms (normalize, featurize, etc.)
├── Microsoft.ML.AutoML/            ← Automated ML pipeline selection
├── Microsoft.ML.FastTree/          ← Tree-based trainers
├── Microsoft.ML.LightGbm/          ← LightGBM integration
├── Microsoft.ML.Recommender/       ← Matrix factorization recommenders
├── Microsoft.ML.TimeSeries/        ← Time series analysis
├── Microsoft.ML.Tokenizers/        ← BPE/WordPiece/SentencePiece tokenizers
├── Microsoft.ML.GenAI.Core/        ← GenAI base types (CausalLM pipeline)
├── Microsoft.ML.GenAI.LLaMA/       ← LLaMA model support
├── Microsoft.ML.GenAI.Phi/         ← Phi model support
├── Microsoft.ML.GenAI.Mistral/     ← Mistral model support
├── Microsoft.ML.TorchSharp/        ← TorchSharp-based trainers
├── Microsoft.ML.OnnxTransformer/   ← ONNX model inference
├── Microsoft.ML.TensorFlow/        ← TensorFlow model inference
├── Microsoft.ML.Vision/            ← Image classification
├── Microsoft.ML.ImageAnalytics/    ← Image transforms
├── Microsoft.ML.CpuMath/           ← SIMD-optimized math operations
├── Microsoft.Data.Analysis/        ← DataFrame API
├── Native/                          ← C/C++ native library sources
└── Common/                          ← Shared internal code
test/
├── Microsoft.ML.TestFramework/      ← Base test classes and helpers
├── Microsoft.ML.TestFrameworkCommon/ ← Shared test utilities
├── Microsoft.ML.Tests/              ← Main functional tests
├── Microsoft.ML.Core.Tests/         ← Core unit tests
├── Microsoft.ML.IntegrationTests/   ← End-to-end integration tests
├── Microsoft.ML.Tokenizers.Tests/   ← Tokenizer tests
├── Microsoft.ML.GenAI.*.Tests/      ← GenAI component tests
└── ... (30+ test projects)
```

## Conventions

### Code Style

- **License header**: Every `.cs` file starts with the 3-line .NET Foundation MIT license header
- **Namespaces**: Match assembly name (`Microsoft.ML`, `Microsoft.ML.Data`, `Microsoft.ML.Trainers`)
- **Usings**: `System.*` first, then `Microsoft.*`, then others
- **Visibility**: Use `[BestFriend]` attribute for internal members shared across assemblies; `private protected` where appropriate
- **Validation**: Use `Contracts.Check*` / `Contracts.Except*` for argument and state validation — not raw `throw` statements
- **XML docs**: Required on all public types and members with `<summary>` tags
- **Style priority**: Match the existing style of the file you're editing, even if it differs from general guidelines
- Follow [dotnet/runtime coding-style](https://github.com/dotnet/runtime/blob/main/docs/coding-guidelines/coding-style.md)

### Test Conventions

- **Framework**: xUnit (`[Fact]`, `[Theory]`, `[InlineData]`)
- **Base class**: Inherit from `TestDataPipeBase` → `BaseTestClass` (provides `ITestOutputHelper`, test data paths, locale pinning to `en-US`)
- **Constructor**: Accept `ITestOutputHelper output` and pass to base
- **Naming**: PascalCase descriptive method names (e.g., `RandomizedPcaTrainerBaselineTest`)
- **Assertions**: `Assert.*` (xUnit), `AwesomeAssertions` for fluent assertions
- **Test data**: Use `Microsoft.ML.TestDatabases` package or files in `test/data/`
- **Baseline output**: Compare against expected output in `test/BaselineOutput/`

### Architecture

- The main entry point is `MLContext` — it exposes catalogs for each ML task
- Data flows through `IDataView` — a lazy, columnar, cursor-based data pipeline
- Trainers implement `IEstimator<T>` → `ITransformer` pattern (fit → transform)
- Custom trainers go in their own project under `src/`
- New test projects mirror source project naming: `Microsoft.ML.Foo` → `Microsoft.ML.Foo.Tests`

## Git Workflow

- Default branch: `main`
- Never commit directly to `main` — always create a feature branch
- Branch naming: `feature/description`, `fix/description`
- PRs are squash-merged
- Must reference a filed issue in PR description
- Address review feedback in additional commits (don't amend/force-push)
- Use `git rebase` for conflict resolution, not merge commits

## CI

- **Primary CI**: Azure DevOps Pipelines (`build/vsts-ci.yml`) — official signed build
- Builds on Windows, Linux (Ubuntu 22.04), macOS
- Test runs include both managed (.NET) and native components
- Code coverage via `coverlet.collector`
- A custom internal Roslyn analyzer (`Microsoft.ML.InternalCodeAnalyzer`) runs on all test projects
