---
name: run-tests
description: "Build and run tests locally with filtering. Use when asked to run tests, verify a fix, or check test results."
---

# Run Tests

## Quick Start

```bash
# All tests (slow — avoid unless necessary)
dotnet test Microsoft.ML.sln

# Specific project (preferred)
dotnet test test/Microsoft.ML.Tests/Microsoft.ML.Tests.csproj

# Filter by name
dotnet test test/Microsoft.ML.Tests/Microsoft.ML.Tests.csproj --filter "FullyQualifiedName~AnomalyDetectionTests"
```

## Test Projects

| Project | Path | Area |
|---------|------|------|
| Microsoft.ML.Tests | test/Microsoft.ML.Tests/ | Main functional tests |
| Microsoft.ML.Core.Tests | test/Microsoft.ML.Core.Tests/ | Core type tests |
| Microsoft.ML.CpuMath.UnitTests | test/Microsoft.ML.CpuMath.UnitTests/ | SIMD math tests |
| Microsoft.ML.AutoML.Tests | test/Microsoft.ML.AutoML.Tests/ | AutoML tests |
| Microsoft.ML.Tokenizers.Tests | test/Microsoft.ML.Tokenizers.Tests/ | Tokenizer tests |
| Microsoft.ML.GenAI.Core.Tests | test/Microsoft.ML.GenAI.Core.Tests/ | GenAI core tests |
| Microsoft.ML.GenAI.LLaMA.Tests | test/Microsoft.ML.GenAI.LLaMA.Tests/ | LLaMA tests |
| Microsoft.ML.GenAI.Phi.Tests | test/Microsoft.ML.GenAI.Phi.Tests/ | Phi tests |
| Microsoft.ML.GenAI.Mistral.Tests | test/Microsoft.ML.GenAI.Mistral.Tests/ | Mistral tests |
| Microsoft.ML.TorchSharp.Tests | test/Microsoft.ML.TorchSharp.Tests/ | TorchSharp tests |
| Microsoft.ML.TimeSeries.Tests | test/Microsoft.ML.TimeSeries.Tests/ | Time series tests |
| Microsoft.ML.IntegrationTests | test/Microsoft.ML.IntegrationTests/ | End-to-end tests |
| Microsoft.ML.Fairlearn.Tests | test/Microsoft.ML.Fairlearn.Tests/ | Fairness tests |
| Microsoft.Data.Analysis.Tests | test/Microsoft.Data.Analysis.Tests/ | DataFrame tests |
| Microsoft.Extensions.ML.Tests | test/Microsoft.Extensions.ML.Tests/ | DI integration tests |
| Microsoft.ML.SearchSpace.Tests | test/Microsoft.ML.SearchSpace.Tests/ | Search space tests |
| Microsoft.ML.Sweeper.Tests | test/Microsoft.ML.Sweeper.Tests/ | Hyperparameter sweep tests |
| Microsoft.ML.Predictor.Tests | test/Microsoft.ML.Predictor.Tests/ | Predictor tests |
| Microsoft.ML.FSharp.Tests | test/Microsoft.ML.FSharp.Tests/ | F# interop tests |

## Filtering

```bash
# By class name
dotnet test PROJECT --filter "FullyQualifiedName~ClassName"

# By single method
dotnet test PROJECT --filter "FullyQualifiedName~ClassName.MethodName"

# By trait/category
dotnet test PROJECT --filter "Category=Unit"
```

## Prerequisites

```bash
# Full build (includes restore)
./build.sh        # Linux/macOS
build.cmd         # Windows

# Or build specific project
dotnet build src/Microsoft.ML/Microsoft.ML.csproj
```
