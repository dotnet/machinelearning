---
applyTo:
  - "src/Microsoft.ML.GenAI*/**"
  - "test/Microsoft.ML.GenAI*/**"
  - "src/Microsoft.ML.TorchSharp/**"
  - "test/Microsoft.ML.TorchSharp*/**"
---

# GenAI & TorchSharp Guidelines

## Overview

The `Microsoft.ML.GenAI.*` projects provide .NET-native support for running large language models (LLaMA, Phi, Mistral) via TorchSharp. These components integrate with Semantic Kernel and Microsoft.Extensions.AI.

## Key Patterns

- `CausalLMPipeline` is the core abstraction for running autoregressive text generation
- Model implementations live in separate projects per architecture: `GenAI.LLaMA`, `GenAI.Phi`, `GenAI.Mistral`
- Shared types and utilities live in `GenAI.Core`
- TorchSharp tensor operations are in `Microsoft.ML.TorchSharp`

## Dependencies

- `Microsoft.SemanticKernel` / `Microsoft.SemanticKernel.Abstractions`
- `Microsoft.Extensions.AI.Abstractions`
- `TorchSharp` (native PyTorch bindings for .NET)

## Testing GenAI

GenAI tests may require model weight files. Check test setup for model download steps or mock data patterns before running.
