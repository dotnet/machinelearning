---
on:
  issues:
    types: [opened]
permissions:
  contents: read
  issues: read
safe-outputs:
  add-labels:
    allowed: [bug, enhancement, question, documentation, perf, test, Build, untriaged, needs-further-triage, need info]
  add-comment: {}
---

# Issue Triage

Analyze newly opened issues and apply appropriate labels.

## Label Definitions

Choose the most specific label that fits:

- **bug**: Reports of broken functionality, regressions, unexpected behavior, or crashes
- **enhancement**: Feature requests or improvements to existing functionality
- **question**: Questions about usage, API behavior, or how to accomplish something
- **documentation**: Documentation gaps, inaccuracies, or requests for new docs
- **perf**: Performance regressions, slow operations, or optimization requests
- **test**: Test infrastructure failures, flaky tests, or test coverage gaps
- **Build**: Build system, CI/CD pipeline, or dependency/packaging issues
- **need info**: Issue is too vague to classify without more details from the author
- **needs-further-triage**: Issue is clear but requires maintainer judgment to prioritize

Always add **untriaged** alongside any other label. This signals to maintainers that a human has not yet reviewed the classification.

## Area Hints

If the issue mentions specific components, note the relevant project area in your comment. Common areas in this repo:

- Tokenizers (`Microsoft.ML.Tokenizers`)
- GenAI / LLM support (`Microsoft.ML.GenAI.*`, TorchSharp)
- AutoML (`Microsoft.ML.AutoML`)
- Data pipeline / DataView (`Microsoft.ML.Data`)
- Image classification (`Microsoft.ML.Vision`, `Microsoft.ML.ImageAnalytics`)
- Time series (`Microsoft.ML.TimeSeries`)
- ONNX / TensorFlow interop
- DataFrame (`Microsoft.Data.Analysis`)

## Instructions

1. Read the issue title and body
2. Determine the single most appropriate label from the list above
3. Add that label plus **untriaged**
4. Leave a comment that includes:
   - Why this label was chosen and the likely area of the codebase affected
   - **For bug reports**: If the author provided enough detail, write a minimal repro (a short C# code snippet using MLContext that demonstrates the issue). If the description is too vague for a repro, ask the author for specific inputs, expected vs actual behavior, and framework version.
   - **For enhancement requests**: Outline a brief plan of what implementing this would involve (which projects/files are likely affected, rough scope).
   - **For questions**: Point to relevant API docs, code samples, or existing tests that might answer the question.
5. If the issue lacks a clear description, use **need info** and ask the author for specifics
