---
on:
  issues:
    types: [opened]

# ###############################################################
# Select a PAT from the pool and override COPILOT_GITHUB_TOKEN.
# Run agentic jobs in an isolated `copilot-pat-pool` environment.
#
# When org-level billing is available, this will be removed.
# See `shared/pat_pool.README.md` for more information.
# ###############################################################
imports:
  - uses: shared/pat_pool.md
    with:
      environment: copilot-pat-pool

environment: copilot-pat-pool

engine:
  id: copilot
  env:
    COPILOT_GITHUB_TOKEN: |
      ${{ case(
        needs.pat_pool.outputs.pat_number == '0', secrets.COPILOT_PAT_0,
        needs.pat_pool.outputs.pat_number == '1', secrets.COPILOT_PAT_1,
        needs.pat_pool.outputs.pat_number == '2', secrets.COPILOT_PAT_2,
        needs.pat_pool.outputs.pat_number == '3', secrets.COPILOT_PAT_3,
        needs.pat_pool.outputs.pat_number == '4', secrets.COPILOT_PAT_4,
        needs.pat_pool.outputs.pat_number == '5', secrets.COPILOT_PAT_5,
        needs.pat_pool.outputs.pat_number == '6', secrets.COPILOT_PAT_6,
        needs.pat_pool.outputs.pat_number == '7', secrets.COPILOT_PAT_7,
        needs.pat_pool.outputs.pat_number == '8', secrets.COPILOT_PAT_8,
        needs.pat_pool.outputs.pat_number == '9', secrets.COPILOT_PAT_9,
        'NO COPILOT PAT AVAILABLE')
      }}

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
