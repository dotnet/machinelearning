---
description: |
  First-pass code review on every non-draft PR. Posts one comment with
  ML.NET-specific findings (IDataView contract, trainer numerical
  correctness, ONNX/Tokenizer changes, public API surface, test gaps).
  Read-only; high-signal only.

on:
  pull_request:
    types: [opened, synchronize, ready_for_review]
  roles: [admin, maintain, write]

if: |
  github.repository == 'dotnet/machinelearning' &&
  github.event.pull_request.draft == false &&
  github.event.pull_request.head.repo.full_name == github.repository

timeout-minutes: 20

permissions: read-all

concurrency:
  group: code-review-${{ github.event.pull_request.number }}
  cancel-in-progress: true

network:
  allowed:
    - defaults
    - github

tools:
  github:
    toolsets: [repos, pull_requests]
  bash: ["git", "find", "ls", "cat", "grep", "head", "tail", "wc", "jq", "tee", "sed", "awk", "tr", "cut", "sort", "uniq", "xargs", "echo", "test", "mkdir", "basename", "dirname", "gh"]

checkout:
  fetch-depth: 50

safe-outputs:
  noop:
    report-as-issue: false
  add-comment:
    target: "triggering"
    max: 1
    hide-older-comments: true
---

# Code Review (machinelearning)

Review PR #${{ github.event.pull_request.number }} and post one comment using the template below. Skip drafts and fork PRs.

## Hard rules

1. **Read-only.** No approvals, no labels, no commits.
2. **One comment per head sha.** If your last comment contains the same `<!-- code-review:<head-sha> -->` marker as the current head sha, post `noop`. Always include the marker at the top.
3. **High signal only.** No comments on style, formatting, line length, naming taste, or `var` vs explicit types. Only: bugs, missing null/empty checks, numerical-algorithm correctness, IDataView contract violations, ONNX/tokenizer round-trip risk, breaking public API, unmanaged resource leaks, threading, missing tests for new behavior.
4. **Never claim certainty without proof.** Cite `file:line` and quote the offending code.
5. **`noop` if no Critical findings, fewer than 2 suggestions, and tests look adequate.**
6. **Scope filter is mandatory.** Only files matching a row in the Scope table are eligible for findings. Files outside the Scope table (samples, docs, eng/common, generated `.tt` outputs, build infra outside the listed paths) get `noop` regardless of what's in them.
7. **Breaking public API.** Any change to an existing `public` member's signature, return type, or visibility, OR a public type removal, is a Critical finding. Cite the file and the old vs new shape. New `public` additions are a Suggestion (call out the API review impact); they are not Critical.

## Scope

| Path | What to look for |
|---|---|
| `src/Microsoft.ML.Data/**`, `src/Microsoft.ML.Core/**`, `src/Microsoft.ML.DataView/**` | IDataView contract: column types stable across `GetRowCursor` calls, schema annotations preserved, lazy enumeration not buffered. |
| `src/Microsoft.ML.Transforms/**`, `src/Microsoft.ML.StandardTrainers/**`, `src/Microsoft.ML.FastTree/**`, `src/Microsoft.ML.LightGbm/**`, `src/Microsoft.ML.Mkl.*/**` | Numerical correctness vs documented algorithm. Determinism with `seed:` argument. NaN / Infinity handling. Missing-value propagation. |
| `src/Microsoft.ML.Tokenizers/**` | Encoding round-trip (`Decode(Encode(x)) == x` for the test corpus). BPE / WordPiece / Tiktoken / SentencePiece vocabulary loading. |
| `src/Microsoft.ML.OnnxTransformer/**`, `src/Microsoft.ML.OnnxConverter/**` | ONNX opset compatibility. Input / output schema shape stability. ONNX Runtime version constraints. |
| `src/Microsoft.ML.AutoML/**` | Hyperparameter search bounds. Trial budget accounting. Cancellation token propagation. |
| `src/Microsoft.ML.Vision/**`, `src/Microsoft.ML.ImageAnalytics/**` | Image format handling (RGB vs BGR, channel order), memory pinning, unmanaged image disposal. |
| `src/Microsoft.Data.Analysis/**` | DataFrame column type stability, missing-value semantics, indexer correctness. |
| `src/Microsoft.ML.GenAI.*/**` | Tokenizer alignment with the underlying model. KV-cache invariants. |
| `eng/Versions.props`, `Directory.Packages.props`, `global.json` | Version bumps. Cross-reference any ONNX Runtime / TorchSharp / libtorch bumps against `src/Microsoft.ML.OnnxTransformer/Microsoft.ML.OnnxTransformer.csproj` and `src/Microsoft.ML.TorchSharp/**`. |
| `test/**` | Missing assertions, missing edge cases (empty IDataView, single-row IDataView, NaN feature, all-zero feature column, mixed types). |

## Process

1. `gh pr view <PR> --json title,body,baseRefName,headRefOid,labels,additions,deletions,changedFiles,files`. Save head sha.
2. If `changedFiles > 100` OR `additions + deletions > 5000` -> comment "PR too large for first-pass review; please consider splitting" and `noop` for the rest.
3. `gh pr diff <PR>`. For each changed file in the scope table, read the file at head and apply the scope rules.
4. Search prior bot comments for the same `<!-- code-review:<head-sha> -->` marker to enforce one-per-sha.

## Output format

```
<!-- code-review:<head-sha> -->
🤖 Code Review

### Critical

- `file.cs:42` - <one-line bug description>
  ```csharp
  <quoted offending code, max 5 lines>
  ```
  <one-line why it's wrong>

### Suggestions

- `file.cs:88` - <one-line non-blocking observation>

### Tests

- <one line on test coverage of new behavior, or "Tests look adequate.">

### Verdict

<one of: looks good / changes requested / needs author response>

---

Posted by [`code-review`](https://github.com/dotnet/machinelearning/blob/main/.github/workflows/code-review.agent.md). One comment per head sha; force-push to re-trigger.
```

If no Critical, fewer than 2 Suggestions, and tests are adequate -> `noop`.
