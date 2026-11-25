---
name: pull-request
description: Write or update the title and description of a dotnet/machinelearning pull request. Use when opening, creating, or editing a PR, or when asked for a PR title or description.
---

# ML.NET Pull Request Descriptions

Write for a reader who needs to understand the reason and behavior without re-reading the diff. Reviewers already have the Files tab, so do not repeat it.

## Title

- Imperative, specific, at most about 72 characters (for example, `Fix OVA softmax scratch sharing across prediction engines`).
- Name the behavior, not the files. No `fix:` prefix and no issue number.

## Body

Use the smallest shape that carries the signal:

- **Trivial or mechanical change:** one line, or title only.
- **Bug fix:** the symptom, the cause, and the fix in 1-3 sentences, plus `Fixes #N`.
- **Feature or multi-behavior change:** one short paragraph on why, then at most 5 bullets describing behaviors (never files).

Add only what helps a future reader:

- A verbatim error or minimal repro.
- Perf numbers with the measurement method.
- A compatibility or behavior change.
- One rejected alternative, if it prevents someone redoing it.
- For public API changes, links to the proposal issue and its `api-approved` comment (required by the [API review process](../../../docs/project-docs/api-review-process.md)).

Leave out:

- File, method, or test inventories and `Summary`/`Changes`/`Tests`/`Validation` section scaffolding.
- Restating the title, template greetings, or the template checklist.
- `Not in this PR` lists, unless they prevent a wrong assumption.
- CI run links, long scripts, and test logs.

Use `Fixes #N` only when the PR fully closes the issue. For partial or tracking work, use `Relates to #N`.

Use the [PR template](../../../PULL_REQUEST_TEMPLATE.md) as a pre-submission checklist. After checking its items, remove the template greeting and checklist from the submitted description, including the checkbox lines. Keep the actual issue and API-approval links.

## Validation and blockers

Say what was and was not validated to the user or reviewer in chat or a PR comment (skipped native tests, environment blockers, failures unrelated to the change). Put it in the body only if it is a permanent limitation of the change.

## Workflow

1. Draft the title and body in chat and wait for the user's approval before publishing. Do not open a PR without it.
2. Use the client's native PR tool when available. With `gh`, pass the body through `--body-file -` (stdin) or a temp file outside the repo. Do not use `-b "..."` or write the file into the worktree.
3. When review changes the approach, rewrite the description instead of appending `Update:` notes. Keep `Fixes`/`Relates to` lines and API-approval links when editing.

## Examples (illustrative)

Mechanical:

```
Fix typo in the MLContext XML documentation
```

Bug fix:

```
Fix NullReferenceException when saving a model with an empty pipeline

`TransformerChain.Save` dereferenced the last transformer without checking
the chain was empty. Return early for an empty chain.

Fixes #1234
```

Multi-behavior:

```
Add reusable softmax scratch buffers to OneVersusAll scoring

Scoring allocated a new softmax buffer per row, which dominated GC time on
wide multiclass models.

- Allocate the buffer once per prediction engine, not per row.
- Keep the shared mapper stateless so engines stay independent.

Relates to #5678
```
