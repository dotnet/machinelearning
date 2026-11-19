# API Approval Verification

Run this procedure when Step 1 of `SKILL.md` detects a public API change: an added, changed, or removed externally accessible type or member (including `protected` members of public types), or a change to the documented contract of an existing API. It follows `docs/project-docs/api-review-process.md`. This is a **blocking** procedure. If any step fails, report it as ❌.

Bug fixes that restore the existing intended contract, purely internal changes, and documentation or test fixes do not need API approval. If you are unsure whether a change affects the public contract, report ⚠️ and ask the author to confirm with a maintainer.

## 1. Locate the `api-approved` Issue

Parse the PR description for issue references (`Fixes #N`, `Closes #N`, `Resolves #N`, bare `#N`, or issue URLs). Use `gh` or GitHub MCP tools to check whether any linked issue has the `api-approved` label.

- **If no linked issue has `api-approved`**: Report ❌ error "This PR changes public API but has no linked issue with the `api-approved` label. Public API changes need an approved proposal before implementation (see `docs/project-docs/api-review-process.md`). Link the approved issue. Do not recommend making an existing public API internal, since that is itself a breaking change."
- `api-suggestion`, `api-needs-work`, and `api-ready-for-review` are **not** approval.

## 2. Find the Recorded Approval

The approving maintainer records the decision in an issue comment with the agreed API declarations and any conditions. The PR should link that comment.

1. Use the approval comment linked from the PR when present. Otherwise, find the maintainer comment that records the approval (usually posted near when `api-approved` was applied).
2. Confirm it was posted by a maintainer, not the requester (self-approval is not allowed).
3. Extract the approved API declarations from the comment, whether they appear in a fenced block or prose, along with any conditions. Do not treat a particular Markdown fence language as a requirement for valid approval.

- **If no recorded approval can be found**: Report ❌ error "Cannot find the recorded API approval. The approving maintainer must record the approved API in an issue comment."

## 3. Compare Implementation Against the Approved Shape

Compare the PR's public declarations in `src/` against the approved shape. Check:

- **Namespaces and type names** must match exactly.
- **Member signatures** (name, return type, parameter types, generic constraints, nullability, default values) must match.
- **Parameter names** must match, since a mismatch breaks named arguments.
- **Extra public surface** not in the approved shape is unapproved and must be flagged.
- **Missing approved surface** should be flagged unless the PR states it is a partial implementation with a tracking issue.
- **Conditions** recorded in the approval (behavior, compatibility, packaging) must be met.
- **Breaking changes** need the `breaking-change` label on the issue and approval that explicitly covers the impact.

Severity:

- ❌ **error**: Extra unapproved API, wrong names or signatures, missing approved API without explanation, unmet approval conditions, or an unapproved breaking change.
- ⚠️ **warning**: Minor differences that may be intentional but need confirmation (for example, extra `[EditorBrowsable(Never)]` attributes or TFM-specific `#if` guards).
