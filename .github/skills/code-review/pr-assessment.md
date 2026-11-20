# PR Assessment

Review-only criteria for assessing a pull request as a whole. Coding conventions live in `.github/copilot-instructions.md`. Use these criteria to write the Motivation, Approach, and Summary fields of the review.

Before reviewing individual lines, decide whether the change is justified, takes the right approach, and is a net positive for the codebase.

## Motivation & Justification

- **Every PR must state what problem it solves and why.** Ask for the rationale if it is missing. A link to an approved API proposal, accepted issue, or prior discussion is enough.
- **Challenge every addition with "Do we need this?"** New code, APIs, options, and abstractions must justify their existence.
- **Require real use cases for new surface area.** Hypothetical benefits are not enough to expand the public API or add trainers, transforms, or options.

## Evidence & Data

- **Require measurements for performance claims.** Ask for BenchmarkDotNet results (see `test/Microsoft.ML.PerformanceTests`) or equivalent data. Do not accept performance claims at face value.
- **Prefer realistic workloads.** Micro-benchmarks on tiny or uniform inputs can overstate gains. Datasets, feature counts, and sparsity should reflect real ML workloads.
- **Check model-quality impact.** Changes to trainers, featurizers, or numerical code should show that metrics (and baselines) stay the same or that any shift is intended and explained.
- **Investigate regressions before merging.** A net improvement does not excuse an unexplained regression in a specific scenario.

## Approach & Alternatives

- **Fix the right problem at the right layer.** Prefer root-cause fixes over workarounds in product code.
- **Redirect early when the approach is wrong.** Don't iterate on details of a flawed design.
- **Ask "Why not just X?"** The burden of proof is on the more complex solution.

## Cost-Benefit & Complexity

- **Weigh whether the change is a net positive** in the typical configuration, not just a narrow scenario.
- **Complexity is a cost.** Reject unnecessary abstraction, indirection, and elaborate solutions for marginal gains.
- **Every addition is a maintenance obligation**, including new package dependencies and native binaries.

## Scope & Focus

- **One concern per PR.** Ask to split large or mixed PRs, especially across assemblies or when mixing formatting with behavior changes.
- **Defer tangential improvements** to follow-up PRs.

## Risk & Compatibility

- **Public API changes need recorded approval.** Changes to the public API or its documented contract need an `api-approved` issue (see `docs/project-docs/api-review-process.md`). Bug fixes that restore the intended contract do not.
- **Flag breaking changes.** Source, binary, behavioral, and saved-model compatibility breaks need the `breaking-change` label, an explained migration path, and explicit approval.
- **Scale scrutiny to blast radius.** Changes to core data pipeline types (`Microsoft.ML.Core`, `Microsoft.ML.Data`), widely used trainers, or native code need stronger validation.
- **Never invoke `/backport` yourself.** Recommend it when appropriate and leave the command to a maintainer.

## Codebase Fit & History

- **Match existing patterns and conventions.** Renames or restructuring should be done uniformly in a dedicated PR, not piecemeal.
- **Check whether a similar approach was tried and rejected before.** If so, require a clear explanation of what is different.
