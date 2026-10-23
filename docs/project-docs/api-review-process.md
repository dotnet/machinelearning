# API review process

Every public API change in a package built from this repository must first be proposed and explicitly approved in a GitHub issue. Obtain approval **before implementing the change or opening an implementation pull request**. This applies to maintainers, external contributors, and coding agents alike.

This process follows the [dotnet/runtime API review process](https://github.com/dotnet/runtime/blob/main/docs/project/api-review-process.md), with ML.NET maintainers responsible for arranging reviews and recording decisions for this repository.

## Which changes require review?

Review is required for externally accessible types and members, including protected members available to consumers through inheritance, across all packages built from this repository, including preview and experimental packages.

Changes requiring approval include:

- Adding types, members, overloads, enum values, or extension methods.
- Changing or removing APIs, including their names, visibility, parameters, return types, default values, generic constraints, nullability, or inheritance.
- Changing the intended public contract, even when signatures stay the same. Describe any changes to documented behaviour, supported inputs, exceptions, or compatibility.

Purely internal implementation changes, documentation or test fixes, and bug fixes that restore the existing intended contract do not require API approval. They still follow the normal [contribution guidelines](contributing.md). If it is unclear whether a change affects the public contract, ask a maintainer in the issue before implementing it.

## Proposing a change

Open an issue using the [API proposal template](https://github.com/dotnet/machinelearning/issues/new?template=propose-an-api.md). If an existing issue already tracks the change, add the proposal there rather than creating a duplicate.

The proposal should contain:

- **Background and motivation:** the problem, concrete user scenarios, and affected packages or assemblies.
- **Proposed API:** C# declarations, including namespaces and containing types, without implementation bodies. For changes or removals, show the existing and proposed API shapes.
- **Usage examples:** realistic calling code showing how the proposal solves the problem. Include migration examples when existing callers would need to change.
- **Alternative designs:** approaches considered, including whether the problem can be solved without changing the public API.
- **Compatibility and risks:** source, binary, and behavioural compatibility, along with relevant performance, dependency, platform, and model-persistence concerns. State explicitly when no breaking change is expected.

Keep the current proposal in the opening issue description as discussion progresses. Note significant revisions so reviewers can distinguish the latest design from earlier versions.

Exploratory prototypes can help assess a design, but are not approval to proceed with implementation. Keep such work in your own fork while the proposal is under review. Any draft PRs should also target your own fork, not `dotnet/machinelearning`.

## Review and approval

An ML.NET maintainer for the affected area discusses the proposal with the requester and arranges API review once the design is actionable. Maintainers manage the following review-state labels; contributors who cannot apply labels can ask for them in the issue.

| Label | Meaning |
| --- | --- |
| `api-suggestion` | An initial proposal or idea under discussion. Not approved for implementation. |
| `api-needs-work` | The proposal needs revision. Update the opening description and ask the maintainer to review it again. Not approved for implementation. |
| `api-ready-for-review` | The maintainer considers the proposal ready for API review. This is not approval to implement it. |
| `api-approved` | API review has approved the design recorded in the issue. Implementation may proceed for that design. |

The reviewing maintainer records the decision in an issue comment, including the agreed API declarations and any conditions or compatibility decisions, then replaces the previous review-state label with `api-approved`. A requester must not self-approve a proposal. General support for an idea, assignment to a milestone, or a pull request approval is not a substitute for this recorded API approval.

If review requires changes, the maintainer records the feedback and replaces the previous review-state label with `api-needs-work`. If the proposal is rejected, the maintainer explains the decision and closes the issue. There is no implementation approval in either case.

Breaking changes must also be identified with `breaking-change`. The issue must explain the impact on existing consumers and the migration path, and approval must explicitly cover that impact and the intended release. API approval does not waive the [release and versioning policy](release-process.md).

## Implementation and pull requests

For changes requiring review, wait until the issue is marked `api-approved` and any approval conditions are met, then:

1. Implement the approved API shape and behaviour.
2. Include the necessary tests, XML documentation, usage samples, and any affected API or test baselines.
3. Link both the proposal issue and its approval comment in the pull request, and complete the API-review item in the pull request template.

If implementation reveals a need to change the approved API, update the proposal and ask the maintainer to move it out of `api-approved` for further review. Obtain renewed approval in the issue before implementing the revised design. The earlier approval covers only the design that was reviewed.

Maintainers must confirm that pull requests containing API changes requiring review match the current approved design before merging. Such pull requests may be deferred or closed if opened before API approval. Passing CI or updating a baseline does not grant API approval, and API approval does not replace normal code review.
