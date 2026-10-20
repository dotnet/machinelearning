---
name: Propose an API
about: Propose adding, changing, or removing a public API
title: '[API Proposal]: '
labels: ['api-suggestion']
assignees: ''

---

Public API changes outside preview or experimental packages require approval in an issue before implementation or an implementation pull request. Please follow the [API review process](https://github.com/dotnet/machinelearning/blob/main/docs/project-docs/api-review-process.md). Maintainers record approval and apply `api-approved` after review.

### Background and motivation

<!-- Describe the problem, concrete user scenarios, and affected packages or assemblies. -->

### Proposed API

<!--
Provide C# declarations in a fenced code block, including namespaces and containing types, without implementation bodies.
For changes or removals, show the existing and proposed API shapes.
Keep this description up to date as the design changes during review.
-->

### Usage examples

<!-- Show realistic calling code using the proposed API. Include migration examples if existing callers must change. -->

### Alternative designs

<!-- Describe alternatives considered, including approaches that do not change the public API. -->

### Compatibility and risks

<!--
Describe source, binary, and behavioural compatibility, and any performance, dependency, platform, or model-persistence concerns.
For breaking changes, explain the impact, migration path, and intended release.
State explicitly if no breaking change is expected.
-->
