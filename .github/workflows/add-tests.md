---
description: "Generates unit tests for code introduced in a pull request when a contributor comments /add-tests."

on:
  slash_command:
    name: add-tests
    events: [pull_request_comment]
  roles: [admin, maintainer, write]

permissions:
  contents: read
  pull-requests: read

imports:
  - shared/repo-build-setup.md

tools:
  github:
    toolsets: [pull_requests, repos]
  edit:
  bash: ["dotnet", "git", "find", "ls", "cat", "grep", "head", "tail", "wc", "mkdir"]

safe-outputs:
  create-pull-request:
    title-prefix: "[tests] "
    labels: [test, automated]
    draft: true
    max: 1
    protected-files: fallback-to-issue
  add-comment:
    max: 3

timeout-minutes: 45
---

# Add Tests for PR Changes

Generate comprehensive unit tests for the code changes introduced in pull request #${{ github.event.issue.number }}.

## Context

The PR comment that triggered this workflow: "${{ steps.sanitized.outputs.text }}"

## Goal

Analyze the pull request diff to identify source files that were added or modified, then generate unit tests that cover those changes. The resulting tests should be submitted as a new draft pull request.

## Instructions

### Step 1: Understand the PR Changes

1. Use the GitHub pull requests tools to fetch the PR diff for PR #${{ github.event.issue.number }}
2. Identify all **source files** (under `src/`) that were added or modified — ignore test files, build files, docs, and config
3. For each changed source file, understand what classes, methods, or functionality was added or changed

### Step 2: Identify Test Gaps

1. For each changed source file, find the corresponding existing test project (test projects mirror source naming: `Microsoft.ML.Foo` → `Microsoft.ML.Foo.Tests` under `test/`)
2. Check if the changed code already has test coverage
3. Focus on code that is **not yet covered** by existing tests

### Step 3: Generate Tests

Use the `code-testing-generator` agent (defined at `.github/agents/code-testing-generator.agent.md`) to generate tests:

1. Follow the Research → Plan → Implement pipeline from the skill
2. **Scope**: Only generate tests for code modified in this PR — do not attempt full-repo coverage
3. **Test framework**: This repo uses xUnit with `[Fact]`, `[Theory]`, `[InlineData]` attributes. Use `AwesomeAssertions` for fluent assertions where appropriate
4. **Test base classes**: Inherit from `TestDataPipeBase` (for data pipeline tests) or `BaseTestClass` (for simpler tests). Both constructors take `ITestOutputHelper output`
5. **Naming**: Test classes as `{Feature}Tests`, test methods as PascalCase descriptive names
6. **License header**: Every `.cs` file must start with the 3-line .NET Foundation MIT license header:
   ```
   // Licensed to the .NET Foundation under one or more agreements.
   // The .NET Foundation licenses this file to you under the MIT license.
   // See the LICENSE file in the project root for more information.
   ```
7. **Style**: Follow existing test patterns in the repo — check adjacent test files for conventions
8. **Validation**: Use `Contracts.Check*` patterns from `Microsoft.ML.Runtime` for input validation in helper code
9. **Build**: Use `dotnet build <TestProject.csproj>` for scoped builds during development
10. **Test**: Use `dotnet test <TestProject.csproj>` to verify tests pass

### Step 4: Validate

1. Build the specific test project(s) you modified
2. Run the tests to verify they pass
3. If tests fail, fix assertions based on actual production code behavior — never skip or ignore tests

### Step 5: Create the PR

Commit all test files and create a draft pull request. The PR description should:
- Reference the original PR (#${{ github.event.issue.number }})
- List the test files created
- Summarize what is covered by the new tests
