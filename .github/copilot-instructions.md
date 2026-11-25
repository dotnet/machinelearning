---
description: "Shared guidance for coding agents working on ML.NET (dotnet/machinelearning)."
---

# ML.NET Development Guide

ML.NET is a cross-platform machine learning framework for .NET, with managed libraries and native dependencies. These are the canonical repository-wide instructions for coding agents.

## Public API Changes

Every public API addition, modification, or removal requires prior maintainer approval recorded in a GitHub issue labelled `api-approved`, before implementation or an implementation PR. This includes protected APIs, preview and experimental packages, and changes to the intended public contract even when signatures stay the same. Follow the [API review process](../docs/project-docs/api-review-process.md).

If required approval is missing, prepare a proposal rather than implementing the change. A linked issue, `api-ready-for-review`, or PR approval is not API approval. Link the approval comment in the PR, implement the approved design, and obtain renewed approval if that design changes.

When creating an API proposal follow *Phase 4: Draft Proposal* of [the dotnet/runtime API proposal skill](https://github.com/dotnet/runtime/blob/main/.github/skills/api-proposal/SKILL.md). Use the other phases of the skill only when explicitly asked for an API change prototype.

Purely internal implementation changes, documentation or test fixes, and bug fixes restoring the existing intended contract do not require API approval. If the public-contract impact is unclear, ask a maintainer in the issue before implementing.

## Build and Test

### First Build

Run commands from the repository root. Before building a fresh checkout:

1. Initialize submodules with `git submodule update --init`.
2. Follow the [Windows](../docs/building/windows-instructions.md) or [Linux/macOS](../docs/building/unix-instructions.md) prerequisites. Windows needs the documented Visual Studio workloads and CMake. Linux/macOS use `eng/install-dependencies.sh` as described in the platform guide.
3. Run the root build once before building or testing individual projects:

| Platform | Bootstrap and build |
|---|---|
| Windows | `.\build.cmd` |
| Linux/macOS | `./build.sh` |

The root build uses Microsoft Arcade to restore and build managed and native code. Arcade selects the SDK from `global.json`'s `tools.dotnet`, not `sdk.version`. Use the SDK installed or selected by the bootstrap for subsequent commands (`.dotnet` by default), rather than an unrelated `dotnet` on `PATH`. Check its `--version` against `tools.dotnet`.

### Incremental Work

After bootstrap, build the affected project and run focused tests. For example, on Windows with the default local SDK installation:

```powershell
.\.dotnet\dotnet.exe build .\src\Microsoft.ML.Core\Microsoft.ML.Core.csproj
.\.dotnet\dotnet.exe test .\test\Microsoft.ML.Core.Tests\Microsoft.ML.Core.Tests.csproj -f net10.0 --filter "FullyQualifiedName~ClassName.MethodName"
```

On Linux/macOS:

```sh
./.dotnet/dotnet build src/Microsoft.ML.Core/Microsoft.ML.Core.csproj
./.dotnet/dotnet test test/Microsoft.ML.Core.Tests/Microsoft.ML.Core.Tests.csproj -f net10.0 --filter "FullyQualifiedName~ClassName.MethodName"
```

Replace the example project and filter with those covering the change. Select supported frameworks from the affected project and [test/Directory.Build.props](../test/Directory.Build.props), which defines the default `net10.0;net11.0` targets (plus `net48` on Windows x64) and a `TestTargetFramework` override. A single-framework run does not cover all target frameworks.

- Run the smallest checks that exercise the changed behavior. For bug fixes, verify that the regression test fails for the intended reason without the fix and passes with it when practical.
- Confirm selected tests actually ran and inspect relevant skips. `NativeDependencyFact` skips when its native library cannot load. A skipped native-dependent test does not validate native behavior.
- State material validation blockers or gaps accurately. Do not treat skipped tests, unmatched filters, or stale binaries as successful coverage.
- Format changed files or the smallest relevant project, then inspect the diff. Do not use solution-wide formatting to introduce unrelated style changes.
- Documentation-only changes do not need a product build. Broaden validation when the affected behavior, platform, or shared infrastructure requires it.

See the [developer guide](../docs/project-docs/developer-guide.md) for additional build, architecture, and baseline-update commands. PR validation is defined in `.vsts-dotnet-ci.yml`, and the official build in `build/vsts-ci.yml`.

## Code Conventions

- Follow the applicable [.editorconfig](../.editorconfig) rules and nearby code. Preserve the owning area's namespace rather than deriving it from the assembly or directory name. Do not mix unrelated cleanup or language modernization into a functional change.
- Normal product and test C# files require the three-line .NET Foundation MIT header. Respect the configured exceptions for `docs/**/*.cs` samples and `test/Microsoft.ML.TestFrameworkCommon/Utility/*.cs` imported code.
- Use `Contracts.Check*` / `Contracts.Except*` where the owning library uses ML host validation. Preserve the established exception patterns in standalone libraries such as Tokenizers and DataFrame. Do not add a core-ML dependency solely to use `Contracts`.
- Follow existing friend-assembly and `[BestFriend]` conventions for cross-assembly internal access. `[BestFriend]` participates in analyzer checks and does not itself grant assembly access.
- Document public product APIs with XML comments. Update related documentation when the change makes it inaccurate or leaves a necessary contract undocumented.
- Keep package versions in the existing central configuration: [Directory.Packages.props](../Directory.Packages.props) and shared properties in [eng/Versions.props](../eng/Versions.props).
- Do not edit `eng/common` directly. It is mirrored from Arcade and changes are overwritten by automation. See its [AGENTS.md](../eng/common/AGENTS.md).

## Test Conventions

- Use the affected project's existing xUnit fixtures, `[Fact]`/`[Theory]` patterns, and `Assert.*` assertions. Prefer extending an existing relevant test class and use descriptive PascalCase test names.
- In suites using the ML test infrastructure, use the appropriate existing base (`TestDataPipeBase`, `BaseTestClass`, or an area-specific subclass). These are not universal requirements. Tokenizers suppresses `MSML_ExtendBaseTestClass`, and code-analyzer tests have their own fixtures.
- `BaseTestClass` descendants provide `ITestOutputHelper`, `GetDataPath`, and an `en-US` current culture for stable baselines. Preserve that default. Culture-specific tests must restore any temporary culture changes.
- Use the affected project's existing data fixtures. ML test-base suites resolve `GetDataPath` under `test/data`. Baselines live in `test/BaselineOutput`. Update them only for understood, intended output changes, not merely to make a failing test pass.
- The internal analyzer is enabled by default for applicable C# test projects. Follow project-specific settings and diagnostic exemptions rather than assuming every rule applies everywhere.

## Architecture and Navigation

- `MLContext`, the public entry point, is implemented in `src/Microsoft.ML.Data/MLContext.cs`. `src/Microsoft.ML` is the umbrella packaging project, not its implementation.
- `IDataView` is a lazy, columnar, cursor-based pipeline. Preserve laziness rather than materializing an entire dataset unnecessarily.
- Trainers follow the `IEstimator<T>` / `ITransformer` pattern (`Fit`, then `Transform`). Extend the owning area where appropriate. A new trainer does not automatically require a new project.
- Native sources live in `src/Native`. Account for the relevant platform and architecture when changing native code or interop.

## Git Workflow

Follow the [contribution guidelines](../docs/project-docs/contributing.md).

- Never commit directly to `main`. Reuse the assigned feature branch, or create one if needed.
- Keep each PR focused on one concern, with a brief description and references to relevant issues.
- The PR title and description become the squash commit message. Use a specific imperative title and state why and the resulting behavior in a few sentences. Do not list files or tests, add `Summary`/`Changes`/`Tests` sections, or paste logs. Keep validation notes and CI links out of the description and report them in chat or a PR comment when relevant. See the [pull-request](skills/pull-request/SKILL.md) skill.
- Address review feedback in additional commits rather than amending or force-pushing.
- Prefer `git rebase` over merge commits for conflict resolution. If this requires rewriting published history, obtain explicit authorization first.
- Squash merge is the default, with exceptions described in the contribution guidelines.
- GitHub writes require explicit approval for the intended action. Preparing a draft or loading a skill does not authorize publication.

## Task-Specific Guidance

Reusable procedures are in [.github/skills](skills/), and prompt templates in [.github/prompts](prompts/). Read the relevant procedure when the task calls for it rather than loading every workflow. Use the [code-review](skills/code-review/SKILL.md) skill when reviewing pull requests and the [pull-request](skills/pull-request/SKILL.md) skill when writing a PR title or description. Availability and automatic discovery depend on the coding client.

The [Copilot setup workflow](workflows/copilot-setup-steps.yml) bootstraps the cloud-agent environment. Do not assume its setup has already run in a local session.
