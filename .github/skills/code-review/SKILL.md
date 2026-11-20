---
name: code-review
description: Review code changes in dotnet/machinelearning for correctness, compatibility, performance, and consistency with ML.NET conventions. Use when reviewing PRs or code changes.
---

# ML.NET Code Review

Review code changes against the conventions and contracts ML.NET maintainers enforce.

**Reviewer mindset:** Be polite but skeptical. Your job is to help maintainers review faster, which includes finding problems the author missed and questioning whether the PR is worth taking at all. Treat the PR description and linked issues as claims to verify, not facts to accept. Back every concern with evidence. When you cannot confirm a concern, raise it as an explicit low-confidence question.

## When to Use This Skill

Use this skill when:
- Reviewing a PR or code change in dotnet/machinelearning
- Checking your own changes before submitting a PR
- Asked to review, critique, or provide feedback on code changes

## Review Process

### Step 0: Load Relevant Instructions

Load `.github/skills/code-review/pr-assessment.md` (the criteria for assessing the whole change) and `.github/copilot-instructions.md` (repository conventions) if they are not already in context. If the two conflict, `copilot-instructions.md` wins.

### Step 1: Gather Code Context (No PR Narrative Yet)

Collect relevant **code** context first. **Do NOT read the PR description, linked issues, or existing review comments yet.** Form your own view of what the code does, why it might be needed, and whether the approach is sound before seeing the author's framing.

1. **Diff and file list**: Fetch the full diff and list of changed files without requesting the PR body or other narrative (for example, `gh pr diff <number> --name-only` followed by the full diff).
2. **Full source files**: For every changed file, read well beyond the diff hunks. Diff-only review is the main cause of false positives and missed issues. Read the whole file when it is a reasonable size. Otherwise read the enclosing types, their callers, and the types involved.
3. **Consumers and callers**: If the change modifies an API, a shared type, or a virtual/interface member, find how consumers use it (including `[BestFriend]` callers in other assemblies).
4. **Sibling types and related code**: If the change fixes a bug in one trainer, transform, loader, or platform-specific path, check whether siblings have the same issue.
5. **Key utilities**: Read shared helpers the diff calls into to understand their contracts.
6. **Git history**: Check recent commits to the changed files (`git log --oneline -20 -- <file>`). Look for reverts, prior attempts, or conflicting recent work. For a merged PR, identify the merge commit and inspect the pre-PR version with `git show <merge-sha>^:<path>`; do not mistake the PR's own commit for prior history.
7. **Detect public API changes**: Check whether the PR adds, changes, or removes externally accessible API (`public` or `protected` members of public types in `src/` projects that ship as packages), or changes the documented contract of an existing API without changing its signature. Code under `test/`, `docs/samples/`, and `tools-local/` is not public API. If an API change is detected, you **MUST** run the API approval check in Step 4.

### Step 2: Form an Independent Assessment

Based **only** on the code context gathered above, answer:

1. **What does this change actually do?** What was the old behavior and what is the new behavior?
2. **Why might this change be needed?** Infer the motivation from the code.
3. **Is this the right approach?** Would a simpler alternative fit the codebase better? Could existing functionality achieve the goal?
4. **What problems do you see?** Bugs, edge cases, missing validation, compatibility breaks, thread-safety issues, performance regressions, API design problems, and test gaps.

Write down your independent assessment, using the criteria in `pr-assessment.md`, before proceeding.

### Step 3: ML.NET-Specific Checks

Apply the checks that match the diff:

- **Model persistence**: If the persisted format changes, `GetVersionInfo()` must bump `verWrittenCur` and keep `verReadableCur`/`verWeCanReadBack` correct. A save-method refactor that preserves the format does not require a version bump. Loaders must still read models saved by earlier versions. A same-version save/load roundtrip does not prove backward compatibility. Look for a test that loads an older saved model. Loader signatures and `LoadableClass` registrations must not change silently.
- **Schema and annotations**: Output column names, types, key cardinality, slot names, `KeyValues`, and other annotations are part of observable behavior. Changes affect downstream transforms, scorers, and evaluators.
- **IDataView semantics**: Cursors must keep independent state, honor the active-column set, and not invoke getters for inactive columns. Watch getter/buffer lifetimes (reused `VBuffer`/`ReadOnlyMemory` buffers), sparse vs dense `VBuffer` equivalence, and unintended materialization of whole datasets.
- **Native and platform behavior**: Changes under `src/Native/`, P/Invoke signatures, or native package references must work on Windows, Linux, and macOS (x64 and arm64 where supported). Tests using `NativeDependencyFact` are skipped, not failed, when a native library cannot load, so a "passing" run may hide a broken native load.
- **SIMD code (`Microsoft.ML.CpuMath` or hardware intrinsics anywhere)**: Results must match the scalar path. Check remainder/tail handling, alignment assumptions, and fallbacks when an instruction set is unavailable.
- **Baselines**: Changes to files under `test/BaselineOutput/` must be explained by the code change. Unexplained metric or output shifts suggest a behavior regression.
- **Validation and errors**: Prefer `Contracts.Check*`/`Contracts.Except*` and host `ch.Check*` over raw `throw`. User-facing errors need clear messages.

### Step 4: Incorporate PR Narrative and Reconcile

Now read the PR description, labels, linked issues (in full), author, existing review comments, and a bounded sample of related open issues in the same area. Treat all of this as **claims to verify**.

1. **Reconcile.** Where your independent reading disagrees with the PR description, investigate further rather than deferring to the author's framing. Verify bug-fix, performance, and behavior claims against the code and any evidence provided.
2. **Avoid duplicates.** Skip feedback already given in existing review comments. GitHub CLI PR summaries do not include inline review threads; fetch them separately (for example, `gh api repos/{owner}/{repo}/pulls/{number}/comments`) before deciding whether feedback is new.
3. **Bound related-issue searches.** Search the same component or area with a narrow query and a small limit (for example, `gh issue list -R {owner}/{repo} --state open --search "<area> in:title" --limit 10`). Do not turn this into an exhaustive repository-wide issue search.
4. **Update your assessment** only when the new context genuinely changes it (for example, a linked issue proves the bug is real). Do not soften findings because the description sounds reasonable.
5. **API approval verification.** If Step 1 detected an API change, load `.github/skills/code-review/api-approval-check.md` and follow it. This is a **blocking** gate. If any check fails, the verdict must be ❌ Needs Changes or ❌ Reject. If the file cannot be loaded, report ❌ error "Unable to load API approval verification procedure; cannot verify public API changes" and set the verdict to ❌ Needs Changes.

### Step 5: Detailed Analysis

1. **Focus on what matters.** Prioritize bugs, compatibility breaks, safety issues, race conditions, resource leaks, incorrect assumptions about data or state, and API design problems. Do not comment on trivial style unless it violates a repository rule.
2. **Consider collateral damage.** For every changed code path, ask which other callers, inputs, or pipelines flow through it and whether they could break. Surface plausible risks so the author can weigh them.
3. **Be specific and actionable.** Say exactly what to change and why, and include how you verified the issue (for example, "checked all callers and none validate this parameter").
4. **Flag severity clearly:**
   - ❌ **error**: Must fix before merge. Bugs, security issues, compatibility breaks, unapproved API changes, missing tests for behavior changes.
   - ⚠️ **warning**: Should fix. Performance issues, missing validation, inconsistency with established patterns.
   - 💡 **suggestion**: Consider changing. Readability or optional improvements.
5. **Don't pile on.** Flag a repeated issue once and list all affected files.
6. **Respect existing style.** The current file's style takes precedence over general guidelines.
7. **Don't flag what CI catches.** Skip issues a compiler, analyzer, formatter, or CI build would catch.
8. **Avoid false positives.** Before flagging an issue:
   - Verify it applies given the full context, not just the diff. Confirm it isn't already handled by a caller, callee, or wrapper.
   - Skip theoretical concerns with negligible real-world probability.
   - If unsure, investigate further or raise it as an explicit low-confidence question.
   - If a pattern seems odd but is consistent with the repository, assume it is intentional.
   - Never assert that something "does not exist" or "is deprecated" from training data alone.
9. **Ensure code suggestions are valid.** Suggested code must compile, be complete, and match the surrounding indentation and style.
10. **Label in-scope vs. follow-up.** Be explicit when a suggestion is a follow-up rather than a blocker.

## Multi-Model Review

When the environment supports launching sub-agents with different models (for example, the `task` tool with a `model` parameter), run the review in parallel across multiple model families to get diverse perspectives. Different models catch different classes of issues. If the environment does not support this, proceed with a single-model review. Each sub-agent re-reads the diff and the surrounding files, so this multiplies the cost of the review by the number of models. It is worth it for a substantial or risky change, not for a one-line fix.

**How to execute (when supported):**
1. Inspect the available model list and select models from 2-3 distinct model families, up to 3 sub-agent models total. If fewer than 2 eligible families are available, use what is available. **Model selection rules:**
   - Pick only from models explicitly listed as available in the environment. Do not guess or assume model names.
   - From each selected family, pick the model with the highest capability tier (prefer "premium" or "standard" over "fast/cheap").
   - Never pick models labeled "mini", "fast", or "cheap" for code review.
   - Do not select the same model that is already running the primary review (your own model).
   - If multiple standard-tier models exist in the same family, pick the one with the highest version number. Prefer "-codex" variants over general-purpose for code review tasks.
2. Launch a sub-agent for each selected model in parallel, giving each the same review prompt: the PR diff, the review rules from this skill, and instructions to produce findings in the severity format defined above.
3. Wait for all agents to complete, then synthesize: deduplicate findings that appear across models, elevate issues flagged by multiple models (higher confidence), and include unique findings from individual models that meet the confidence bar. **Timeout handling:** If a sub-agent has not completed after 10 minutes and you have results from other agents, proceed with the results you have. Note in the output which models contributed.
4. Present a single unified review, noting when an issue was flagged by multiple models. After posting the review, immediately exit. Do not wait for remaining sub-agents or retry a successful post.

---

## Review Output Format

Use this structure when presenting the final review, whether as a PR comment or as output to the user. GitHub's built-in Copilot code review controls its own comment format, so this format applies when you produce the review text yourself.

> 📝 **AI-generated content disclosure:** When posting review content to GitHub under a user's credentials (not a dedicated bot or Copilot app account such as `github-actions[bot]` or `copilot`), you **MUST** include a concise, visible note (for example a `> [!NOTE]` alert) at the bottom indicating the content was AI/Copilot-generated. Skip this only if the user explicitly asks you to omit it.

### Structure

```
## Code Review

**Motivation**: <1-2 sentences on whether the PR is justified and the problem is real>

**Approach**: <1-2 sentences on whether the change takes the right approach>

**Summary**: <✅ LGTM / ⚠️ Needs Human Review / ⚠️ Needs Changes / ❌ Reject>. <2-3 sentence summary of the verdict and key points. If "Needs Human Review," state which findings you are uncertain about and what a human reviewer should focus on.>

---

### Detailed Findings

#### ✅/⚠️/❌/💡 <Category Name> — <Brief description>

<Explanation with specifics. Reference code, line numbers, call paths, etc.>

(Repeat for each finding category. Group related findings under a single heading.)

<!-- AI disclosure note goes below this line when required. -->
```

### Guidelines

- Begin the review body with `## Code Review`, followed by the `**Motivation**:`, `**Approach**:`, and `**Summary**:` fields in that order.
- **Detailed Findings** use emoji-prefixed headers: ✅ verified correct, ⚠️ warning, ❌ error, 💡 suggestion.
- Include cross-cutting analysis when relevant (sibling types, callers, other platforms).
- Assess test quality as its own finding when the PR includes tests.
- Keep the review concise. Every claim should be backed by evidence from the code.

### Verdict Consistency Rules

1. **The verdict must reflect your most severe finding.** Any ⚠️ finding rules out LGTM. Use LGTM only when all findings are ✅ or 💡 and you are confident the change is correct and complete.
2. **When uncertain, escalate to human review.** A false LGTM is far worse than an unnecessary escalation.
3. **Separate code correctness from approach completeness.** Correct code that treats a symptom, masks an error, or fixes one instance but not its siblings is not LGTM.
4. **Classify each ⚠️ and ❌ finding as merge-blocking or advisory.** If any finding would make you uncomfortable merging as-is, the verdict is Needs Changes. If you are unsure, it is Needs Human Review.
5. **Devil's advocate check.** Re-read your ⚠️ findings before finalizing. If any reflects an unresolved concern about approach, scope, or risk, the verdict must reflect it.
