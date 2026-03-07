# AI-Native Development Infrastructure

This document describes the AI-native development infrastructure added to the ML.NET repository. These files teach GitHub Copilot (and other AI agents) how to navigate, build, test, and contribute to this codebase.

## What Was Created

### Instructions (teach AI your repo)

| File | Type | Purpose |
|------|------|---------|
| `.github/copilot-instructions.md` | Generated | Global Copilot context — repo overview, build/test commands, conventions, project structure |
| `.github/instructions/tests.instructions.md` | Generated | Test-specific guidance — xUnit patterns, base classes, assertion styles, naming |
| `.github/instructions/genai.instructions.md` | Generated | GenAI component guidance — LLaMA/Phi/Mistral via TorchSharp, Semantic Kernel |
| `.github/instructions/tokenizers.instructions.md` | Generated | Tokenizer guidance — BPE/WordPiece/SentencePiece, data packages |

### Agents (multi-step AI workflows)

| File | Type | Purpose |
|------|------|---------|
| `.github/agents/pr.md` | Configured | 4-phase PR workflow: Pre-Flight → Gate → Fix → Report |
| `.github/agents/pr/post-gate.md` | Configured | Multi-model fix exploration (Phase 3-4) |
| `.github/agents/pr/SHARED-RULES.md` | Configured | Shared rules and model configuration |
| `.github/agents/write-tests-agent.md` | Configured | Test writing dispatcher following xUnit conventions |
| `.github/agents/learn-from-pr.md` | Configured | Self-improvement — extract lessons from PRs |

### Skills (focused AI capabilities)

| File | Type | Purpose |
|------|------|---------|
| `.github/skills/pr-build-status/SKILL.md` | Configured | Read Azure Pipelines + GitHub Actions CI results |
| `.github/skills/try-fix/SKILL.md` | Configured | Single-shot fix → test → report cycle |
| `.github/skills/run-tests/SKILL.md` | Configured | Build and run tests with filtering |
| `.github/skills/verify-tests-fail/SKILL.md` | Configured | Prove tests catch bugs (fail without fix, pass with) |
| `.github/skills/pr-finalize/SKILL.md` | Universal | Verify PR title/description match implementation |
| `.github/skills/issue-triage/SKILL.md` | Universal | Triage open issues by milestone/priority |
| `.github/skills/find-reviewable-pr/SKILL.md` | Universal | Find PRs needing review |
| `.github/skills/learn-from-pr/SKILL.md` | Universal | Analyze PRs for lessons learned |
| `.github/skills/ai-summary-comment/SKILL.md` | Universal | Post unified progress comments on PRs |

### Workflows (automated GitHub Actions)

| File | Type | Purpose |
|------|------|---------|
| `.github/workflows/copilot-setup-steps.yml` | Pre-existing | Remote Copilot Coding Agent build environment |
| `.github/workflows/find-similar-issues.yml` | Universal | AI duplicate detection on new issues |
| `.github/workflows/inclusive-heat-sensor.yml` | Universal | Detects heated language in comments |

### Prompts

| File | Type | Purpose |
|------|------|---------|
| `.github/prompts/release-notes.prompt.md` | Configured | Generate classified release notes between commits |

## File Types

- **Generated** — Produced by analyzing the ML.NET repo's specific structure and conventions
- **Configured** — Template filled with ML.NET's build/test commands and project structure
- **Universal** — Works on any GitHub repo unchanged
- **Pre-existing** — Already present in the repo before onboarding

## CI Feedback Loop

The CI feedback loop enables AI agents to iterate on failures:

```
Agent writes code → Push → CI runs → Agent reads results → Agent fixes → Repeat
```

Three components make this work:
1. **`copilot-setup-steps.yml`** — Remote Copilot Coding Agent can build the repo (pre-existing)
2. **`pr-build-status` skill** — Agent reads Azure Pipelines/GitHub Actions results
3. **Build/test commands in instructions** — Agent knows how to build and test locally

## Next Steps

1. **Review `copilot-instructions.md`** — It's the highest-impact file. Verify it captures your team's conventions accurately.
2. **Review scoped instructions** — Check that glob patterns in `tests.instructions.md`, `genai.instructions.md`, and `tokenizers.instructions.md` match your project structure.
3. **Commit**:
   ```bash
   git add .github/
   git commit -m "Add AI-native development infrastructure"
   ```
4. **Test it** — Open a PR and ask Copilot to review it.
5. **Improve** — After your first PR with AI involvement, use `learn-from-pr` to refine the instructions based on real experience.
