# PR Agent: Shared Rules

Rules that apply across all PR agent phases. Referenced by `pr.md` and `post-gate.md`.

---

## Multi-Model Configuration

Phase 3 uses these AI models for try-fix exploration (run **SEQUENTIALLY**):

| Order | Model |
|-------|-------|
| 1 | `claude-sonnet-4-5` |
| 2 | `gpt-4.1` |
| 3 | `gemini-2.5-pro` |

**Note:** The `model` parameter is passed to the `task` tool's agent invocation. Each model runs try-fix independently.

**⚠️ SEQUENTIAL ONLY**: try-fix runs modify the same files and use the same build/test environment. Never run in parallel.

### Recommended Default Models

If no specific models are configured, use a diverse set across providers:

| Order | Model | Why |
|-------|-------|-----|
| 1 | `claude-sonnet-4-5` | Strong code reasoning |
| 2 | `gpt-4.1` | Fast, different perspective |
| 3 | `gemini-2.5-pro` | Different training data |

Adjust based on available models and budget. More models = more fix diversity = better chance of finding optimal solution.

---

## Phase Completion Protocol

**Before changing ANY phase status to ✅ COMPLETE:**

1. Review the phase checklist
2. Verify all required items are addressed
3. Then mark the phase as ✅ COMPLETE

**Rule:** Status ✅ means "work complete and verified", not "I finished thinking about it."

---

## Stop on Environment Blockers

If you encounter a blocker that prevents completing a phase:

1. **Try ONE retry** (install missing tool, rebuild, etc.)
2. **If still blocked after one retry**, skip the blocked phase and continue
3. **Document what was skipped and why** in the Report phase
4. **Always prefer continuing with partial results** over stopping completely

| Blocker Type | Max Retries | Then Do |
|--------------|-------------|---------|
| Missing tool/dependency | 1 install attempt | Skip phase, continue |
| Server errors (500, timeout) | 1 retry | Skip phase, continue |
| Build failures in try-fix | 2 attempts | Skip remaining models, proceed to Report |
| Configuration issues | 1 fix attempt | Skip phase, continue |

---

## No Direct Git State Changes

The agent should not run git commands that change branch state during PR review. Use read-only commands:

- ✅ `gh pr diff`, `gh pr view`, `gh issue view`
- ❌ `git checkout`, `git switch`, `git stash`, `git reset`

Exception: `git checkout HEAD -- .` and `git clean -fd` are allowed for cleanup between try-fix attempts.
