# CI scan shared instructions

Reusable methodology shared by [`ci-scan`](../SKILL.md) and
[`ci-scan-feedback`](../../ci-scan-feedback/SKILL.md). The scanner reads this file
once at the start of a run and follows the named sections by reference; the feedback skill
reads only the [Recognized skip-reason vocabulary](#skip-reasons) and
[Rubric](#rubric) sections. Keeping the heavy, slow-changing detail here keeps each agent prompt
short enough to stay fully in context (progressive disclosure) while still giving the agent the
same depth on every run.

Read it from `.github/skills/ci-scan/references/ci-scan.instructions.md` at the start. Each
section below has an HTML anchor so a prompt can cite it precisely (for example
"follow [Same-run dedup cache](#dedup-cache)").

## Table of contents

- [Repository profile](#repo-profile)
- [Environment constraints](#environment-constraints)
- [Data sources](#data-sources)
- [Source build selection and follow-up gate](#source-selection)
- [Failure classification](#classification)
- [Occurrence counting and window widening](#occurrence-counting)
- [Search existing issues](#search-existing)
- [Same-run dedup cache](#dedup-cache)
- [Signature specificity](#signature-specificity)
- [Bad vs good signatures](#bad-vs-good)
- [Sanitization](#sanitization)
- [New-issue template](#new-issue-template)
- [Recognized skip-reason vocabulary](#skip-reasons)
- [Rubric](#rubric)
- [Output discipline](#output-discipline)

<a id="repo-profile"></a>
## Repository profile

These are the only repo-specific values the methodology depends on. The TorchSharp copy of this
file differs from this one only in this table and in the profile-gated lines flagged inline; every
other section is identical so the two scanners behave the same way.

| Key | Value |
|---|---|
| Repo | `dotnet/machinelearning` |
| AzDO org / project | `dnceng-public` / `public` |
| Pipeline | `MachineLearning-CI`, definition id **167** |
| Branch scanned | `refs/heads/main` |
| Helix present | **yes** (`Send to Helix` legs route xunit work items through `helix.dot.net`) |
| Build Analysis present | **yes** (Arcade attaches `Build_Analysis_KnownIssues_v1`; `Known Build Error` issues feed it) |
| Issue model | `Known Build Error` KBE issues consumed by Build Analysis |
| Issue labels | `Known Build Error`, `blocking-clean-ci`, and `Build` for compile-time breaks |
| Title prefix | `[ci-scan] ` |

`Helix present` and `Build Analysis present` gate two pieces of behavior, flagged inline below as
**(profile: Helix)** and **(profile: Build Analysis)**. When a key is `no`, skip the gated step.

<a id="environment-constraints"></a>
## Environment constraints

Use these patterns consistently during local execution.

- **Pre-bind every URL that contains `?` or `&` to a shell variable on its own line, then
  `curl -s "$url"`.** Inline query strings are rejected as "Permission denied" even when quoted,
  because the tool approver treats `?`/`&` as interactive prompts.

  ```bash
  url='https://dev.azure.com/dnceng-public/public/_apis/build/builds?definitions=167&branchName=refs/heads/main&statusFilter=completed&resultFilter=succeeded,failed,partiallySucceeded&%24top=25&api-version=7.1'
  curl -s "$url" | tee /tmp/mlnet-ci-scan/builds.json | jq -r '.value[0] | "\(.id) \(.result)"'
  ```

- **OData `$top`/`$skip` must be percent-encoded as `%24top` / `%24skip`** inside the URL.
- **Prefer `tee` for evidence files** so the command output and saved artifact stay visible.
- **Each bash call is a fresh subshell.** Nothing persists except files under `/tmp/mlnet-ci-scan/`.
  Write intermediate state there (`mkdir -p /tmp/mlnet-ci-scan/coverage` first).
- **AzDO REST is anonymous.** Never add auth headers; stay on the `dnceng-public/public` host.

<a id="data-sources"></a>
## Data sources

- **AzDO REST** - `https://dev.azure.com/dnceng-public/public/_apis/build/...`, anonymous.
  - List builds: `?definitions=167&branchName=refs/heads/main&statusFilter=completed&resultFilter=succeeded,failed,partiallySucceeded&%24top=25&api-version=7.1`. The list is sorted DESC by queue time, so the later-in-wall-clock `follow_up` appears in the array *before* the older `source` build it follows.
  - Timeline: `/builds/{id}/timeline?api-version=7.1` returns a flat `records[]`; reconstruct the `Stage -> Phase -> Job -> Task` tree via `parentId`. A failed record with a non-null `log.id` is a leaf worth reading.
  - Task log: each leaf record exposes `log.url`; `curl -s "$log_url"` returns plain text.
- **Helix REST** **(profile: Helix)** - `https://helix.dot.net/api/jobs/{jobId}/workitems?api-version=2019-06-17`. Each work item has `Name`, `State`, `ExitCode`, `ConsoleOutputUri`. A work item failed when `ExitCode != 0` or `State == "Failed"`. Extract `{jobId}` from a `Send to Helix` task log line `Sent Helix Job: <GUID>`.
- **Build Analysis attachment** **(profile: Build Analysis)**, best-effort dedupe - `https://dev.azure.com/dnceng-public/public/_apis/build/builds/{id}/attachments/Build_Analysis_KnownIssues_v1?api-version=7.1`. A `404` means none is attached; treat that as "no known-issue match" and continue, never as an error.

<a id="source-selection"></a>
## Source build selection and follow-up gate

Pick the build to scan, then prove the failure is still live before filing.

1. **Select `source`.** From the build list, pick the most recent build with
   `result in {failed, partiallySucceeded}` that has at least one COMPLETED build with a strictly
   later `finishTime`. That later build is the `follow_up` anchor. Because the list is sorted DESC
   by queue time, `follow_up` sits in the array *before* `source`.
2. **Skip reasons at selection time:**
   - `source.finishTime` older than 14 days: record `stale build window (>14d)`.
   - No `follow_up` (the source is the absolute latest completed build): record
     `no follow-up build yet, defer to next run`.
   - No qualifying failed build in the last 7 days: record `no failed build in 7d`.
3. **Follow-up gate (run per signature, after classification).** Inspect `follow_up`:
   - `follow_up.result == succeeded`, or it failed/partiallySucceeded WITHOUT this signature:
     record `signature absent from follow-up build #<id>` and draft nothing.
   - `follow_up` contains the signature: proceed to filing.
   - For build breaks, also search merged PRs touching the failing source file (or citing the
     error code) with `merged:>=<source.finishTime>`. On a match record
     `fix already merged after source build` and draft nothing.

<a id="classification"></a>
## Failure classification

Classification decides WHERE to read the signature text from, not whether to file. Save the
canonical failing log to `/tmp/mlnet-ci-scan/failure.log` before extracting, because the
[match-count gate](#new-issue-template) greps it for the verbatim signature.

```bash
log_url='<console URL from the AzDO task log or Helix work item>'
curl -s "$log_url" | tee /tmp/mlnet-ci-scan/failure.log | tail -200
```

1. **Build break.** Failing task is a compile/restore/native step (`Build`, `Restore`, `Compile`,
   `Configure CMake`, `Build native`) and the test task is absent or `skipped`. Read the signature
   from the failing compile log: the `CSxxxx` diagnostic, linker error, native compiler error, or
   cmake error line. Build breaks block every leg, so they may be drafted on first sight (see the
   [stability gate](#occurrence-counting)). Apply the `Build` label.
2. **Phase/stage-only failure with no failed job underneath.** A compile break aggregated at the
   phase level (no leaf Job record). Open the Phase log plus the latest log of any non-succeeded
   child Task and treat it as a build break.
3. **Test failure.** Failing task is the test run (`Run Tests`, `dotnet test`, an `xunit` task).
   Locate the first `[FAIL]` / `Failed:` / `Assert.*` line. The signature is the test method FQN
   plus the first line of the assertion/exception message.
4. **Helix-routed test failure** **(profile: Helix).** `Send to Helix` succeeded but the Job still
   failed. Pull the Helix job id from its log, query work items, fetch the failing work item's
   console log, and locate the `[FAIL]` line there instead of in the AzDO task log.
5. **Dead-lettered Helix work item** **(profile: Helix).** Console URI contains
   `helix-workitem-deadletter`. Extract the `[FAIL]` line if present; if absent there is no stable
   signature, so skip emission and record `infra noise - no stable signature`.
6. **Infra-shaped job failure.** `Initialize job` failed, agent disconnect, `Pool is offline`,
   queue-capacity timeout, transient network. No stable signature: skip emission and record
   `infra noise - no stable signature`.

For each (1)/(2)/(3)/(4) signature compute the tuple `(category, leg, signature)` and proceed to
[occurrence counting](#occurrence-counting).

<a id="occurrence-counting"></a>
## Occurrence counting and window widening

Count how many of the last ~10 completed builds of definition 167 contain the signature. Multiple
legs, retries, or work items of the SAME build id count as ONE occurrence, never two.

**Stability gate.** A signature is stable when it appears in `>= 2` distinct builds in the window,
OR it is a build break that fails every leg of the source build (block-everyone severity worth
filing on first sight). A one-off that appears in a single build is not stable: record
`< 2 occurrences and not blocking` and let the next run revisit.

**Window widening.** If the signature appears in *every* sampled build (100% in the ~10-build
window), the true first occurrence likely predates the window. Widen the build list
(`&%24skip=10`, `&%24skip=20`, ...) up to ~40 additional builds and stop as soon as you find a
build where the signature is absent. Report the build immediately after that gap as
`First build it occurred`. If you hit the cap without finding a gap, set `First build it occurred`
to the oldest build scanned and add the note `Persistent across the entire scanned window; true
origin may predate <oldest-build-date>.`

<a id="search-existing"></a>
## Search existing issues

Before filing, search for an already-open issue covering the same signature. This is the primary
dedup mechanism: with Build Analysis present it avoids a redundant KBE, and where Build Analysis is
absent it is the ONLY cross-run dedup. Search the `[ci-scan]` issue space (the issue model named in
the [profile](#repo-profile)):

```bash
sig_short='<most distinctive sanitized substring of the signature, <= 80 chars>'
gh issue list --repo dotnet/machinelearning --state open --label "Known Build Error" \
  --search "$sig_short in:title,body" --json number,title,url | tee /tmp/mlnet-ci-scan/existing.json
```

The same failure can be recorded in different wordings, so do not conclude "no existing issue" from
one query:

- Search the most distinctive single substring (the assertion stem or `CSxxxx` + symbol), not the
  whole joined line.
- If the first search misses, try a second distinctive substring (for example the test FQN alone).
- Include closed issues in a second pass (`--state all`) when a recent closure may be the right
  tracker; a freshly closed "fixed" issue means do not re-file unless the failure clearly recurs
  after the fix.

On a confirmed match record `existing-issue #<n>` and draft nothing. Verify the candidate actually
matches the same test/family AND the same failing line before trusting it; a coincidental substring
hit is not a match. When two candidates look equally plausible and you cannot disambiguate, record
`existing-issue #<n>` for the closest and note the ambiguity in the tally rather than filing a
duplicate.

<a id="dedup-cache"></a>
## Same-run dedup cache

A single failure surfaces on many legs of the same build, so dedupe on the signature alone, NOT on
`leg|signature` (that would draft one issue per leg). Cache drafted signatures in
`/tmp/mlnet-ci-scan/drafted.tsv` as `<key>\t<draft_id>` where `key = <signature_norm>` and
`<signature_norm>` is the signature with tab/CR/newline stripped (raw signatures are copied
verbatim from logs and may carry whitespace that would corrupt the TSV).

```bash
signature_norm=$(printf '%s' "<signature>" | tr -d '\t\n\r')
test -f /tmp/mlnet-ci-scan/drafted.tsv && cut -f1 /tmp/mlnet-ci-scan/drafted.tsv | grep -Fxq -- "$signature_norm"   # dup if exit 0
printf '%s\t%s\n' "$signature_norm" "draft-<n>" >> /tmp/mlnet-ci-scan/drafted.tsv                                  # append after every draft
```

On a cache hit record `dup of drafted-issue draft-<n> earlier in this run` and stop for that
signature. Append the key after every issue draft. Run the [specificity](#signature-specificity)
check before appending: a signature too generic to be specific must be rejected up front, never
cached, because a broad key collapses unrelated failures into one issue.

<a id="signature-specificity"></a>
## Signature specificity

A signature must be specific enough that Build Analysis (or a human searching) matches the real
failure and nothing else. Prefer the most distinctive stable substring of the failing line:

- Keep the test method FQN plus the assertion/exception stem.
- Keep `CSxxxx` / linker / cmake error codes and the offending symbol.
- Strip everything volatile per [Sanitization](#sanitization): absolute paths, GUIDs, machine
  names, timestamps, ports, PIDs, durations, and run-specific numeric ids.
- Never use a bare exit code, a generic exception type with no message, or a phrase that also
  appears on `[PASS]` / `[SKIP]` lines of the same log.

<a id="bad-vs-good"></a>
## Bad vs good signatures

| Failing line | Bad signature (too broad) | Good signature (specific + stable) |
|---|---|---|
| `[FAIL] Microsoft.ML.Tests.OnnxTests.SimpleTest : Assert.Equal() Failure: Expected 0.81, got 0.79` | `Assert.Equal() Failure` | `OnnxTests.SimpleTest : Assert.Equal() Failure: Expected 0.81` |
| `D:\a\1\s\src\Foo.cs(120,5): error CS0246: The type or namespace name 'Bar' could not be found` | `error CS0246` | `Foo.cs: error CS0246: The type or namespace name 'Bar' could not be found` |
| `Process terminated. Exit code 134. Stack: ...` on one leg only | `Exit code 134` | `Process terminated. Exit code 134.` only if the same stack frame recurs; otherwise treat as infra noise |
| `Test run failed` summary line | `Test run failed` | (reject - find the underlying `[FAIL]` line instead) |

<a id="sanitization"></a>
## Sanitization

Sanitize every log excerpt before embedding it in an issue body. Replace, do not delete, so the
line stays readable:

- Absolute paths -> repo-relative (`D:\a\1\s\src\Foo.cs` -> `src/Foo.cs`).
- GUIDs, Helix job/work-item ids, build numbers -> `<id>`.
- Machine/agent names, IP addresses, ports -> `<host>` / `<port>`.
- Timestamps, durations, PIDs -> `<time>` / `<pid>`.
- Keep the diagnostic code, symbol names, assertion text, and `[FAIL]` marker verbatim - those are
  the signature.

<a id="new-issue-template"></a>
## New-issue template

Prepare one issue draft per stable signature when every gate passes and the per-run cap allows.

**Match-count gate.** Before drafting, confirm the literal match block is a verbatim substring of
`/tmp/mlnet-ci-scan/failure.log`:

```bash
grep -F -c -- "<literal match substring>" /tmp/mlnet-ci-scan/failure.log   # must be >= 1
```

If the count is `0`, do not draft; record `signature did not match failure.log (N=0)` instead.

````markdown
## Signature

`<one-line normalized failure>`

## Failing line (raw)

```
<one [FAIL] or compile-error line, sanitized>
```

## Match signature (literal substring)

```
<exact substring Build Analysis should match on - the literal verified by the match-count gate>
```

## Category

<Build break | Test failure>

## Affected legs (in the source build)

- `<leg display name>` (task log: `<url>`)
- ...

## First build it occurred

- Build: `<azdo build url>`
- Finished: `<UTC timestamp>`
- Commit: `<sha>`
- Occurrences in the scanned window: `<n>`
- Computed within the scanned window; may not be the true origin.

## Reasoning

<why this is a real failure and not flake; cite the source line and the occurrence count>

---

Prepared with the repository-local [`ci-scan`](https://github.com/dotnet/machinelearning/blob/main/.github/skills/ci-scan/SKILL.md)
skill, which scans `dnceng-public` definition 167 on `main` and converts stable failures into
maintainer-approved `Known Build Error` issues. Comment here to flag a false positive or add context;
the local [`ci-scan-feedback`](https://github.com/dotnet/machinelearning/blob/main/.github/skills/ci-scan-feedback/SKILL.md)
skill can use that feedback to propose scanner improvements.
````

**(profile: Build Analysis)** Rename the `## Match signature (literal substring)` heading to
`## Build Analysis match (literal substring)` so it reads naturally for KBE consumers; the content
is identical. Apply labels `Known Build Error` and `blocking-clean-ci`, plus `Build` for build
breaks. When this repo has no Build Analysis, keep the `## Match signature` heading and rely on the
manual dedup search instead.

<a id="skip-reasons"></a>
## Recognized skip-reason vocabulary

Every skipped signature MUST carry a reason, and the reason SHOULD reuse one of these phrasings so
[`ci-scan-feedback`](../../ci-scan-feedback/SKILL.md) can aggregate the tally stably. The list is not
exhaustive; new reasons should follow the same short, lower-case shape.

- `cap reached`
- `< 2 occurrences and not blocking`
- `infra noise - no stable signature`
- `signature absent from follow-up build #<id>`
- `stale build window (>14d)`
- `no follow-up build yet, defer to next run`
- `no failed build in 7d`
- `fix already merged after source build`
- `dup of drafted-issue draft-<n> earlier in this run`
- `existing-issue #<n>`
- `suspected infra outage`
- `signature did not match failure.log (N=<count>)`
- `weak signature`

<a id="rubric"></a>
## Rubric

[`ci-scan-feedback`](../../ci-scan-feedback/SKILL.md) scores each `[ci-scan]` issue against these
criteria. A failing criterion is a candidate signal for a prompt edit.

- **Title scoped to a single failure shape** - a test FQN + assertion stem or a single compile
  error, not a list of legs.
- **Classification matches the failure** - KBE-eligible test/hang/build break carries the right
  labels; infra noise should never have been filed at all.
- **Match block is specific** - the literal-substring block is a stable substring of the real
  failing line per [Signature specificity](#signature-specificity), not a bare method name,
  generic exception, exit code, or a phrase shared with `[PASS]`/`[SKIP]` lines.
- **Occurrence count is honest** - the figure is consistent with the cited build and the
  [occurrence rules](#occurrence-counting).
- **Follow-up gate respected** - the issue cites a real failing build and was not filed for a
  signature already absent from the follow-up build.

<a id="output-discipline"></a>
## Output discipline

- Each definition gets exactly one walk-through per run. Do not revisit.
- One signature = one outcome line in `/tmp/mlnet-ci-scan/coverage/<pipeline>.txt`.
- No meta / aggregate / outage issues. Every issue is keyed to a single `(category, signature)`.
- Do not add `area-*` labels; area triage is owned elsewhere.
- Do not comment on existing KBEs; Build Analysis tracks occurrence counts in the issue body.
- Do not propose alternative scanner designs in issue bodies. Change the skill or use a maintainer
  comment as input to the feedback skill.
- The final response MUST include the Step 7 summary table.
