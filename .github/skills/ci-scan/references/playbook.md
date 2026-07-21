# CI scan playbook

This playbook contains the detailed ML.NET CI failure-scanning methodology. The local execution and approval rules in [`../SKILL.md`](../SKILL.md) override any historical workflow-oriented wording below.

# CI Outer-Loop Failure Scanner (machinelearning)

You are a CI triage agent for `dotnet/machinelearning`. Each local run walks the last ~25 completed builds of AzDO definition 167 (`MachineLearning-CI`) on `main`, classifies failures, and drafts a `Known Build Error` issue for every actionable signature so Build Analysis can mark matching PR failures as ignorable after a maintainer approves the write.

To suggest changes, edit this file or use the [`ci-scan-feedback`](../../ci-scan-feedback/SKILL.md) skill. It scores recent scanner artifacts and maintainer feedback against the [Rubric](ci-scan.instructions.md#rubric), then drafts targeted edits locally.

## Read this first

The detailed methodology lives in [`ci-scan.instructions.md`](ci-scan.instructions.md). Read it once at the start of every run and follow its sections by name. Keeping the depth there (data-source endpoints, classification edge cases, occurrence widening, dedup bash, signature specificity, sanitization, the issue template, and the skip-reason vocabulary) keeps this playbook short enough to stay fully in context while every run gets the same rigor.

```bash
mkdir -p /tmp/mlnet-ci-scan/coverage
cat .github/skills/ci-scan/references/ci-scan.instructions.md
```

The [Repository profile](ci-scan.instructions.md#repo-profile) table holds the only repo-specific values; for `dotnet/machinelearning` both `Helix present` and `Build Analysis present` are `yes`, so run the steps flagged **(profile: Helix)** and **(profile: Build Analysis)**.

## Hard rules

These invariants are not delegated to the shared file. Honor them even if a shared section reads more permissively.

1. **Read-only by default.** Draft every GitHub write and show it to the user. Never call `gh issue create` or any other mutating command without explicit approval for the exact write.
2. **Cap 3 issue drafts per run.** On cap, record `cap reached` and stop drafting.
3. **Labels: only `Known Build Error` and `blocking-clean-ci`,** plus `Build` for compile-time breaks. Area triage is owned elsewhere; never apply `area-*` labels here.
4. **Every issue title starts with `[ci-scan] `.**
5. **One signature = one issue.** Dedup on the signature alone, never on `leg|signature` (a single failure surfaces on many legs). Search existing issues per [Search existing issues](ci-scan.instructions.md#search-existing) and the [Same-run dedup cache](ci-scan.instructions.md#dedup-cache) before drafting.
6. **Skip infra noise and weak signatures.** Apply the [stability gate](ci-scan.instructions.md#occurrence-counting): a signature must appear in `>= 2` of the last ~10 builds OR be a block-everyone build break. Otherwise skip with the matching [recognized reason](ci-scan.instructions.md#skip-reasons).
7. **All state under `/tmp/mlnet-ci-scan/`;** each bash call is a fresh subshell.
8. **AzDO REST is anonymous;** stay on `https://dev.azure.com/dnceng-public/public/_apis/build/...`. Follow every rule in [Environment constraints](ci-scan.instructions.md#environment-constraints).
9. **Sanitize every embedded log excerpt** per [Sanitization](ci-scan.instructions.md#sanitization).
10. **Exit immediately on empty build window.** When Step 1 determines no scannable build exists, immediately append one Step 7 coverage line to `/tmp/mlnet-ci-scan/coverage/MachineLearning-CI.txt` in the documented `<sig-short>  <outcome>  <reason-if-skipped>` format as `-  skipped: <reason>`, print the full Step 7 summary table (the `| total-signatures | issues-drafted | reused-existing | skipped-with-reason |` header followed by the `| 0 | 0 | 0 | 1 |` row), report the skip reason, and stop. Do not fetch the AzDO timeline, download task logs, query Helix work items, or execute any step beyond Step 1.

## Step 1 - Select the source build

Fetch the build list and choose `source` + `follow_up` exactly as in [Source build selection and follow-up gate](ci-scan.instructions.md#source-selection).

```bash
url='https://dev.azure.com/dnceng-public/public/_apis/build/builds?definitions=167&branchName=refs/heads/main&statusFilter=completed&resultFilter=succeeded,failed,partiallySucceeded&%24top=25&api-version=7.1'
curl -s "$url" | tee /tmp/mlnet-ci-scan/builds.json | jq -r '.value[] | "\(.id) \(.result) \(.finishTime)"' | head -25
```

If selection yields no scannable build, apply **Hard Rule 10** immediately.

## Step 2 - Walk the timeline

```bash
src_id=<source build id>
url="https://dev.azure.com/dnceng-public/public/_apis/build/builds/${src_id}/timeline?api-version=7.1"
curl -s "$url" | tee /tmp/mlnet-ci-scan/timeline.json | jq '.records | length'
```

Reconstruct `Stage -> Phase -> Job -> Task` via `parentId`; a failed record with non-null `log.id` is a leaf. `MachineLearning-CI` legs follow `MachineLearning-CI (<OS>_<arch> <Configuration>_Build)`.

## Step 3 - Classify each failure

Classify every failed leaf using [Failure classification](ci-scan.instructions.md#classification). ML.NET is `Helix present: yes`, so handle the Helix-routed and dead-letter cases (4 and 5) in addition to build break (1), phase-only (2), and test failure (3). Save the canonical log to `/tmp/mlnet-ci-scan/failure.log` before extracting. Then count occurrences and apply the stability gate per [Occurrence counting and window widening](ci-scan.instructions.md#occurrence-counting), producing `(category, leg, signature)` tuples.

## Step 4 - Follow-up gate

For each stable signature, run the per-signature follow-up gate from [Source build selection and follow-up gate](ci-scan.instructions.md#source-selection). Skip with `signature absent from follow-up build #<id>` or `fix already merged after source build` where they apply.

## Step 5 - Dedup

Run [Search existing issues](ci-scan.instructions.md#search-existing) (cross-run, the Build Analysis dedup path) and then the [Same-run dedup cache](ci-scan.instructions.md#dedup-cache) (signature-only key). On a match draft nothing and record `existing-issue #<n>` or `dup of drafted-issue draft-<n> earlier in this run`.

## Step 6 - Draft the KBE

For each signature that clears every gate and is within the cap, pass the [match-count gate](ci-scan.instructions.md#new-issue-template) and prepare one issue draft using the [New-issue template](ci-scan.instructions.md#new-issue-template). ML.NET is `Build Analysis present: yes`, so use the `## Build Analysis match (literal substring)` heading and propose `Known Build Error` + `blocking-clean-ci` (plus `Build` for build breaks). Append the signature to the dedup cache after drafting.

## Step 7 - Tally

Append one outcome line per signature to `/tmp/mlnet-ci-scan/coverage/MachineLearning-CI.txt`:

```
<sig-short>  <outcome>  <reason-if-skipped>
```

`<outcome>` is one of `drafted-issue draft-<n>`, `existing-issue #<n>`, or `skipped: <reason>` using the [recognized vocabulary](ci-scan.instructions.md#skip-reasons). Follow [Output discipline](ci-scan.instructions.md#output-discipline). At end of run, print this table:

```
| total-signatures | issues-drafted | reused-existing | skipped-with-reason |
```
