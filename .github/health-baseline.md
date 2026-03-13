# Repo Health — Known Baseline

> Last updated: 2026-03-06
> Next review: 2026-04-05
>
> Items listed here are known and accepted by the team. The health check
> workflow will classify these as "baselined" rather than "new" findings.

## Baselined Issues

| # | Title | Reason | Baselined |
|---|-------|--------|-----------|
| #5805 | MKLImports PDB not included with packages | Known packaging tech debt (P0, open since 2021) | 2026-03-06 |
| #7447 | Legacy images need to be updated | Tracked infra work (P1) | 2026-03-06 |
| #6588 | Error for linux-arm/arm64 processor targets | Known platform limitation (P1, open since 2023) | 2026-03-06 |
| #6370 | Exposing the tree for multiclass classification | Feature request backlog (P1, open since 2022) | 2026-03-06 |
| #6353 | CreateEnumerable fails in VS FSI but works in notebook | Known environment issue (P1, open since 2022) | 2026-03-06 |
| #5798 | Add substitutes for IntelMKL methods for non-x86/x64 | Known platform tech debt (P1, open since 2021) | 2026-03-06 |
| #5744 | Memory leak | Known tech debt (P1, open since 2021) | 2026-03-06 |
| #5587 | Migrate to VSTest for all Unit Tests | Tracked migration work (P1, open since 2021) | 2026-03-06 |
| #5569 | OMP Error #15 in Image Classification with AutoML | Known OMP conflict (P1, open since 2020) | 2026-03-06 |
| #5566 | SVMLightLoader fails above 128 dense rows | Known loader limitation (P1, open since 2020) | 2026-03-06 |
| #4210 | TensorFlow scoring sample should pass empty dataview | Known sample issue (P1, open since 2019) | 2026-03-06 |
| #4145 | mlnet generated projects don't include cs files | Known CLI tooling issue (P1, open since 2019) | 2026-03-06 |
| #3988 | CustomMappingEstimator save/load issues | Known serialization limitation (P1, open since 2019) | 2026-03-06 |
| #3766 | GetFeatureWeights categorical splits support | Feature gap (P1, open since 2019) | 2026-03-06 |
| #3701 | How to inspect OneVersusAll models | API gap (P1, open since 2019) | 2026-03-06 |
| #3684 | AutoML: Allow serialized IDataView input | Feature request (P1, open since 2019) | 2026-03-06 |
| #2774 | OneHotEncoding Bin mode wrong dimension | Known bug (P1, open since 2019) | 2026-03-06 |
| #2467 | OvaModelParameters not strongly-typed | API design debt (P1, open since 2019) | 2026-03-06 |
| #2185 | KeyToVectorMappingEstimator schema conditions | Known bug (P1, open since 2019) | 2026-03-06 |
| #2167 | GAM Trainer models depend on feature flocks | Known design issue (P1, open since 2019) | 2026-03-06 |
| #1990 | Vector length 0 as missing value is problematic | Known data type issue (P1, open since 2019) | 2026-03-06 |
| #1004 | Consistency issue with LdaTransform | Known transform bug (P1, open since 2018) | 2026-03-06 |
| #765 | Add Reshape Transform | Feature request (P1, open since 2018) | 2026-03-06 |
| #590 | Cannot combine OneVersusAll with FFMM | Known trainer limitation (P1, open since 2018) | 2026-03-06 |

## Baselined PRs

| # | Title | Reason | Baselined |
|---|-------|--------|-----------|
| #7416 | Update TorchSharp to 0.105.0 | Dependency update in progress (open since Mar 2025) | 2026-03-06 |
| #7406 | [GenAI] Use BitsAndBytes for 4bit quantization | Feature work in progress (open since Mar 2025) | 2026-03-06 |
| #7094 | Add support for Apache.Arrow.Types.Decimal128Type | Community contribution awaiting review (open since Mar 2024) | 2026-03-06 |
| #6749 | Update Projects to .NET 8 in MLNET 4.0 Branch | Long-running migration (open since Jun 2023) | 2026-03-06 |
| #6664 | (WIP) Generic DataFrame Math | Work in progress (open since May 2023) | 2026-03-06 |
| #6449 | Add DataViewSchema overloads to ConvertToOnnx | Community contribution awaiting review (open since Nov 2022) | 2026-03-06 |

## Baselined Pipeline Failures

| Pipeline | Failure | Reason | Baselined |
|----------|---------|--------|-----------|
| *(none detected — AzDO pipeline status requires PAT to query)* | — | Check after first run with AZDO_PAT configured | 2026-03-06 |

## Review Policy

- Re-evaluate this file every 30 days
- Remove items when they are resolved
- Add date when baselining new items
- If an item has been baselined > 90 days, escalate it
