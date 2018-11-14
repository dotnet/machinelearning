// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    public abstract partial class EvaluatorBase<TAgg>
    {
        internal abstract class AucAggregatorBase
        {
            protected Single Score;
            protected Single Label;

            public void ProcessRow(Single label, Single score, Single weight = 1)
            {
                Label = label;
                Score = score;
                ProcessRowCore(weight);
            }

            protected abstract void ProcessRowCore(Single weight);

            public abstract void Finish();

            public abstract Double ComputeWeightedAuc(out Double unweighted);
        }

        internal abstract class AucAggregatorBase<T> : AucAggregatorBase
        {
            private readonly ReservoirSamplerWithoutReplacement<T> _posReservoir;
            private readonly ReservoirSamplerWithoutReplacement<T> _negReservoir;

            private readonly List<T> _posExamples;
            private readonly List<T> _negExamples;

            protected IEnumerable<T> PosSample;
            protected IEnumerable<T> NegSample;

            protected AucAggregatorBase(IRandom rand, int reservoirSize)
            {
                Contracts.Assert(reservoirSize >= -1);

                ValueGetter<T> sampleGetter = GetSampleGetter();
                if (reservoirSize > 0)
                {
                    _posReservoir = new ReservoirSamplerWithoutReplacement<T>(rand, reservoirSize, sampleGetter);
                    _negReservoir = new ReservoirSamplerWithoutReplacement<T>(rand, reservoirSize, sampleGetter);
                }
                else if (reservoirSize == -1)
                {
                    _posExamples = new List<T>();
                    _negExamples = new List<T>();
                }
            }

            protected abstract ValueGetter<T> GetSampleGetter();

            protected override void ProcessRowCore(Single weight)
            {
                if (_posReservoir == null && _posExamples == null)
                    return;

                if (_posReservoir != null)
                {
                    if (Label > 0)
                        _posReservoir.Sample();
                    else
                        _negReservoir.Sample();
                }
                else if (Label > 0)
                    AddExample(_posExamples);
                else
                    AddExample(_negExamples);
            }

            protected abstract void AddExample(List<T> examples);

            public override void Finish()
            {
                if (_posReservoir == null && _posExamples == null)
                    return;

                if (_posReservoir != null)
                {
                    Contracts.Assert(_negReservoir != null);
                    _posReservoir.Lock();
                    PosSample = _posReservoir.GetSample();
                    _negReservoir.Lock();
                    NegSample = _negReservoir.GetSample();
                }
                else
                {
                    Contracts.AssertValue(_posExamples);
                    Contracts.AssertValue(_negExamples);
                    PosSample = _posExamples;
                    NegSample = _negExamples;
                }
            }

            public override Double ComputeWeightedAuc(out Double unweighted)
            {
                if (_posReservoir == null && _posExamples == null)
                {
                    unweighted = 0;
                    return 0;
                }

                Contracts.Check(PosSample != null && NegSample != null, "Must call Finish() before computing AUC");
                return ComputeWeightedAucCore(out unweighted);
            }

            protected abstract Double ComputeWeightedAucCore(out double unweighted);
        }

        internal sealed class UnweightedAucAggregator : AucAggregatorBase<Single>
        {
            public UnweightedAucAggregator(IRandom rand, int reservoirSize)
                : base(rand, reservoirSize)
            {
            }

            protected override Double ComputeWeightedAucCore(out Double unweighted)
            {
                Contracts.AssertValue(PosSample);
                Contracts.AssertValue(NegSample);

                using (var posSorted = PosSample.OrderByDescending(x => x).GetEnumerator())
                using (var negSorted = NegSample.OrderByDescending(x => x).GetEnumerator())
                {
                    var cumPosWeight = 0.0;
                    var cumNegWeight = 0.0;
                    var cumAuc = 0.0;
                    var hasMorePos = posSorted.MoveNext();
                    var hasMoreNeg = negSorted.MoveNext();
                    var curScorePosWeight = 0.0;
                    var posScore = 0.0;
                    while (hasMorePos && hasMoreNeg)
                    {
                        posScore = posSorted.Current;
                        var negScore = negSorted.Current;
                        if (posScore > negScore)
                        {
                            cumPosWeight++;
                            hasMorePos = posSorted.MoveNext();
                        }
                        else if (posScore < negScore)
                        {
                            cumAuc += cumPosWeight;
                            cumNegWeight++;
                            hasMoreNeg = negSorted.MoveNext();
                        }
                        else
                        {
                            curScorePosWeight = 0.0;
                            var curScoreNegWeight = 0.0;
                            var score = posScore;
                            while (score == posScore)
                            {
                                curScorePosWeight++;
                                hasMorePos = posSorted.MoveNext();
                                if (!hasMorePos)
                                    break;
                                posScore = posSorted.Current;
                            }
                            while (score == negScore)
                            {
                                curScoreNegWeight++;
                                hasMoreNeg = negSorted.MoveNext();
                                if (!hasMoreNeg)
                                    break;
                                negScore = negSorted.Current;
                            }
                            cumAuc += cumPosWeight * curScoreNegWeight;
                            cumAuc += 0.5 * curScorePosWeight * curScoreNegWeight;
                            cumPosWeight += curScorePosWeight;
                            cumNegWeight += curScoreNegWeight;
                        }
                    }
                    while (hasMorePos)
                    {
                        cumPosWeight++;
                        hasMorePos = posSorted.MoveNext();
                    }
                    while (hasMoreNeg)
                    {
                        cumAuc += cumPosWeight;
                        if (posScore == negSorted.Current)
                            cumAuc -= 0.5 * curScorePosWeight;
                        cumNegWeight++;
                        hasMoreNeg = negSorted.MoveNext();
                    }
                    return unweighted = cumAuc / (cumPosWeight * cumNegWeight);
                }
            }

            protected override ValueGetter<Single> GetSampleGetter()
            {
                return (ref Single dst) => dst = Score;
            }

            protected override void AddExample(List<Single> examples)
            {
                Contracts.AssertValue(examples);
                examples.Add(Score);
            }
        }

        internal sealed class WeightedAucAggregator : AucAggregatorBase<WeightedAucAggregator.AucInfo>
        {
            public struct AucInfo
            {
                public Single Score;
                public Single Weight;
            }

            private Single _weight;

            public WeightedAucAggregator(IRandom rand, int reservoirSize)
                : base(rand, reservoirSize)
            {
            }

            protected override Double ComputeWeightedAucCore(out Double unweighted)
            {
                Contracts.AssertValue(PosSample);
                Contracts.AssertValue(NegSample);

                using (var posSorted = PosSample.OrderByDescending(x => x.Score).GetEnumerator())
                using (var negSorted = NegSample.OrderByDescending(x => x.Score).GetEnumerator())
                {
                    var cumPosCount = 0L;
                    var cumNegCount = 0L;
                    var cumPosWeight = 0.0;
                    var cumNegWeight = 0.0;
                    var cumWeightedAuc = 0.0;
                    var cumAuc = 0.0;
                    var hasMorePos = posSorted.MoveNext();
                    var hasMoreNeg = negSorted.MoveNext();
                    var curScorePosWeight = 0.0;
                    var curScorePosCount = 0L;
                    var posScore = 0.0;
                    while (hasMorePos && hasMoreNeg)
                    {
                        posScore = posSorted.Current.Score;
                        var negScore = negSorted.Current.Score;
                        if (posScore > negScore)
                        {
                            var weight = posSorted.Current.Weight;
                            cumPosWeight += weight;
                            cumPosCount++;
                            hasMorePos = posSorted.MoveNext();
                        }
                        else if (posScore < negScore)
                        {
                            var weight = negSorted.Current.Weight;
                            cumWeightedAuc += cumPosWeight * weight;
                            cumAuc += cumPosCount;
                            cumNegWeight += weight;
                            cumNegCount++;
                            hasMoreNeg = negSorted.MoveNext();
                        }
                        else
                        {
                            curScorePosWeight = 0.0;
                            curScorePosCount = 0;
                            var curScoreNegWeight = 0.0;
                            var curScoreNegCount = 0L;
                            var score = posScore;
                            while (score == posScore)
                            {
                                var posWeight = posSorted.Current.Weight;
                                curScorePosWeight += posWeight;
                                curScorePosCount++;
                                hasMorePos = posSorted.MoveNext();
                                if (!hasMorePos)
                                    break;
                                posScore = posSorted.Current.Score;
                            }
                            while (score == negScore)
                            {
                                var negWeight = negSorted.Current.Weight;
                                curScoreNegWeight += negWeight;
                                curScoreNegCount++;
                                hasMoreNeg = negSorted.MoveNext();
                                if (!hasMoreNeg)
                                    break;
                                negScore = negSorted.Current.Score;
                            }
                            cumWeightedAuc += cumPosWeight * curScoreNegWeight;
                            cumWeightedAuc += 0.5 * curScorePosWeight * curScoreNegWeight;
                            cumPosWeight += curScorePosWeight;
                            cumNegWeight += curScoreNegWeight;
                            cumAuc += cumPosCount * curScoreNegCount;
                            cumAuc += 0.5 * curScorePosCount * curScoreNegCount;
                            cumPosCount += curScorePosCount;
                            cumNegCount += curScoreNegCount;
                        }
                    }
                    while (hasMorePos)
                    {
                        var weight = posSorted.Current.Weight;
                        cumPosWeight += weight;
                        cumPosCount++;
                        hasMorePos = posSorted.MoveNext();
                    }
                    while (hasMoreNeg)
                    {
                        var weight = negSorted.Current.Weight;
                        cumWeightedAuc += cumPosWeight * weight;
                        cumAuc += cumPosCount;
                        if (posScore == negSorted.Current.Score)
                        {
                            cumWeightedAuc -= 0.5 * curScorePosWeight * weight;
                            cumAuc -= 0.5 * curScorePosCount;
                        }
                        cumNegWeight += weight;
                        cumNegCount++;
                        hasMoreNeg = negSorted.MoveNext();
                    }
                    unweighted = cumAuc / ((Double)cumPosCount * cumNegCount);
                    return cumWeightedAuc / (cumPosWeight * cumNegWeight);
                }
            }

            protected override ValueGetter<AucInfo> GetSampleGetter()
            {
                return (ref AucInfo dst) => dst = new AucInfo() { Score = Score, Weight = _weight };
            }

            protected override void ProcessRowCore(Single weight)
            {
                _weight = weight;
                base.ProcessRowCore(weight);
            }

            protected override void AddExample(List<AucInfo> examples)
            {
                Contracts.AssertValue(examples);
                examples.Add(new AucInfo() { Score = Score, Weight = _weight });
            }
        }

        internal abstract class AuPrcAggregatorBase
        {
            protected Single Score;
            protected Single Label;
            protected Single Weight;

            public void ProcessRow(Single label, Single score, Single weight = 1)
            {
                Label = label;
                Score = score;
                Weight = weight;
                ProcessRowCore();
            }

            protected abstract void ProcessRowCore();

            public abstract Double ComputeWeightedAuPrc(out Double unweighted);
        }

        private protected abstract class AuPrcAggregatorBase<T> : AuPrcAggregatorBase
        {
            protected readonly ReservoirSamplerWithoutReplacement<T> Reservoir;

            protected AuPrcAggregatorBase(IRandom rand, int reservoirSize)
            {
                Contracts.Assert(reservoirSize > 0);

                ValueGetter<T> sampleGetter = GetSampleGetter();
                Reservoir = new ReservoirSamplerWithoutReplacement<T>(rand, reservoirSize, sampleGetter);
            }

            protected abstract ValueGetter<T> GetSampleGetter();

            protected override void ProcessRowCore()
            {
                Reservoir.Sample();
            }

            public override Double ComputeWeightedAuPrc(out Double unweighted)
            {
                if (Reservoir.Size == 0)
                    return unweighted = 0;
                return ComputeWeightedAuPrcCore(out unweighted);
            }

            protected abstract Double ComputeWeightedAuPrcCore(out Double unweighted);
        }

        private protected sealed class UnweightedAuPrcAggregator : AuPrcAggregatorBase<UnweightedAuPrcAggregator.Info>
        {
            public struct Info
            {
                public Single Score;
                public Single Label;
            }

            public UnweightedAuPrcAggregator(IRandom rand, int reservoirSize)
                : base(rand, reservoirSize)
            {
            }

            /// <summary>
            /// Compute the AUPRC using the "lower trapesoid" estimator, as described in the paper
            /// <a href="https://www.ecmlpkdd2013.org/wp-content/uploads/2013/07/aucpr_2013ecml_corrected.pdf">https://www.ecmlpkdd2013.org/wp-content/uploads/2013/07/aucpr_2013ecml_corrected.pdf</a>.
            /// </summary>
            protected override Double ComputeWeightedAuPrcCore(out Double unweighted)
            {
                Reservoir.Lock();
                var sample = Reservoir.GetSample().ToArray();
                int posCount = 0;
                int negCount = 0;
                foreach (var info in sample)
                {
                    if (info.Label > 0)
                        posCount++;
                    else
                        negCount++;
                }

                // Start with everything predicted 0, in each step change the prediction of the largest
                // current example from 0 to 1.
                var sortedIndices = Enumerable.Range(0, posCount + negCount).OrderByDescending(i => sample[i].Score);

                var prevRecall = 0.0;
                var prevPrecisionMin = 1.0;
                int truePos = 0;
                int falsePos = 0;
                var cumAuPrc = 0.0;
                foreach (var i in sortedIndices)
                {
                    if (sample[i].Label > 0)
                    {
                        // If the current example is positive, both recall and precision increase.
                        truePos++;
                        var curRecall = (Double)truePos / posCount;
                        var curPrecision = (Double)truePos / (truePos + falsePos);
                        cumAuPrc += (curRecall - prevRecall) * (prevPrecisionMin + curPrecision) / 2;
                        prevPrecisionMin = curPrecision;
                        prevRecall = curRecall;
                    }
                    else
                    {
                        // If the current example is negative, recall stays the same and precision decreases.
                        falsePos++;
                        prevPrecisionMin = (Double)truePos / (truePos + falsePos);
                    }
                }
                return unweighted = cumAuPrc;
            }

            protected override ValueGetter<Info> GetSampleGetter()
            {
                return
                    (ref Info dst) =>
                    {
                        dst.Score = Score;
                        dst.Label = Label;
                    };
            }
        }

        private protected sealed class WeightedAuPrcAggregator : AuPrcAggregatorBase<WeightedAuPrcAggregator.Info>
        {
            public struct Info
            {
                public Single Score;
                public Single Label;
                public Single Weight;
            }

            public WeightedAuPrcAggregator(IRandom rand, int reservoirSize)
                : base(rand, reservoirSize)
            {
            }

            /// <summary>
            /// Compute the AUPRC using the "lower trapesoid" estimator, as described in the paper
            /// <a href="https://www.ecmlpkdd2013.org/wp-content/uploads/2013/07/aucpr_2013ecml_corrected.pdf">https://www.ecmlpkdd2013.org/wp-content/uploads/2013/07/aucpr_2013ecml_corrected.pdf</a>.
            /// </summary>
            protected override Double ComputeWeightedAuPrcCore(out Double unweighted)
            {
                Reservoir.Lock();
                var sample = Reservoir.GetSample();
                int posCount = 0;
                int negCount = 0;
                Double posWeight = 0;
                Double negWeight = 0;
                foreach (var info in sample)
                {
                    if (info.Label > 0)
                    {
                        posCount++;
                        posWeight += info.Weight;
                    }
                    else
                    {
                        negCount++;
                        negWeight += info.Weight;
                    }
                }

                // Start with everything predicted 0, in each step change the prediction of the largest
                // current example from 0 to 1.
                var sorted = sample.Select((info, i) => new KeyValuePair<int, Info>(i, info))
                    .OrderByDescending(kvp => kvp.Value.Score);

                var prevWeightedRecall = 0.0;
                var prevWeightedPrecisionMin = 1.0;
                var truePosWeight = 0.0;
                var falsePosWeight = 0.0;
                var cumWeightedAuPrc = 0.0;
                var prevRecall = 0.0;
                var prevPrecision = 1.0;
                var truePosCount = 0.0;
                var falsePosCount = 0.0;
                unweighted = 0;
                foreach (var kvp in sorted)
                {
                    if (kvp.Value.Label > 0)
                    {
                        // If the current example is positive, both recall and precision increase.
                        truePosWeight += kvp.Value.Weight;
                        truePosCount++;
                        var curWeightedRecall = truePosWeight / posWeight;
                        var curWeightedPrecision = truePosWeight / (truePosWeight + falsePosWeight);
                        var curRecall = truePosCount / posCount;
                        var curPrecision = truePosCount / (truePosCount + falsePosCount);
                        cumWeightedAuPrc += (curWeightedRecall - prevWeightedRecall) * (prevWeightedPrecisionMin + curWeightedPrecision) / 2;
                        prevWeightedPrecisionMin = curWeightedPrecision;
                        prevWeightedRecall = curWeightedRecall;
                        unweighted += (curRecall - prevRecall) * (prevPrecision + curPrecision) / 2;
                        prevPrecision = curPrecision;
                        prevRecall = curRecall;
                    }
                    else
                    {
                        // If the current example is negative, recall stays the same and precision decreases.
                        falsePosWeight += kvp.Value.Weight;
                        falsePosCount++;
                        prevWeightedPrecisionMin = truePosWeight / (truePosWeight + falsePosWeight);
                        prevPrecision = truePosCount / (truePosCount + falsePosCount);
                    }
                }
                return cumWeightedAuPrc;
            }

            protected override ValueGetter<Info> GetSampleGetter()
            {
                return
                    (ref Info dst) =>
                    {
                        dst.Score = Score;
                        dst.Label = Label;
                        dst.Weight = Weight;
                    };
            }
        }
    }
}
