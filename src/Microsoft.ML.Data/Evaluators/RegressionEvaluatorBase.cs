// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    public abstract class RegressionLossEvaluatorBase<TAgg> : RowToRowEvaluatorBase<TAgg>
        where TAgg : EvaluatorBase<TAgg>.AggregatorBase
    {
        public abstract class ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "Loss function", ShortName = "loss")]
            public ISupportRegressionLossFactory LossFunction = new SquaredLossFactory();
        }

        public const string L1 = "L1(avg)";
        public const string L2 = "L2(avg)";
        public const string Rms = "RMS(avg)";
        public const string Loss = "Loss-fn(avg)";
        public const string RSquared = "R Squared";

        protected readonly IRegressionLoss LossFunction;

        protected RegressionLossEvaluatorBase(ArgumentsBase args, IHostEnvironment env, string registrationName)
            : base(env, registrationName)
        {
            Host.CheckUserArg(args.LossFunction != null, nameof(args.LossFunction), "Loss function must be specified.");
            LossFunction = args.LossFunction.CreateComponent(env);
        }
    }

    public abstract class RegressionEvaluatorBase<TAgg, TScore, TMetrics> : RegressionLossEvaluatorBase<TAgg>
        where TAgg : RegressionEvaluatorBase<TAgg, TScore, TMetrics>.RegressionAggregatorBase
    {
        protected RegressionEvaluatorBase(ArgumentsBase args, IHostEnvironment env, string registrationName)
            : base(args, env, registrationName)
        {
        }

        protected override void GetAggregatorConsolidationFuncs(TAgg aggregator, AggregatorDictionaryBase[] dictionaries,
            out Action<uint, ReadOnlyMemory<char>, TAgg> addAgg, out Func<Dictionary<string, IDataView>> consolidate)
        {
            var stratCol = new List<uint>();
            var stratVal = new List<ReadOnlyMemory<char>>();
            var isWeighted = new List<bool>();
            var l1 = new List<TMetrics>();
            var l2 = new List<TMetrics>();
            var rms = new List<TMetrics>();
            var loss = new List<TMetrics>();
            var rSquared = new List<TMetrics>();

            bool hasStrats = Utils.Size(dictionaries) > 0;
            bool hasWeight = aggregator.Weighted;

            addAgg =
                (stratColKey, stratColVal, agg) =>
                {
                    Host.Check(agg.Weighted == hasWeight, "All aggregators must either be weighted or unweighted");

                    stratCol.Add(stratColKey);
                    stratVal.Add(stratColVal);
                    isWeighted.Add(false);
                    l1.Add(agg.UnweightedCounters.L1);
                    l2.Add(agg.UnweightedCounters.L2);
                    rms.Add(agg.UnweightedCounters.Rms);
                    loss.Add(agg.UnweightedCounters.Loss);
                    rSquared.Add(agg.UnweightedCounters.RSquared);
                    if (agg.Weighted)
                    {
                        stratCol.Add(stratColKey);
                        stratVal.Add(stratColVal);
                        isWeighted.Add(true);
                        l1.Add(agg.WeightedCounters.L1);
                        l2.Add(agg.WeightedCounters.L2);
                        rms.Add(agg.WeightedCounters.Rms);
                        loss.Add(agg.WeightedCounters.Loss);
                        rSquared.Add(agg.WeightedCounters.RSquared);
                    }
                };

            consolidate =
                () =>
                {
                    var overallDvBldr = new ArrayDataViewBuilder(Host);
                    if (hasStrats)
                    {
                        overallDvBldr.AddColumn(MetricKinds.ColumnNames.StratCol, GetKeyValueGetter(dictionaries), 0, dictionaries.Length, stratCol.ToArray());
                        overallDvBldr.AddColumn(MetricKinds.ColumnNames.StratVal, TextType.Instance, stratVal.ToArray());
                    }
                    if (hasWeight)
                        overallDvBldr.AddColumn(MetricKinds.ColumnNames.IsWeighted, BoolType.Instance, isWeighted.ToArray());
                    aggregator.AddColumn(overallDvBldr, L1, l1.ToArray());
                    aggregator.AddColumn(overallDvBldr, L2, l2.ToArray());
                    aggregator.AddColumn(overallDvBldr, Rms, rms.ToArray());
                    aggregator.AddColumn(overallDvBldr, Loss, loss.ToArray());
                    aggregator.AddColumn(overallDvBldr, RSquared, rSquared.ToArray());

                    var result = new Dictionary<string, IDataView>();
                    result.Add(MetricKinds.OverallMetrics, overallDvBldr.GetDataView());
                    return result;
                };
        }

        public abstract class RegressionAggregatorBase : AggregatorBase
        {
            public abstract class CountersBase
            {
                protected Double SumWeights;
                protected TMetrics TotalL1Loss;
                protected TMetrics TotalL2Loss;
                protected TMetrics TotalLoss;
                protected Double TotalLabelW;
                protected Double TotalLabelSquaredW;

                public TMetrics L1
                {
                    get
                    {
                        var res = Zero();
                        if (SumWeights > 0)
                            Normalize(ref TotalL1Loss, ref res);
                        return res;
                    }
                }

                public TMetrics L2
                {
                    get
                    {
                        var res = Zero();
                        if (SumWeights > 0)
                            Normalize(ref TotalL2Loss, ref res);
                        return res;
                    }
                }

                public abstract TMetrics Rms { get; }

                //Note this can be NaN if regressor reports loss as NaN
                public TMetrics Loss
                {
                    get
                    {
                        var res = Zero();
                        if (SumWeights > 0)
                            Normalize(ref TotalLoss, ref res);
                        return res;
                    }
                }

                public abstract TMetrics RSquared { get; }

                public void Update(ref TScore score, float label, float weight, ref TMetrics loss)
                {
                    SumWeights += weight;
                    TotalLabelW += label * weight;
                    TotalLabelSquaredW += label * label * weight;
                    UpdateCore(label, ref score, ref loss, weight);
                }

                protected abstract void UpdateCore(float label, ref TScore score, ref TMetrics loss, float weight);

                protected abstract void Normalize(ref TMetrics src, ref TMetrics dst);

                protected abstract TMetrics Zero();
            }

            private ValueGetter<float> _labelGetter;
            private ValueGetter<TScore> _scoreGetter;
            private ValueGetter<float> _weightGetter;
            protected TScore Score;
            protected TMetrics Loss;

            protected readonly IRegressionLoss LossFunction;

            public readonly bool Weighted;

            public abstract CountersBase UnweightedCounters { get; }
            public abstract CountersBase WeightedCounters { get; }

            protected RegressionAggregatorBase(IHostEnvironment env, IRegressionLoss lossFunction, bool weighted, string stratName)
                : base(env, stratName)
            {
                Host.AssertValue(lossFunction);
                LossFunction = lossFunction;
                Weighted = weighted;
            }

            public override void InitializeNextPass(IRow row, RoleMappedSchema schema)
            {
                Contracts.Assert(PassNum < 1);
                Contracts.AssertValue(schema.Label);

                var score = schema.GetUniqueColumn(MetadataUtils.Const.ScoreValueKind.Score);

                _labelGetter = RowCursorUtils.GetLabelGetter(row, schema.Label.Index);
                _scoreGetter = row.GetGetter<TScore>(score.Index);
                Contracts.AssertValue(_labelGetter);
                Contracts.AssertValue(_scoreGetter);

                if (schema.Weight != null)
                    _weightGetter = row.GetGetter<float>(schema.Weight.Index);
            }

            public override void ProcessRow()
            {
                float label = 0;
                _labelGetter(ref label);
                _scoreGetter(ref Score);

                if (float.IsNaN(label))
                {
                    NumUnlabeledInstances++;
                    return;
                }

                if (IsNaN(ref Score))
                {
                    NumBadScores++;
                    return;
                }

                float weight = 1;
                if (_weightGetter != null)
                {
                    _weightGetter(ref weight);
                    if (!FloatUtils.IsFinite(weight))
                    {
                        NumBadWeights++;
                        weight = 1;
                    }
                }

                ApplyLossFunction(ref Score, label, ref Loss);
                UnweightedCounters.Update(ref Score, label, 1, ref Loss);
                if (WeightedCounters != null)
                    WeightedCounters.Update(ref Score, label, weight, ref Loss);
            }

            protected abstract void ApplyLossFunction(ref TScore score, float label, ref TMetrics loss);

            protected abstract bool IsNaN(ref TScore score);

            public abstract void AddColumn(ArrayDataViewBuilder dvBldr, string metricName, params TMetrics[] metric);
        }
    }
}