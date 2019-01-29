// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(typeof(BinaryClassifierEvaluator), typeof(BinaryClassifierEvaluator), typeof(BinaryClassifierEvaluator.Arguments), typeof(SignatureEvaluator),
    "Binary Classifier Evaluator", BinaryClassifierEvaluator.LoadName, "BinaryClassifier", "Binary", "bin")]

[assembly: LoadableClass(typeof(BinaryClassifierMamlEvaluator), typeof(BinaryClassifierMamlEvaluator), typeof(BinaryClassifierMamlEvaluator.Arguments), typeof(SignatureMamlEvaluator),
    "Binary Classifier Evaluator", BinaryClassifierEvaluator.LoadName, "BinaryClassifier", "Binary", "bin")]

// This is for deserialization from a binary model file.
[assembly: LoadableClass(typeof(BinaryPerInstanceEvaluator), null, typeof(SignatureLoadRowMapper),
    "", BinaryPerInstanceEvaluator.LoaderSignature)]

[assembly: LoadableClass(typeof(void), typeof(Evaluate), null, typeof(SignatureEntryPointModule), "Evaluators")]

namespace Microsoft.ML.Data
{
    public sealed class BinaryClassifierEvaluator : RowToRowEvaluatorBase<BinaryClassifierEvaluator.Aggregator>
    {
        public sealed class Arguments
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Probability value for classification thresholding")]
            public Single Threshold;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Use raw score value instead of probability for classification thresholding", ShortName = "useRawScore")]
            public bool UseRawScoreThreshold = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of samples to use for p/r curve generation. Specify 0 for no p/r curve generation", ShortName = "numpr")]
            public int NumRocExamples;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of samples to use for AUC calculation. If 0, AUC is not computed. If -1, the whole dataset is used", ShortName = "numauc")]
            public int MaxAucExamples = -1;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of samples to use for AUPRC calculation. Specify 0 for no AUPRC calculation", ShortName = "numauprc")]
            public int NumAuPrcExamples = 100000;
        }

        public const string LoadName = "BinaryClassifierEvaluator";

        // Overall metrics.
        public const string Accuracy = "Accuracy";
        public const string PosPrecName = "Positive precision";
        public const string PosRecallName = "Positive recall";
        public const string NegPrecName = "Negative precision";
        public const string NegRecallName = "Negative recall";
        public const string Auc = "AUC";
        public const string LogLoss = "Log-loss";
        public const string LogLossReduction = "Log-loss reduction";
        public const string Entropy = "Test-set entropy (prior Log-Loss/instance)";
        public const string F1 = "F1 Score";
        public const string AuPrc = "AUPRC";

        public enum Metrics
        {
            [EnumValueDisplay(BinaryClassifierEvaluator.Accuracy)]
            Accuracy,
            [EnumValueDisplay(BinaryClassifierEvaluator.PosPrecName)]
            PosPrecName,
            [EnumValueDisplay(BinaryClassifierEvaluator.PosRecallName)]
            PosRecallName,
            [EnumValueDisplay(BinaryClassifierEvaluator.NegPrecName)]
            NegPrecName,
            [EnumValueDisplay(BinaryClassifierEvaluator.NegRecallName)]
            NegRecallName,
            [EnumValueDisplay(BinaryClassifierEvaluator.Auc)]
            Auc,
            [EnumValueDisplay(BinaryClassifierEvaluator.LogLoss)]
            LogLoss,
            [EnumValueDisplay(BinaryClassifierEvaluator.LogLossReduction)]
            LogLossReduction,
            [EnumValueDisplay(BinaryClassifierEvaluator.F1)]
            F1,
            [EnumValueDisplay(BinaryClassifierEvaluator.AuPrc)]
            AuPrc,
        }

        /// <summary>
        /// Binary classification evaluator outputs a data view with this name, which contains the p/r data.
        /// It contains the columns listed below, and in case data also contains a weight column, it contains
        /// also columns for the weighted values.
        /// and false positive rate.
        /// </summary>
        public const string PrCurve = "PrCurve";

        // Column names for the p/r data view.
        public const string Precision = "Precision";
        public const string Recall = "Recall";
        public const string FalsePositiveRate = "FPR";
        public const string Threshold = "Threshold";

        private readonly Single _threshold;
        private readonly bool _useRaw;
        private readonly int _prCount;
        private readonly int _aucCount;
        private readonly int _auPrcCount;

        public BinaryClassifierEvaluator(IHostEnvironment env, Arguments args)
            : base(env, LoadName)
        {
            var host = Host.NotSensitive();
            host.CheckValue(args, nameof(args));
            host.CheckUserArg(args.MaxAucExamples >= -1, nameof(args.MaxAucExamples), "Must be at least -1");
            host.CheckUserArg(args.NumRocExamples >= 0, nameof(args.NumRocExamples), "Must be non-negative");
            host.CheckUserArg(args.NumAuPrcExamples >= 0, nameof(args.NumAuPrcExamples), "Must be non-negative");

            _useRaw = args.UseRawScoreThreshold;
            _threshold = args.Threshold;
            _prCount = args.NumRocExamples;
            _aucCount = args.MaxAucExamples;
            _auPrcCount = args.NumAuPrcExamples;
        }

        private protected override void CheckScoreAndLabelTypes(RoleMappedSchema schema)
        {
            var score = schema.GetUniqueColumn(MetadataUtils.Const.ScoreValueKind.Score);
            var host = Host.SchemaSensitive();
            var t = score.Type;
            if (t != NumberType.Float)
                throw host.ExceptSchemaMismatch(nameof(schema), "score", score.Name, "R4", t.ToString());
            host.Check(schema.Label.HasValue, "Could not find the label column");
            t = schema.Label.Value.Type;
            if (t != NumberType.R4 && t != NumberType.R8 && t != BoolType.Instance && t.GetKeyCount() != 2)
                throw host.ExceptSchemaMismatch(nameof(schema), "label", schema.Label.Value.Name, "R4, R8, BL or a 2-value key", t.ToString());
        }

        private protected override void CheckCustomColumnTypesCore(RoleMappedSchema schema)
        {
            var prob = schema.GetColumns(MetadataUtils.Const.ScoreValueKind.Probability);
            var host = Host.SchemaSensitive();
            if (prob != null)
            {
                host.CheckParam(prob.Count == 1, nameof(schema), "Cannot have multiple probability columns");
                var probType = prob[0].Type;
                if (probType != NumberType.Float)
                    throw host.ExceptSchemaMismatch(nameof(schema), "probability", prob[0].Name, "R4", probType.ToString());
            }
            else if (!_useRaw)
            {
                throw host.ExceptParam(nameof(schema),
                    "Cannot compute the predicted label from the probability column because it does not exist");
            }
        }

        // Add also the probability column.
        private protected override Func<int, bool> GetActiveColsCore(RoleMappedSchema schema)
        {
            var pred = base.GetActiveColsCore(schema);
            var prob = schema.GetColumns(MetadataUtils.Const.ScoreValueKind.Probability);
            Host.Assert(prob == null || prob.Count == 1);
            return i => Utils.Size(prob) > 0 && i == prob[0].Index || pred(i);
        }

        private protected override Aggregator GetAggregatorCore(RoleMappedSchema schema, string stratName)
        {
            var classNames = GetClassNames(schema);
            return new Aggregator(Host, classNames, schema.Weight != null, _aucCount, _auPrcCount, _threshold, _useRaw, _prCount, stratName);
        }

        private ReadOnlyMemory<char>[] GetClassNames(RoleMappedSchema schema)
        {
            // Get the label names if they exist, or use the default names.
            var labelNames = default(VBuffer<ReadOnlyMemory<char>>);
            var labelCol = schema.Label.Value;
            if (labelCol.Type is KeyType &&
                labelCol.Metadata.Schema.GetColumnOrNull(MetadataUtils.Kinds.KeyValues)?.Type is VectorType vecType &&
                vecType.Size > 0 && vecType.ItemType == TextType.Instance)
            {
                labelCol.GetKeyValues(ref labelNames);
            }
            else
                labelNames = new VBuffer<ReadOnlyMemory<char>>(2, new[] { "positive".AsMemory(), "negative".AsMemory() });

            ReadOnlyMemory<char>[] names = new ReadOnlyMemory<char>[2];
            labelNames.CopyTo(names);
            return names;
        }

        private protected override IRowMapper CreatePerInstanceRowMapper(RoleMappedSchema schema)
        {
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.CheckParam(schema.Label != null, nameof(schema), "Could not find the label column");
            var scoreInfo = schema.GetUniqueColumn(MetadataUtils.Const.ScoreValueKind.Score);

            var probInfos = schema.GetColumns(MetadataUtils.Const.ScoreValueKind.Probability);
            var probCol = Utils.Size(probInfos) > 0 ? probInfos[0].Name : null;
            return new BinaryPerInstanceEvaluator(Host, schema.Schema, scoreInfo.Name, probCol, schema.Label.Value.Name, _threshold, _useRaw);
        }

        public override IEnumerable<MetricColumn> GetOverallMetricColumns()
        {
            yield return new MetricColumn("Accuracy", Accuracy);
            yield return new MetricColumn("PosPrec", PosPrecName);
            yield return new MetricColumn("PosRecall", PosRecallName);
            yield return new MetricColumn("NegPrec", NegPrecName);
            yield return new MetricColumn("NegRecall", NegRecallName);
            yield return new MetricColumn("AUC", Auc);
            yield return new MetricColumn("LogLoss", LogLoss, MetricColumn.Objective.Minimize);
            yield return new MetricColumn("LogLossReduction", LogLossReduction);
            yield return new MetricColumn("Entropy", Entropy);
            yield return new MetricColumn("F1", F1);
            yield return new MetricColumn("AUPRC", AuPrc);
        }

        private protected override void GetAggregatorConsolidationFuncs(Aggregator aggregator, AggregatorDictionaryBase[] dictionaries,
            out Action<uint, ReadOnlyMemory<char>, Aggregator> addAgg, out Func<Dictionary<string, IDataView>> consolidate)
        {
            var stratCol = new List<uint>();
            var stratVal = new List<ReadOnlyMemory<char>>();
            var isWeighted = new List<bool>();
            var auc = new List<Double>();
            var accuracy = new List<Double>();
            var posPrec = new List<Double>();
            var posRecall = new List<Double>();
            var negPrec = new List<Double>();
            var negRecall = new List<Double>();
            var logLoss = new List<Double>();
            var logLossRed = new List<Double>();
            var entropy = new List<Double>();
            var f1 = new List<Double>();
            var auprc = new List<Double>();

            var counts = new List<Double[]>();
            var weights = new List<Double[]>();
            var confStratCol = new List<uint>();
            var confStratVal = new List<ReadOnlyMemory<char>>();

            var scores = new List<Single>();
            var precision = new List<Double>();
            var recall = new List<Double>();
            var fpr = new List<Double>();
            var weightedPrecision = new List<Double>();
            var weightedRecall = new List<Double>();
            var weightedFpr = new List<Double>();
            var prStratCol = new List<uint>();
            var prStratVal = new List<ReadOnlyMemory<char>>();

            bool hasStrats = Utils.Size(dictionaries) > 0;
            bool hasWeight = aggregator.Weighted;

            addAgg =
                (stratColKey, stratColVal, agg) =>
                {
                    Host.Check(agg.Weighted == hasWeight, "All aggregators must either be weighted or unweighted");
                    Host.Check((agg.AuPrcAggregator == null) == (aggregator.AuPrcAggregator == null),
                        "All aggregators must either compute AUPRC or not compute AUPRC");

                    agg.Finish();
                    stratCol.Add(stratColKey);
                    stratVal.Add(stratColVal);
                    isWeighted.Add(false);
                    auc.Add(agg.UnweightedAuc);
                    accuracy.Add(agg.UnweightedCounters.Acc);
                    posPrec.Add(agg.UnweightedCounters.PrecisionPos);
                    posRecall.Add(agg.UnweightedCounters.RecallPos);
                    negPrec.Add(agg.UnweightedCounters.PrecisionNeg);
                    negRecall.Add(agg.UnweightedCounters.RecallNeg);
                    logLoss.Add(agg.UnweightedCounters.LogLoss);
                    logLossRed.Add(agg.UnweightedCounters.LogLossReduction);
                    entropy.Add(agg.UnweightedCounters.Entropy);
                    f1.Add(agg.UnweightedCounters.F1);
                    if (agg.AuPrcAggregator != null)
                        auprc.Add(agg.UnweightedAuPrc);

                    confStratCol.AddRange(new[] { stratColKey, stratColKey });
                    confStratVal.AddRange(new[] { stratColVal, stratColVal });
                    counts.Add(new[] { agg.UnweightedCounters.NumTruePos, agg.UnweightedCounters.NumFalseNeg });
                    counts.Add(new[] { agg.UnweightedCounters.NumFalsePos, agg.UnweightedCounters.NumTrueNeg });
                    if (agg.Scores != null)
                    {
                        Host.AssertValue(agg.Precision);
                        Host.AssertValue(agg.Recall);
                        Host.AssertValue(agg.FalsePositiveRate);

                        scores.AddRange(agg.Scores);
                        precision.AddRange(agg.Precision);
                        recall.AddRange(agg.Recall);
                        fpr.AddRange(agg.FalsePositiveRate);

                        if (hasStrats)
                        {
                            prStratCol.AddRange(agg.Scores.Select(x => stratColKey));
                            prStratVal.AddRange(agg.Scores.Select(x => stratColVal));
                        }
                    }
                    if (agg.Weighted)
                    {
                        stratCol.Add(stratColKey);
                        stratVal.Add(stratColVal);
                        isWeighted.Add(true);
                        auc.Add(agg.WeightedAuc);
                        accuracy.Add(agg.WeightedCounters.Acc);
                        posPrec.Add(agg.WeightedCounters.PrecisionPos);
                        posRecall.Add(agg.WeightedCounters.RecallPos);
                        negPrec.Add(agg.WeightedCounters.PrecisionNeg);
                        negRecall.Add(agg.WeightedCounters.RecallNeg);
                        logLoss.Add(agg.WeightedCounters.LogLoss);
                        logLossRed.Add(agg.WeightedCounters.LogLossReduction);
                        entropy.Add(agg.WeightedCounters.Entropy);
                        f1.Add(agg.WeightedCounters.F1);
                        if (agg.AuPrcAggregator != null)
                            auprc.Add(agg.WeightedAuPrc);
                        weights.Add(new[] { agg.WeightedCounters.NumTruePos, agg.WeightedCounters.NumFalseNeg });
                        weights.Add(new[] { agg.WeightedCounters.NumFalsePos, agg.WeightedCounters.NumTrueNeg });

                        if (agg.Scores != null)
                        {
                            Host.AssertValue(agg.WeightedPrecision);
                            Host.AssertValue(agg.WeightedRecall);
                            Host.AssertValue(agg.WeightedFalsePositiveRate);

                            weightedPrecision.AddRange(agg.WeightedPrecision);
                            weightedRecall.AddRange(agg.WeightedRecall);
                            weightedFpr.AddRange(agg.WeightedFalsePositiveRate);
                        }
                    }
                };

            consolidate =
                () =>
                {
                    var overallDvBldr = new ArrayDataViewBuilder(Host);
                    if (hasStrats)
                    {
                        overallDvBldr.AddColumn(MetricKinds.ColumnNames.StratCol, GetKeyValueGetter(dictionaries), (ulong)dictionaries.Length, stratCol.ToArray());
                        overallDvBldr.AddColumn(MetricKinds.ColumnNames.StratVal, TextType.Instance, stratVal.ToArray());
                    }
                    if (hasWeight)
                        overallDvBldr.AddColumn(MetricKinds.ColumnNames.IsWeighted, BoolType.Instance, isWeighted.ToArray());
                    overallDvBldr.AddColumn(Auc, NumberType.R8, auc.ToArray());
                    overallDvBldr.AddColumn(Accuracy, NumberType.R8, accuracy.ToArray());
                    overallDvBldr.AddColumn(PosPrecName, NumberType.R8, posPrec.ToArray());
                    overallDvBldr.AddColumn(PosRecallName, NumberType.R8, posRecall.ToArray());
                    overallDvBldr.AddColumn(NegPrecName, NumberType.R8, negPrec.ToArray());
                    overallDvBldr.AddColumn(NegRecallName, NumberType.R8, negRecall.ToArray());
                    overallDvBldr.AddColumn(LogLoss, NumberType.R8, logLoss.ToArray());
                    overallDvBldr.AddColumn(LogLossReduction, NumberType.R8, logLossRed.ToArray());
                    overallDvBldr.AddColumn(Entropy, NumberType.R8, entropy.ToArray());
                    overallDvBldr.AddColumn(F1, NumberType.R8, f1.ToArray());
                    if (aggregator.AuPrcAggregator != null)
                        overallDvBldr.AddColumn(AuPrc, NumberType.R8, auprc.ToArray());

                    var confDvBldr = new ArrayDataViewBuilder(Host);
                    if (hasStrats)
                    {
                        confDvBldr.AddColumn(MetricKinds.ColumnNames.StratCol, GetKeyValueGetter(dictionaries), (ulong)dictionaries.Length, confStratCol.ToArray());
                        confDvBldr.AddColumn(MetricKinds.ColumnNames.StratVal, TextType.Instance, confStratVal.ToArray());
                    }
                    ValueGetter<VBuffer<ReadOnlyMemory<char>>> getSlotNames =
                        (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                            dst = new VBuffer<ReadOnlyMemory<char>>(aggregator.ClassNames.Length, aggregator.ClassNames);
                    confDvBldr.AddColumn(MetricKinds.ColumnNames.Count, getSlotNames, NumberType.R8, counts.ToArray());

                    if (hasWeight)
                        confDvBldr.AddColumn(MetricKinds.ColumnNames.Weight, getSlotNames, NumberType.R8, weights.ToArray());

                    var result = new Dictionary<string, IDataView>();
                    result.Add(MetricKinds.OverallMetrics, overallDvBldr.GetDataView());
                    result.Add(MetricKinds.ConfusionMatrix, confDvBldr.GetDataView());

                    if (scores.Count > 0)
                    {
                        var dvBldr = new ArrayDataViewBuilder(Host);
                        if (hasStrats)
                        {
                            dvBldr.AddColumn(MetricKinds.ColumnNames.StratCol, GetKeyValueGetter(dictionaries), (ulong)dictionaries.Length, prStratCol.ToArray());
                            dvBldr.AddColumn(MetricKinds.ColumnNames.StratVal, TextType.Instance, prStratVal.ToArray());
                        }
                        dvBldr.AddColumn(Threshold, NumberType.R4, scores.ToArray());
                        dvBldr.AddColumn(Precision, NumberType.R8, precision.ToArray());
                        dvBldr.AddColumn(Recall, NumberType.R8, recall.ToArray());
                        dvBldr.AddColumn(FalsePositiveRate, NumberType.R8, fpr.ToArray());
                        if (weightedPrecision.Count > 0)
                        {
                            dvBldr.AddColumn("Weighted " + Precision, NumberType.R8, weightedPrecision.ToArray());
                            dvBldr.AddColumn("Weighted " + Recall, NumberType.R8, weightedRecall.ToArray());
                            dvBldr.AddColumn("Weighted " + FalsePositiveRate, NumberType.R8, weightedFpr.ToArray());
                        }
                        result.Add(PrCurve, dvBldr.GetDataView());
                    }
                    return result;
                };
        }

        public sealed class Aggregator : AggregatorBase
        {
            public sealed class Counters
            {
                private readonly bool _useRaw;
                private readonly Single _threshold;

                public Double NumTruePos;
                public Double NumTrueNeg;
                public Double NumFalsePos;
                public Double NumFalseNeg;
                private Double _numLogLossPositives;
                private Double _numLogLossNegatives;
                private Double _logLoss;

                public Double Acc
                {
                    get
                    {
                        return (NumTrueNeg + NumTruePos) / (NumTruePos + NumTrueNeg + NumFalseNeg + NumFalsePos);
                    }
                }

                public Double RecallPos
                {
                    get
                    {
                        return (NumTruePos + NumFalseNeg > 0) ? NumTruePos / (NumTruePos + NumFalseNeg) : 0;
                    }
                }

                public Double PrecisionPos
                {
                    get
                    {
                        return (NumTruePos + NumFalsePos > 0) ? NumTruePos / (NumTruePos + NumFalsePos) : 0;
                    }
                }

                public Double RecallNeg
                {
                    get
                    {
                        return (NumTrueNeg + NumFalsePos > 0) ? NumTrueNeg / (NumTrueNeg + NumFalsePos) : 0;
                    }
                }

                public Double PrecisionNeg
                {
                    get
                    {
                        return (NumTrueNeg + NumFalseNeg > 0) ? NumTrueNeg / (NumTrueNeg + NumFalseNeg) : 0;
                    }
                }

                public Double Entropy
                {
                    get
                    {
                        return MathUtils.Entropy((NumTruePos + NumFalseNeg) /
                            (NumTruePos + NumTrueNeg + NumFalseNeg + NumFalsePos));
                    }
                }

                public Double LogLoss
                {
                    get
                    {
                        return Double.IsNaN(_logLoss) ? Double.NaN : (_numLogLossPositives + _numLogLossNegatives > 0)
                            ? _logLoss / (_numLogLossPositives + _numLogLossNegatives) : 0;
                    }
                }

                public Double LogLossReduction
                {
                    get
                    {
                        if (_numLogLossPositives + _numLogLossNegatives == 0)
                            return 0;
                        var logLoss = _logLoss / (_numLogLossPositives + _numLogLossNegatives);
                        var priorPos = _numLogLossPositives / (_numLogLossPositives + _numLogLossNegatives);
                        var priorLogLoss = MathUtils.Entropy(priorPos);
                        return 100 * (priorLogLoss - logLoss) / priorLogLoss;
                    }
                }

                public Double F1 { get { return 2 * PrecisionPos * RecallPos / (PrecisionPos + RecallPos); } }

                public Counters(bool useRaw, Single threshold)
                {
                    _useRaw = useRaw;
                    _threshold = threshold;
                }

                public void Update(Single score, Single prob, Single label, Double logloss, Single weight)
                {
                    bool predictPositive = _useRaw ? score > _threshold : prob > _threshold;

                    if (label > 0)
                    {
                        if (predictPositive)
                            NumTruePos += weight;
                        else
                            NumFalseNeg += weight;
                    }
                    else
                    {
                        if (predictPositive)
                            NumFalsePos += weight;
                        else
                            NumTrueNeg += weight;
                    }

                    if (!Single.IsNaN(prob))
                    {
                        if (label > 0)
                            _numLogLossPositives += weight;
                        else
                            _numLogLossNegatives += weight;
                    }

                    _logLoss += logloss * weight;
                }
            }

            private struct RocInfo
            {
                public Single Score;
                public Single Label;
                public Single Weight;
            }

            private readonly ReservoirSamplerWithoutReplacement<RocInfo> _prCurveReservoir;
            public readonly List<Single> Scores;
            public readonly List<Double> Precision;
            public readonly List<Double> Recall;
            public readonly List<Double> FalsePositiveRate;
            public readonly List<Double> WeightedPrecision;
            public readonly List<Double> WeightedRecall;
            public readonly List<Double> WeightedFalsePositiveRate;

            internal readonly AuPrcAggregatorBase AuPrcAggregator;
            public double WeightedAuPrc;
            public double UnweightedAuPrc;

            private readonly AucAggregatorBase _aucAggregator;
            public double WeightedAuc;
            public double UnweightedAuc;

            public readonly Counters UnweightedCounters;
            public readonly Counters WeightedCounters;

            public readonly bool Weighted;

            private ValueGetter<Single> _labelGetter;
            private ValueGetter<Single> _scoreGetter;
            private ValueGetter<Single> _weightGetter;
            private ValueGetter<Single> _probGetter;
            private Single _score;
            private Single _label;
            private Single _weight;

            public readonly ReadOnlyMemory<char>[] ClassNames;

            public Aggregator(IHostEnvironment env, ReadOnlyMemory<char>[] classNames, bool weighted, int aucReservoirSize,
                int auPrcReservoirSize, Single threshold, bool useRaw, int prCount, string stratName)
                : base(env, stratName)
            {
                Host.Assert(Utils.Size(classNames) == 2);
                Host.Assert(aucReservoirSize >= -1);
                Host.Assert(prCount >= 0);
                Host.Assert(auPrcReservoirSize >= 0);
                Host.Assert(useRaw || 0 <= threshold && threshold <= 1);

                ClassNames = classNames;
                UnweightedCounters = new Counters(useRaw, threshold);
                WeightedCounters = weighted ? new Counters(useRaw, threshold) : null;
                Weighted = weighted;
                if (weighted)
                {
                    _aucAggregator = new WeightedAucAggregator(Host.Rand, aucReservoirSize);
                    if (auPrcReservoirSize > 0)
                        AuPrcAggregator = new WeightedAuPrcAggregator(Host.Rand, auPrcReservoirSize);
                }
                else
                {
                    _aucAggregator = new UnweightedAucAggregator(Host.Rand, aucReservoirSize);
                    if (auPrcReservoirSize > 0)
                        AuPrcAggregator = new UnweightedAuPrcAggregator(Host.Rand, auPrcReservoirSize);
                }

                if (prCount > 0)
                {
                    ValueGetter<RocInfo> prSampleGetter =
                        (ref RocInfo dst) =>
                        {
                            dst.Label = _label;
                            dst.Score = _score;
                            dst.Weight = _weight;
                        };
                    _prCurveReservoir = new ReservoirSamplerWithoutReplacement<RocInfo>(Host.Rand, prCount, prSampleGetter);
                    Precision = new List<Double>();
                    Recall = new List<Double>();
                    FalsePositiveRate = new List<Double>();
                    Scores = new List<Single>();
                    if (weighted)
                    {
                        WeightedPrecision = new List<Double>();
                        WeightedRecall = new List<Double>();
                        WeightedFalsePositiveRate = new List<Double>();
                    }
                }
            }

            internal override void InitializeNextPass(Row row, RoleMappedSchema schema)
            {
                Host.Assert(schema.Label.HasValue);
                Host.Assert(PassNum < 1);

                var score = schema.GetUniqueColumn(MetadataUtils.Const.ScoreValueKind.Score);

                _labelGetter = RowCursorUtils.GetLabelGetter(row, schema.Label.Value.Index);
                _scoreGetter = row.GetGetter<Single>(score.Index);
                Host.AssertValue(_labelGetter);
                Host.AssertValue(_scoreGetter);

                var prob = schema.GetColumns(new RoleMappedSchema.ColumnRole(MetadataUtils.Const.ScoreValueKind.Probability));
                Host.Assert(prob == null || prob.Count == 1);

                if (prob != null)
                    _probGetter = row.GetGetter<Single>(prob[0].Index);
                else
                    _probGetter = (ref Single value) => value = Single.NaN;

                Host.Assert((schema.Weight != null) == Weighted);
                if (Weighted)
                    _weightGetter = row.GetGetter<Single>(schema.Weight.Value.Index);
            }

            public override void ProcessRow()
            {
                _labelGetter(ref _label);
                _scoreGetter(ref _score);
                if (!FloatUtils.IsFinite(_score))
                {
                    NumBadScores++;
                    return;
                }
                if (Single.IsNaN(_label))
                {
                    NumUnlabeledInstances++;
                    return;
                }

                Single prob = 0;
                _probGetter(ref prob);

                Double logloss;
                if (!Single.IsNaN(prob))
                {
                    if (_label > 0)
                    {
                        // REVIEW: Should we bring back the option to use ln instead of log2?
                        logloss = -Math.Log(prob, 2);
                    }
                    else
                        logloss = -Math.Log(1.0 - prob, 2);
                }
                else
                    logloss = Double.NaN;

                UnweightedCounters.Update(_score, prob, _label, logloss, 1);

                Host.Assert((_weightGetter != null) == Weighted);
                if (_weightGetter != null)
                {
                    _weightGetter(ref _weight);
                    if (!FloatUtils.IsFinite(_weight))
                    {
                        NumBadWeights++;
                        _weight = 1;
                    }
                    _aucAggregator.ProcessRow(_label, _score, _weight);
                    WeightedCounters.Update(_score, prob, _label, logloss, _weight);
                }
                else
                    _aucAggregator.ProcessRow(_label, _score);

                if (_prCurveReservoir != null)
                    _prCurveReservoir.Sample();
                if (AuPrcAggregator != null)
                    AuPrcAggregator.ProcessRow(_label, _score, _weight);
            }

            public void Finish()
            {
                Contracts.Assert(!IsActive());

                _aucAggregator.Finish();
                WeightedAuc = _aucAggregator.ComputeWeightedAuc(out UnweightedAuc);
                if (AuPrcAggregator != null)
                    WeightedAuPrc = AuPrcAggregator.ComputeWeightedAuPrc(out UnweightedAuPrc);
                FinishOtherMetrics();
            }

            private void FinishOtherMetrics()
            {
                if (_prCurveReservoir != null)
                    ComputePrCurves();
            }

            private void ComputePrCurves()
            {
                Host.AssertValue(_prCurveReservoir);
                Host.AssertValue(Scores);
                Host.AssertValue(Precision);
                Host.AssertValue(Recall);
                Host.AssertValue(FalsePositiveRate);

                _prCurveReservoir.Lock();
                var prSample = _prCurveReservoir.GetSample();
                Scores.Clear();
                Precision.Clear();
                Recall.Clear();
                FalsePositiveRate.Clear();
                if (Weighted)
                {
                    Host.AssertValue(WeightedPrecision);
                    Host.AssertValue(WeightedRecall);
                    Host.AssertValue(WeightedFalsePositiveRate);

                    WeightedPrecision.Clear();
                    WeightedRecall.Clear();
                    WeightedFalsePositiveRate.Clear();
                }

                Double pos = 0;
                Double neg = 0;
                Double wpos = 0;
                Double wneg = 0;
                Single scoreCur = Single.PositiveInfinity;
                foreach (var point in prSample.OrderByDescending(x => x.Score)
                    .Concat(new[] { new RocInfo() { Score = Single.NegativeInfinity } }))
                {
                    // Add the next point to the precision/recall/fpr lists.
                    if (point.Score < scoreCur)
                    {
                        if (pos + neg > 0)
                        {
                            Scores.Add(scoreCur);
                            Precision.Add(pos / (pos + neg));
                            Recall.Add(pos);
                            FalsePositiveRate.Add(neg);
                            if (Weighted)
                            {
                                WeightedPrecision.Add(wpos / (wpos + wneg));
                                WeightedRecall.Add(wpos);
                                WeightedFalsePositiveRate.Add(wneg);
                            }
                        }
                        scoreCur = point.Score;
                    }
                    if (Single.IsNegativeInfinity(point.Score))
                        continue;

                    if (point.Label > 0)
                        pos++;
                    else
                        neg++;
                    if (Weighted)
                    {
                        if (point.Label > 0)
                            wpos += point.Weight;
                        else
                            wneg += point.Weight;
                    }
                }

                // normalize recall and false positive rate
                for (int i = 0; i < Recall.Count; i++)
                {
                    Recall[i] /= pos;
                    FalsePositiveRate[i] /= neg;
                }
                if (Weighted)
                {
                    for (int i = 0; i < WeightedRecall.Count; i++)
                    {
                        WeightedRecall[i] /= wpos;
                        WeightedFalsePositiveRate[i] /= wneg;
                    }
                }
            }
        }

        /// <summary>
        /// Evaluates scored binary classification data.
        /// </summary>
        /// <param name="data">The scored data.</param>
        /// <param name="label">The name of the label column in <paramref name="data"/>.</param>
        /// <param name="score">The name of the score column in <paramref name="data"/>.</param>
        /// <param name="probability">The name of the probability column in <paramref name="data"/>, the calibrated version of <paramref name="score"/>.</param>
        /// <param name="predictedLabel">The name of the predicted label column in <paramref name="data"/>.</param>
        /// <returns>The evaluation results for these calibrated outputs.</returns>
        public CalibratedBinaryClassificationMetrics Evaluate(IDataView data, string label, string score, string probability, string predictedLabel)
        {
            Host.CheckValue(data, nameof(data));
            Host.CheckNonEmpty(label, nameof(label));
            Host.CheckNonEmpty(score, nameof(score));
            Host.CheckNonEmpty(probability, nameof(probability));
            Host.CheckNonEmpty(predictedLabel, nameof(predictedLabel));

            var roles = new RoleMappedData(data, opt: false,
                RoleMappedSchema.ColumnRole.Label.Bind(label),
                RoleMappedSchema.CreatePair(MetadataUtils.Const.ScoreValueKind.Score, score),
                RoleMappedSchema.CreatePair(MetadataUtils.Const.ScoreValueKind.Probability, probability),
                RoleMappedSchema.CreatePair(MetadataUtils.Const.ScoreValueKind.PredictedLabel, predictedLabel));

            var resultDict = ((IEvaluator)this).Evaluate(roles);
            Host.Assert(resultDict.ContainsKey(MetricKinds.OverallMetrics));
            var overall = resultDict[MetricKinds.OverallMetrics];

            CalibratedBinaryClassificationMetrics result;
            using (var cursor = overall.GetRowCursorForAllColumns())
            {
                var moved = cursor.MoveNext();
                Host.Assert(moved);
                result = new CalibratedBinaryClassificationMetrics(Host, cursor);
                moved = cursor.MoveNext();
                Host.Assert(!moved);
            }
            return result;
        }

        /// <summary>
        /// Evaluates scored binary classification data, without probability-based metrics.
        /// </summary>
        /// <param name="data">The scored data.</param>
        /// <param name="label">The name of the label column in <paramref name="data"/>.</param>
        /// <param name="score">The name of the score column in <paramref name="data"/>.</param>
        /// <param name="predictedLabel">The name of the predicted label column in <paramref name="data"/>.</param>
        /// <returns>The evaluation results for these uncalibrated outputs.</returns>
        /// <seealso cref="Evaluate(IDataView, string, string, string)"/>
        public BinaryClassificationMetrics Evaluate(IDataView data, string label, string score, string predictedLabel)
        {
            Host.CheckValue(data, nameof(data));
            Host.CheckNonEmpty(label, nameof(label));
            Host.CheckNonEmpty(score, nameof(score));
            Host.CheckNonEmpty(predictedLabel, nameof(predictedLabel));

            var roles = new RoleMappedData(data, opt: false,
                RoleMappedSchema.ColumnRole.Label.Bind(label),
                RoleMappedSchema.CreatePair(MetadataUtils.Const.ScoreValueKind.Score, score),
                RoleMappedSchema.CreatePair(MetadataUtils.Const.ScoreValueKind.PredictedLabel, predictedLabel));

            var resultDict = ((IEvaluator)this).Evaluate(roles);
            Host.Assert(resultDict.ContainsKey(MetricKinds.OverallMetrics));
            var overall = resultDict[MetricKinds.OverallMetrics];

            BinaryClassificationMetrics result;
            using (var cursor = overall.GetRowCursorForAllColumns())
            {
                var moved = cursor.MoveNext();
                Host.Assert(moved);
                result = new BinaryClassificationMetrics(Host, cursor);
                moved = cursor.MoveNext();
                Host.Assert(!moved);
            }
            return result;
        }
    }

    public sealed class BinaryPerInstanceEvaluator : PerInstanceEvaluatorBase
    {
        public const string LoaderSignature = "BinaryPerInstance";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "BIN INST",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(BinaryPerInstanceEvaluator).Assembly.FullName);
        }

        private const int AssignedCol = 0;
        private const int LogLossCol = 1;

        public const string LogLoss = "Log-loss";
        public const string Assigned = "Assigned";

        private readonly string _probCol;
        private readonly int _probIndex;
        private readonly Single _threshold;
        private readonly bool _useRaw;
        private readonly ColumnType[] _types;

        public BinaryPerInstanceEvaluator(IHostEnvironment env, Schema schema, string scoreCol, string probCol, string labelCol, Single threshold, bool useRaw)
            : base(env, schema, scoreCol, labelCol)
        {
            _threshold = threshold;
            _useRaw = useRaw;

            using (var ch = Host.Start("Finding Input Columns"))
            {
                _probCol = probCol;
                _probIndex = -1;
                if (string.IsNullOrEmpty(_probCol) || !schema.TryGetColumnIndex(_probCol, out _probIndex))
                    ch.Warning("Data does not contain a probability column. Will not output the Log-loss column");
                CheckInputColumnTypes(schema);
            }

            _types = new ColumnType[2];
            _types[LogLossCol] = NumberType.R8;
            _types[AssignedCol] = BoolType.Instance;
        }

        private BinaryPerInstanceEvaluator(IHostEnvironment env, ModelLoadContext ctx, Schema schema)
            : base(env, ctx, schema)
        {
            // *** Binary format **
            // base
            // int: Id of the probability column name
            // float: _threshold
            // byte: _useRaw

            _probCol = ctx.LoadStringOrNull();
            _probIndex = -1;
            if (_probCol != null && !schema.TryGetColumnIndex(_probCol, out _probIndex))
                throw Host.ExceptParam(nameof(schema), "Did not find the probability column '{0}'", _probCol);

            CheckInputColumnTypes(schema);

            _threshold = ctx.Reader.ReadFloat();
            _useRaw = ctx.Reader.ReadBoolByte();
            Host.CheckDecode(!string.IsNullOrEmpty(_probCol) || _useRaw);
            Host.CheckDecode(FloatUtils.IsFinite(_threshold));

            _types = new ColumnType[2];
            _types[LogLossCol] = NumberType.R8;
            _types[AssignedCol] = BoolType.Instance;
        }

        public static BinaryPerInstanceEvaluator Create(IHostEnvironment env, ModelLoadContext ctx, Schema schema)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new BinaryPerInstanceEvaluator(env, ctx, schema);
        }

        public override void Save(ModelSaveContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format **
            // base
            // int: Id of the probability column name
            // float: _threshold
            // byte: _useRaw

            base.Save(ctx);
            ctx.SaveStringOrNull(_probCol);
            Contracts.Assert(FloatUtils.IsFinite(_threshold));
            ctx.Writer.Write(_threshold);
            Contracts.Assert(!string.IsNullOrEmpty(_probCol) || _useRaw);
            ctx.Writer.WriteBoolByte(_useRaw);
        }

        private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
        {
            if (_probIndex >= 0)
            {
                return
                    col =>
                        activeOutput(LogLossCol) && (col == _probIndex || col == LabelIndex) ||
                        activeOutput(AssignedCol) && (_useRaw && col == ScoreIndex || !_useRaw && col == _probIndex);
            }
            Host.Assert(_useRaw);
            return col => activeOutput(AssignedCol) && col == ScoreIndex;
        }

        private protected override Delegate[] CreateGettersCore(Row input, Func<int, bool> activeCols, out Action disposer)
        {
            Host.Assert(LabelIndex >= 0);
            Host.Assert(ScoreIndex >= 0);
            Host.Assert(_probIndex >= 0 || _useRaw);

            disposer = null;

            long cachedPosition = -1;
            Single label = 0;
            Single prob = 0;
            Single score = 0;

            ValueGetter<Single> nanGetter = (ref Single value) => value = Single.NaN;
            var labelGetter = _probIndex >= 0 && activeCols(LogLossCol) ?
                RowCursorUtils.GetLabelGetter(input, LabelIndex) : nanGetter;
            ValueGetter<Single> probGetter;
            if (_probIndex >= 0 && activeCols(LogLossCol))
                probGetter = input.GetGetter<Single>(_probIndex);
            else
                probGetter = nanGetter;
            ValueGetter<Single> scoreGetter;
            if (activeCols(AssignedCol) && ScoreIndex >= 0)
                scoreGetter = input.GetGetter<Single>(ScoreIndex);
            else
                scoreGetter = nanGetter;

            Action updateCacheIfNeeded;
            Func<bool> getPredictedLabel;
            if (_useRaw)
            {
                updateCacheIfNeeded =
                    () =>
                    {
                        if (cachedPosition != input.Position)
                        {
                            labelGetter(ref label);
                            probGetter(ref prob);
                            scoreGetter(ref score);
                            cachedPosition = input.Position;
                        }
                    };
                getPredictedLabel = () => GetPredictedLabel(score);
            }
            else
            {
                updateCacheIfNeeded =
                    () =>
                    {
                        if (cachedPosition != input.Position)
                        {
                            labelGetter(ref label);
                            probGetter(ref prob);
                            cachedPosition = input.Position;
                        }
                    };
                getPredictedLabel = () => GetPredictedLabel(prob);
            }

            var getters = _probIndex >= 0 ? new Delegate[2] : new Delegate[1];
            if (activeCols(AssignedCol))
            {
                ValueGetter<bool> predFn =
                    (ref bool dst) =>
                    {
                        updateCacheIfNeeded();
                        dst = getPredictedLabel();
                    };
                getters[_probIndex >= 0 ? AssignedCol : 0] = predFn;
            }
            if (_probIndex >= 0 && activeCols(LogLossCol))
            {
                ValueGetter<Double> loglossFn =
                    (ref Double dst) =>
                    {
                        updateCacheIfNeeded();
                        dst = GetLogLoss(prob, label);
                    };
                getters[LogLossCol] = loglossFn;
            }
            return getters;
        }

        private Double GetLogLoss(Single prob, Single label)
        {
            if (Single.IsNaN(prob) || Single.IsNaN(label))
                return Double.NaN;
            if (label > 0)
                return -Math.Log(prob, 2);
            return -Math.Log(1.0 - prob, 2);
        }

        private bool GetPredictedLabel(Single val)
        {
            //Behavior for NA values is undefined.
            return Single.IsNaN(val) ? false : val > _threshold;
        }

        private protected override Schema.DetachedColumn[] GetOutputColumnsCore()
        {
            if (_probIndex >= 0)
            {
                var infos = new Schema.DetachedColumn[2];
                infos[LogLossCol] = new Schema.DetachedColumn(LogLoss, _types[LogLossCol], null);
                infos[AssignedCol] = new Schema.DetachedColumn(Assigned, _types[AssignedCol], null);
                return infos;
            }
            return new[] { new Schema.DetachedColumn(Assigned, _types[AssignedCol], null), };
        }

        private void CheckInputColumnTypes(Schema schema)
        {
            Host.AssertNonEmpty(ScoreCol);
            Host.AssertValueOrNull(_probCol);
            Host.AssertNonEmpty(LabelCol);

            var t = schema[(int)LabelIndex].Type;
            if (t != NumberType.R4 && t != NumberType.R8 && t != BoolType.Instance && t.GetKeyCount() != 2)
                throw Host.Except("Label column '{0}' has type '{1}' but must be R4, R8, BL or a 2-value key", LabelCol, t);

            t = schema[ScoreIndex].Type;
            if (t != NumberType.Float)
                throw Host.Except("Score column '{0}' has type '{1}' but must be R4", ScoreCol, t);

            if (_probIndex >= 0)
            {
                Host.Assert(!string.IsNullOrEmpty(_probCol));
                t = schema[_probIndex].Type;
                if (t != NumberType.Float)
                    throw Host.Except("Probability column '{0}' has type '{1}' but must be R4", _probCol, t);
            }
            else if (!_useRaw)
                throw Host.Except("Cannot compute the predicted label from the probability column because it does not exist");
        }
    }

    public sealed class BinaryClassifierMamlEvaluator : MamlEvaluatorBase
    {
        public class Arguments : ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Probability column name", ShortName = "prob")]
            public string ProbabilityColumn;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Probability value for classification thresholding")]
            public Single Threshold;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Use raw score value instead of probability for classification thresholding", ShortName = "useRawScore")]
            public bool UseRawScoreThreshold = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of samples to use for p/r curve generation. Specify 0 for no p/r curve generation", ShortName = "numpr")]
            public int NumRocExamples = 100000;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of samples to use for AUC calculation. If 0, AUC is not computed. If -1, the whole dataset is used", ShortName = "numauc")]
            public int MaxAucExamples = -1;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of samples to use for AUPRC calculation. Specify 0 for no AUPRC calculation", ShortName = "numauprc")]
            public int NumAuPrcExamples = 100000;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Precision-Recall results filename", ShortName = "pr", Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly)]
            public string PRFilename;
        }

        private const string FoldAccuracy = "OVERALL 0/1 ACCURACY";
        private const string FoldLogLoss = "LOG LOSS/instance";
        private const string FoldLogLosRed = "LOG-LOSS REDUCTION (RIG)";

        private readonly BinaryClassifierEvaluator _evaluator;

        private readonly string _prFileName;
        private readonly string _probCol;

        private protected override IEvaluator Evaluator => _evaluator;

        public BinaryClassifierMamlEvaluator(IHostEnvironment env, Arguments args)
            : base(args, env, MetadataUtils.Const.ScoreColumnKind.BinaryClassification, "BinaryClassifierMamlEvaluator")
        {
            Host.CheckValue(args, nameof(args));
            Utils.CheckOptionalUserDirectory(args.PRFilename, nameof(args.PRFilename));

            var evalArgs = new BinaryClassifierEvaluator.Arguments();
            evalArgs.Threshold = args.Threshold;
            evalArgs.UseRawScoreThreshold = args.UseRawScoreThreshold;
            evalArgs.MaxAucExamples = args.MaxAucExamples;
            evalArgs.NumRocExamples = string.IsNullOrEmpty(args.PRFilename) ? 0 : args.NumRocExamples;
            evalArgs.NumAuPrcExamples = args.NumAuPrcExamples;

            _prFileName = args.PRFilename;
            _probCol = args.ProbabilityColumn;
            _evaluator = new BinaryClassifierEvaluator(Host, evalArgs);
        }

        private protected override IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> GetInputColumnRolesCore(RoleMappedSchema schema)
        {
            var cols = base.GetInputColumnRolesCore(schema);

            var scoreCol = EvaluateUtils.GetScoreColumn(Host, schema.Schema, ScoreCol, nameof(Arguments.ScoreColumn),
                MetadataUtils.Const.ScoreColumnKind.BinaryClassification);

            // Get the optional probability column.
            var probCol = EvaluateUtils.GetOptAuxScoreColumn(Host, schema.Schema, _probCol, nameof(Arguments.ProbabilityColumn),
                scoreCol.Index, MetadataUtils.Const.ScoreValueKind.Probability, NumberType.Float.Equals);
            if (probCol.HasValue)
                cols = MetadataUtils.Prepend(cols, RoleMappedSchema.CreatePair(MetadataUtils.Const.ScoreValueKind.Probability, probCol.Value.Name));
            return cols;
        }

        private protected override void PrintFoldResultsCore(IChannel ch, Dictionary<string, IDataView> metrics)
        {
            ch.AssertValue(metrics);

            IDataView fold;
            if (!metrics.TryGetValue(MetricKinds.OverallMetrics, out fold))
                throw ch.Except("No overall metrics found");

            IDataView conf;
            if (!metrics.TryGetValue(MetricKinds.ConfusionMatrix, out conf))
                throw ch.Except("No overall metrics found");

            (string Source, string Name)[] cols =
            {
                (BinaryClassifierEvaluator.Accuracy, FoldAccuracy),
                (BinaryClassifierEvaluator.LogLoss, FoldLogLoss),
                (BinaryClassifierEvaluator.LogLossReduction, FoldLogLosRed)
            };

            var colsToKeep = new List<string>();
            colsToKeep.Add(FoldAccuracy);
            colsToKeep.Add(FoldLogLoss);
            colsToKeep.Add(BinaryClassifierEvaluator.Entropy);
            colsToKeep.Add(FoldLogLosRed);
            colsToKeep.Add(BinaryClassifierEvaluator.Auc);

            int index;
            if (fold.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.IsWeighted, out index))
                colsToKeep.Add(MetricKinds.ColumnNames.IsWeighted);
            if (fold.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.StratCol, out index))
                colsToKeep.Add(MetricKinds.ColumnNames.StratCol);
            if (fold.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.StratVal, out index))
                colsToKeep.Add(MetricKinds.ColumnNames.StratVal);

            fold = new ColumnCopyingTransformer(Host, cols).Transform(fold);

            // Select the columns that are specified in the Copy
            fold = ColumnSelectingTransformer.CreateKeep(Host, fold, colsToKeep.ToArray());

            string weightedConf;
            var unweightedConf = MetricWriter.GetConfusionTable(Host, conf, out weightedConf);
            string weightedFold;
            var unweightedFold = MetricWriter.GetPerFoldResults(Host, fold, out weightedFold);
            ch.Assert(string.IsNullOrEmpty(weightedConf) == string.IsNullOrEmpty(weightedFold));
            if (!string.IsNullOrEmpty(weightedConf))
            {
                ch.Info(MessageSensitivity.None, weightedConf);
                ch.Info(MessageSensitivity.None, weightedFold);
            }
            ch.Info(MessageSensitivity.None, unweightedConf);
            ch.Info(MessageSensitivity.None, unweightedFold);
        }

        private protected override IDataView GetOverallResultsCore(IDataView overall)
        {
            return ColumnSelectingTransformer.CreateDrop(Host, overall, BinaryClassifierEvaluator.Entropy);
        }

        private protected override void PrintAdditionalMetricsCore(IChannel ch, Dictionary<string, IDataView>[] metrics)
        {
            ch.AssertNonEmpty(metrics);

            if (!string.IsNullOrEmpty(_prFileName))
            {
                IDataView pr;
                if (!TryGetPrMetrics(metrics, out pr))
                    throw ch.Except("Did not find p/r metrics");

                ch.Trace(MessageSensitivity.None, "Saving p/r data view");
                // If the data view contains stratification columns, filter so that only the overall metrics
                // will be present, and drop them.
                pr = MetricWriter.GetNonStratifiedMetrics(Host, pr);
                MetricWriter.SavePerInstance(Host, ch, _prFileName, pr);
            }
        }

        public override IEnumerable<MetricColumn> GetOverallMetricColumns()
        {
            yield return new MetricColumn("Accuracy", BinaryClassifierEvaluator.Accuracy);
            yield return new MetricColumn("PosPrec", BinaryClassifierEvaluator.PosPrecName);
            yield return new MetricColumn("PosRecall", BinaryClassifierEvaluator.PosRecallName);
            yield return new MetricColumn("NegPrec", BinaryClassifierEvaluator.NegPrecName);
            yield return new MetricColumn("NegRecall", BinaryClassifierEvaluator.NegRecallName);
            yield return new MetricColumn("Auc", BinaryClassifierEvaluator.Auc);
            yield return new MetricColumn("LogLoss", BinaryClassifierEvaluator.LogLoss, MetricColumn.Objective.Minimize);
            yield return new MetricColumn("LogLossReduction", BinaryClassifierEvaluator.LogLossReduction);
            yield return new MetricColumn("F1", BinaryClassifierEvaluator.F1);
            yield return new MetricColumn("AuPrc", BinaryClassifierEvaluator.AuPrc);
        }

        // This method saves the p/r plots, and returns the p/r metrics data view.
        // In case there are results from multiple folds, they are averaged using
        // vertical averaging for the p/r plot, and appended using AppendRowsDataView for
        // the p/r data view.
        private bool TryGetPrMetrics(Dictionary<string, IDataView>[] metrics, out IDataView pr)
        {
            Host.AssertNonEmpty(metrics);
            pr = null;
            var prList = new List<IDataView>();
            for (int i = 0; i < metrics.Length; i++)
            {
                var dict = metrics[i];
                IDataView idv;
                if (!dict.TryGetValue(BinaryClassifierEvaluator.PrCurve, out idv))
                    return false;
                if (metrics.Length != 1)
                    idv = EvaluateUtils.AddFoldIndex(Host, idv, i, metrics.Length);
                else
                    pr = idv;
                prList.Add(idv);
            }
            if (metrics.Length != 1)
                pr = AppendRowsDataView.Create(Host, prList[0].Schema, prList.ToArray());

#if !CORECLR
            SavePrPlots(prList);
#endif
            return true;
        }

#if !CORECLR
        // Vertical averaging.
        private void SavePrPlots(List<IDataView> prList)
        {
            Host.AssertNonEmpty(prList);

            //PR curve
            var prPlot = new XYPlot();
            prPlot.LegendX = "Recall";
            prPlot.LegendY = "Precision";
            prPlot.MinX = 0;
            prPlot.MaxX = 1;
            prPlot.MinY = 0;
            prPlot.MaxY = 1;
            prPlot.InitializeChart(addLegend: false);

            var avgPoints = GetCurve(prList, BinaryClassifierEvaluator.Recall, BinaryClassifierEvaluator.Precision, 1);

            prPlot.AddCurveXY(avgPoints, "");
            if (prList.Count > 1)
            {
                var decimated = new List<XYPlot.XYPoint>();
                double currentX = 0.0;
                const double increment = 0.1;
                foreach (var t in avgPoints.OrderBy(q => q.X))
                {
                    if (t.X >= currentX)
                    {
                        decimated.Add(t);
                        currentX += increment;
                    }
                }
                prPlot.AddMarkerXYErr(decimated, "");
            }

            string basename = _prFileName;
            if (basename.Length > 4 && basename[basename.Length - 4] == '.')
                basename = basename.Substring(0, basename.Length - 4);

            prPlot.Save(basename + ".pr.jpg");

            avgPoints = GetCurve(prList, BinaryClassifierEvaluator.FalsePositiveRate, BinaryClassifierEvaluator.Recall);

            //ROC curve
            var rocPlot = new XYPlot();
            rocPlot.LegendX = "FPR";
            rocPlot.LegendY = "Recall=TPR";
            rocPlot.MinX = 0;
            rocPlot.MaxX = 1;
            rocPlot.MinY = 0;
            rocPlot.MaxY = 1;
            rocPlot.InitializeChart(addLegend: false);

            rocPlot.AddCurveXY(avgPoints, "");
            if (prList.Count > 1)
            {
                var decimated = new List<XYPlot.XYPoint>();
                double currentX = 0.0;
                double increment = 0.1;
                foreach (var t in avgPoints.OrderBy(q => q.X))
                {
                    if (t.X >= currentX)
                    {
                        decimated.Add(t);
                        currentX += increment;
                    }
                }
                rocPlot.AddMarkerXYErr(decimated, "");
            }
            rocPlot.Save(basename + ".roc.jpg");
        }

        private List<XYPlot.XYPoint> GetCurve(List<IDataView> prList, string xAxisName, string yAxisName, Double yInit = 0)
        {
            var cursors = new IRowCursor[prList.Count];
            var xGetters = new ValueGetter<Double>[prList.Count];
            var yGetters = new ValueGetter<Double>[prList.Count];
            for (int i = 0; i < prList.Count; i++)
            {
                int xIndex;
                if (!prList[i].Schema.TryGetColumnIndex(xAxisName, out xIndex))
                    throw Host.Except("Data view does not contain column '{0}'", xAxisName);
                int yIndex;
                if (!prList[i].Schema.TryGetColumnIndex(yAxisName, out yIndex))
                    throw Host.Except("Data view does not contain column '{0}'", yAxisName);

                cursors[i] = prList[i].GetRowCursor(col => col == xIndex || col == yIndex);
                xGetters[i] = cursors[i].GetGetter<Double>(xIndex);
                yGetters[i] = cursors[i].GetGetter<Double>(yIndex);
            }

            var avgPoints = new List<XYPlot.XYPoint>();

            var xPrev = new Double[prList.Count];
            var xCur = new Double[prList.Count];
            var yPrev = new Double[prList.Count];
            var yCur = new Double[prList.Count];
            if (yInit != 0)
            {
                for (int i = 0; i < yPrev.Length; i++)
                    yPrev[i] = yInit;
            }

            // Get the first points in all the curves.
            for (int i = 0; i < cursors.Length; i++)
            {
                if (cursors[i].MoveNext())
                {
                    xGetters[i](ref xCur[i]);
                    yGetters[i](ref yCur[i]);
                }
            }

            while (true)
            {
                // Find the next point as the point with the smallest x value, among the cursors that are not done.
                int argMin = -1;
                Double min = 2;
                for (int i = 0; i < cursors.Length; i++)
                {
                    if (cursors[i].State == CursorState.Done)
                        continue;

                    if (xCur[i] < min)
                    {
                        min = xCur[i];
                        argMin = i;
                    }
                }

                // We stop when all the cursors are done.
                if (argMin < 0)
                    break;

                // Calculate the average and std deviation of y value at x=min.
                // Use StdDev = Sqrt(Avg(y^2)-Avg(y)^2), then stdErr = stdDev/Sqrt(sample size)
                var yAvg = yCur[argMin];
                var yVar = yCur[argMin] * yCur[argMin];
                for (int i = 0; i < yCur.Length; i++)
                {
                    if (i == argMin)
                        continue;

                    var deltaPos = xCur[i] - xCur[argMin];
                    var deltaNeg = xCur[argMin] - xPrev[i];
                    var currentY = (deltaPos * yPrev[i] + deltaNeg * yCur[i]) / (deltaPos + deltaNeg);
                    yAvg += currentY;
                    yVar += currentY * currentY;
                }
                yAvg /= prList.Count;
                yVar = yVar / prList.Count - yAvg * yAvg;
                var yStdErr = Math.Sqrt(Math.Max(0.0, yVar)) / Math.Sqrt(prList.Count);
                avgPoints.Add(new XYPlot.XYPoint(min, yAvg, yStdErr));

                // Advanced the cursor whose x value was used for the current point.
                xPrev[argMin] = xCur[argMin];
                yPrev[argMin] = yCur[argMin];
                if (cursors[argMin].MoveNext())
                {
                    xGetters[argMin](ref xCur[argMin]);
                    yGetters[argMin](ref yCur[argMin]);
                }

                cursors[argMin].MoveNext();
            }

            foreach (var curs in cursors)
                curs.Dispose();

            return avgPoints;
        }
#endif
        private protected override IEnumerable<string> GetPerInstanceColumnsToSave(RoleMappedSchema schema)
        {
            Host.CheckValue(schema, nameof(schema));
            Host.CheckParam(schema.Label.HasValue, nameof(schema), "Schema must contain a label column");

            // The binary classifier evaluator outputs the label, score and probability columns.
            yield return schema.Label.Value.Name;
            var scoreCol = EvaluateUtils.GetScoreColumn(Host, schema.Schema, ScoreCol, nameof(Arguments.ScoreColumn),
                MetadataUtils.Const.ScoreColumnKind.BinaryClassification);
            yield return scoreCol.Name;
            var probCol = EvaluateUtils.GetOptAuxScoreColumn(Host, schema.Schema, _probCol, nameof(Arguments.ProbabilityColumn),
                scoreCol.Index, MetadataUtils.Const.ScoreValueKind.Probability, NumberType.Float.Equals);
            // Return the output columns. The LogLoss column is returned only if the probability column exists.
            if (probCol.HasValue)
            {
                yield return probCol.Value.Name;
                yield return BinaryPerInstanceEvaluator.LogLoss;
            }

            // REVIEW: Identify by metadata.
            int col;
            if (schema.Schema.TryGetColumnIndex("FeatureContributions", out col))
                yield return "FeatureContributions";

            yield return BinaryPerInstanceEvaluator.Assigned;
        }
    }

    internal static partial class Evaluate
    {
        [TlcModule.EntryPoint(Name = "Models.BinaryClassificationEvaluator", Desc = "Evaluates a binary classification scored dataset.")]
        public static CommonOutputs.ClassificationEvaluateOutput Binary(IHostEnvironment env, BinaryClassifierMamlEvaluator.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("EvaluateBinary");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            string label;
            string weight;
            string name;
            MatchColumns(host, input, out label, out weight, out name);
            IMamlEvaluator evaluator = new BinaryClassifierMamlEvaluator(host, input);
            var data = new RoleMappedData(input.Data, label, null, null, weight, name);
            var metrics = evaluator.Evaluate(data);

            var warnings = ExtractWarnings(host, metrics);
            var overallMetrics = ExtractOverallMetrics(host, metrics, evaluator);
            var perInstanceMetrics = evaluator.GetPerInstanceMetrics(data);
            var confusionMatrix = ExtractConfusionMatrix(host, metrics);

            return new CommonOutputs.ClassificationEvaluateOutput()
            {
                Warnings = warnings,
                OverallMetrics = overallMetrics,
                PerInstanceMetrics = perInstanceMetrics,
                ConfusionMatrix = confusionMatrix
            };
        }

        private static void MatchColumns(IHost host, MamlEvaluatorBase.ArgumentsBase input, out string label, out string weight, out string name)
        {
            var schema = input.Data.Schema;
            label = TrainUtils.MatchNameOrDefaultOrNull(host, schema,
                nameof(BinaryClassifierMamlEvaluator.Arguments.LabelColumn),
                input.LabelColumn, DefaultColumnNames.Label);
            weight = TrainUtils.MatchNameOrDefaultOrNull(host, schema,
                nameof(BinaryClassifierMamlEvaluator.Arguments.WeightColumn),
                input.WeightColumn, DefaultColumnNames.Weight);
            name = TrainUtils.MatchNameOrDefaultOrNull(host, schema,
                nameof(BinaryClassifierMamlEvaluator.Arguments.NameColumn),
                input.NameColumn, DefaultColumnNames.Name);
        }

        private static IDataView ExtractWarnings(IHost host, Dictionary<string, IDataView> metrics)
        {
            IDataView warnings;
            if (!metrics.TryGetValue(MetricKinds.Warnings, out warnings))
            {
                warnings = new EmptyDataView(host, SimpleSchemaUtils.Create(host,
                    new KeyValuePair<string, ColumnType>(MetricKinds.ColumnNames.WarningText, TextType.Instance)));
            }

            return warnings;
        }

        private static IDataView ExtractOverallMetrics(IHost host, Dictionary<string, IDataView> metrics, IMamlEvaluator evaluator)
        {
            IDataView overallMetrics;
            if (!metrics.TryGetValue(MetricKinds.OverallMetrics, out overallMetrics))
            {
                overallMetrics = new EmptyDataView(host,
                    SimpleSchemaUtils.Create(host,
                        evaluator.GetOverallMetricColumns()
                            .Select(mc => new KeyValuePair<string, ColumnType>(mc.LoadName, NumberType.R8))
                            .ToArray()));
            }

            return overallMetrics;
        }

        private static IDataView ExtractConfusionMatrix(IHost host, Dictionary<string, IDataView> metrics)
        {
            IDataView confusionMatrix;
            if (!metrics.TryGetValue(MetricKinds.ConfusionMatrix, out confusionMatrix))
            {
                confusionMatrix = new EmptyDataView(host,
                    SimpleSchemaUtils.Create(host, new KeyValuePair<string, ColumnType>(MetricKinds.ColumnNames.Count, NumberType.R8)));
            }

            return confusionMatrix;
        }
    }
}