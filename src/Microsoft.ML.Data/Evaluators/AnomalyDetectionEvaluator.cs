// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(typeof(AnomalyDetectionEvaluator), typeof(AnomalyDetectionEvaluator), typeof(AnomalyDetectionEvaluator.Arguments), typeof(SignatureEvaluator),
    "Anomaly Detection Evaluator", AnomalyDetectionEvaluator.LoadName, "AnomalyDetection", "Anomaly")]

[assembly: LoadableClass(typeof(AnomalyDetectionMamlEvaluator), typeof(AnomalyDetectionMamlEvaluator), typeof(AnomalyDetectionMamlEvaluator.Arguments), typeof(SignatureMamlEvaluator),
    "Anomaly Detection Evaluator", AnomalyDetectionEvaluator.LoadName, "AnomalyDetection", "Anomaly")]

namespace Microsoft.ML.Data
{
    [BestFriend]
    internal sealed class AnomalyDetectionEvaluator : EvaluatorBase<AnomalyDetectionEvaluator.Aggregator>
    {
        public sealed class Arguments
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Expected number of false positives")]
            public int K = 10;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Expected false positive rate")]
            public Double P = 0.01;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of top-scored predictions to display", ShortName = "n")]
            public int NumTopResults = 50;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to calculate metrics in one pass")]
            public bool Stream = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of samples to use for AUC calculation. If 0, AUC is not computed. If -1, the whole dataset is used", ShortName = "numauc")]
            public int MaxAucExamples = -1;
        }

        public const string LoadName = "AnomalyDetectionEvaluator";

        // Overall metrics.
        public static class OverallMetrics
        {
            public const string DrAtK = "DR @K FP";
            public const string DrAtPFpr = "DR @P FPR";
            public const string DrAtNumPos = "DR @NumPos";
            public const string NumAnomalies = "NumAnomalies";
            public const string ThreshAtK = "Threshold @K FP";
            public const string ThreshAtP = "Threshold @P FPR";
            public const string ThreshAtNumPos = "Threshold @NumPos";
        }

        /// <summary>
        /// The anomaly detection evaluator outputs a data view by this name, which contains the the examples
        /// with the top scores in the test set. It contains the three columns listed below, with each row corresponding
        /// to one test example.
        /// </summary>
        public const string TopKResults = "TopKResults";

        public static class TopKResultsColumns
        {
            public const string Instance = "Instance";
            public const string AnomalyScore = "Anomaly Score";
            public const string Label = "Label";
        }

        private readonly int _k;
        private readonly Double _p;
        private readonly int _numTopResults;
        private readonly bool _streaming;
        private readonly int _aucCount;

        public AnomalyDetectionEvaluator(IHostEnvironment env, Arguments args)
            : base(env, LoadName)
        {
            Host.CheckUserArg(args.K > 0, nameof(args.K), "Must be positive");
            Host.CheckUserArg(0 <= args.P && args.P <= 1, nameof(args.P), "Must be in [0,1]");
            Host.CheckUserArg(args.NumTopResults >= 0, nameof(args.NumTopResults), "Must be non-negative");
            Host.CheckUserArg(args.MaxAucExamples >= -1, nameof(args.MaxAucExamples), "Must be at least -1");

            _k = args.K;
            _p = args.P;
            _numTopResults = args.NumTopResults;
            _streaming = args.Stream;
            _aucCount = args.MaxAucExamples;
        }

        private protected override void CheckScoreAndLabelTypes(RoleMappedSchema schema)
        {
            var score = schema.GetUniqueColumn(AnnotationUtils.Const.ScoreValueKind.Score);
            var t = score.Type;
            if (t != NumberDataViewType.Single)
                throw Host.ExceptSchemaMismatch(nameof(schema), "score", score.Name, "float", t.ToString());
            Host.Check(schema.Label.HasValue, "Could not find the label column");
            t = schema.Label.Value.Type;
            if (t != NumberDataViewType.Single && t.GetKeyCount() != 2)
                throw Host.ExceptSchemaMismatch(nameof(schema), "label", schema.Label.Value.Name, "float or a KeyType with cardinality 2", t.ToString());
        }

        private protected override Aggregator GetAggregatorCore(RoleMappedSchema schema, string stratName)
        {
            return new Aggregator(Host, _aucCount, _numTopResults, _k, _p, _streaming, schema.Name == null ? -1 : schema.Name.Value.Index, stratName);
        }

        internal override IDataTransform GetPerInstanceMetricsCore(RoleMappedData data)
        {
            return NopTransform.CreateIfNeeded(Host, data.Data);
        }

        public override IEnumerable<MetricColumn> GetOverallMetricColumns()
        {
            yield return new MetricColumn("DrAtK", OverallMetrics.DrAtK, canBeWeighted: false);
            yield return new MetricColumn("DrAtPFpr", OverallMetrics.DrAtPFpr, canBeWeighted: false);
            yield return new MetricColumn("DrAtNumPos", OverallMetrics.DrAtNumPos, canBeWeighted: false);
            yield return new MetricColumn("NumAnomalies", OverallMetrics.NumAnomalies, MetricColumn.Objective.Info, canBeWeighted: false);
            yield return new MetricColumn("ThreshAtK", OverallMetrics.ThreshAtK, MetricColumn.Objective.Info, canBeWeighted: false);
            yield return new MetricColumn("ThreshAtP", OverallMetrics.ThreshAtP, MetricColumn.Objective.Info, canBeWeighted: false);
            yield return new MetricColumn("ThreshAtNumPos", OverallMetrics.ThreshAtNumPos, MetricColumn.Objective.Info, canBeWeighted: false);
        }

        private protected override void GetAggregatorConsolidationFuncs(Aggregator aggregator, AggregatorDictionaryBase[] dictionaries,
            out Action<uint, ReadOnlyMemory<char>, Aggregator> addAgg, out Func<Dictionary<string, IDataView>> consolidate)
        {
            var stratCol = new List<uint>();
            var stratVal = new List<ReadOnlyMemory<char>>();
            var auc = new List<Double>();
            var drAtK = new List<Double>();
            var drAtP = new List<Double>();
            var drAtNumAnomalies = new List<Double>();
            var thresholdAtK = new List<Single>();
            var thresholdAtP = new List<Single>();
            var thresholdAtNumAnomalies = new List<Single>();
            var numAnoms = new List<long>();

            var scores = new List<Single>();
            var labels = new List<Single>();
            var names = new List<ReadOnlyMemory<char>>();
            var topKStratCol = new List<uint>();
            var topKStratVal = new List<ReadOnlyMemory<char>>();

            bool hasStrats = Utils.Size(dictionaries) > 0;

            addAgg =
                (stratColKey, stratColVal, agg) =>
                {
                    agg.Finish();
                    stratCol.Add(stratColKey);
                    stratVal.Add(stratColVal);
                    auc.Add(agg.Auc);
                    drAtK.Add(agg.DrAtK);
                    drAtP.Add(agg.DrAtP);
                    drAtNumAnomalies.Add(agg.DrAtNumAnomalies);
                    thresholdAtK.Add(agg.ThresholdAtK);
                    thresholdAtP.Add(agg.ThresholdAtP);
                    thresholdAtNumAnomalies.Add(agg.ThresholdAtNumAnomalies);
                    numAnoms.Add(agg.AggCounters.NumAnomalies);

                    names.AddRange(agg.Names.Take(agg.NumTopExamples));
                    scores.AddRange(agg.Scores.Take(agg.NumTopExamples));
                    labels.AddRange(agg.Labels.Take(agg.NumTopExamples));

                    if (hasStrats)
                    {
                        topKStratCol.AddRange(agg.Scores.Select(x => stratColKey));
                        topKStratVal.AddRange(agg.Scores.Select(x => stratColVal));
                    }
                };

            consolidate =
                () =>
                {
                    var overallDvBldr = new ArrayDataViewBuilder(Host);
                    if (hasStrats)
                    {
                        overallDvBldr.AddColumn(MetricKinds.ColumnNames.StratCol, GetKeyValueGetter(dictionaries), (ulong)dictionaries.Length, stratCol.ToArray());
                        overallDvBldr.AddColumn(MetricKinds.ColumnNames.StratVal, TextDataViewType.Instance, stratVal.ToArray());
                    }
                    overallDvBldr.AddColumn(BinaryClassifierEvaluator.Auc, NumberDataViewType.Double, auc.ToArray());
                    overallDvBldr.AddColumn(OverallMetrics.DrAtK, NumberDataViewType.Double, drAtK.ToArray());
                    overallDvBldr.AddColumn(OverallMetrics.DrAtPFpr, NumberDataViewType.Double, drAtP.ToArray());
                    overallDvBldr.AddColumn(OverallMetrics.DrAtNumPos, NumberDataViewType.Double, drAtNumAnomalies.ToArray());
                    overallDvBldr.AddColumn(OverallMetrics.ThreshAtK, NumberDataViewType.Single, thresholdAtK.ToArray());
                    overallDvBldr.AddColumn(OverallMetrics.ThreshAtP, NumberDataViewType.Single, thresholdAtP.ToArray());
                    overallDvBldr.AddColumn(OverallMetrics.ThreshAtNumPos, NumberDataViewType.Single, thresholdAtNumAnomalies.ToArray());
                    overallDvBldr.AddColumn(OverallMetrics.NumAnomalies, NumberDataViewType.Int64, numAnoms.ToArray());

                    var topKdvBldr = new ArrayDataViewBuilder(Host);
                    if (hasStrats)
                    {
                        topKdvBldr.AddColumn(MetricKinds.ColumnNames.StratCol, GetKeyValueGetter(dictionaries), (ulong)dictionaries.Length, topKStratCol.ToArray());
                        topKdvBldr.AddColumn(MetricKinds.ColumnNames.StratVal, TextDataViewType.Instance, topKStratVal.ToArray());
                    }
                    topKdvBldr.AddColumn(TopKResultsColumns.Instance, TextDataViewType.Instance, names.ToArray());
                    topKdvBldr.AddColumn(TopKResultsColumns.AnomalyScore, NumberDataViewType.Single, scores.ToArray());
                    topKdvBldr.AddColumn(TopKResultsColumns.Label, NumberDataViewType.Single, labels.ToArray());

                    var result = new Dictionary<string, IDataView>();
                    result.Add(MetricKinds.OverallMetrics, overallDvBldr.GetDataView());
                    result.Add(TopKResults, topKdvBldr.GetDataView());

                    return result;
                };
        }

        public sealed class Aggregator : AggregatorBase
        {
            public abstract class CountersBase
            {
                protected readonly struct Info
                {
                    public readonly Single Label;
                    public readonly Single Score;

                    public Info(Single label, Single score)
                    {
                        Label = label;
                        Score = score;
                    }
                }

                public long NumAnomalies;
                protected long NumUpdates;

                protected readonly int K;
                protected readonly Double P;

                protected abstract long NumExamples { get; }

                protected CountersBase(int k, Double p)
                {
                    K = k;
                    P = p;
                }

                public void Update(Single label, Single score)
                {
                    Contracts.Assert(!Single.IsNaN(label));
                    UpdateCore(label, score);
                }

                protected abstract void UpdateCore(Single label, Single score);

                // Return detection rate @k, assign detection rate @p to drAtP, and detection rate @number of anomalies to drAtNumPos.
                public Double GetMetrics(int k, Double p, out Double drAtP, out Double drAtNumPos,
                    out Single thresholdAtK, out Single thresholdAtP, out Single thresholdAtNumPos)
                {
                    if (NumAnomalies == 0 || NumAnomalies == NumExamples)
                    {
                        thresholdAtK = thresholdAtP = thresholdAtNumPos = Single.NaN;
                        return drAtP = drAtNumPos = Double.NaN;
                    }

                    var sorted = GetSortedExamples();
                    drAtP = DetectionRate(sorted, (int)(p * (NumExamples - NumAnomalies)), out thresholdAtP);
                    drAtNumPos = sorted.Take((int)NumAnomalies).Count(result => result.Label > 0) / (Double)NumAnomalies;
                    thresholdAtNumPos = sorted.Take((int)NumAnomalies).Last().Score;
                    return DetectionRate(sorted, k, out thresholdAtK);
                }

                protected abstract IEnumerable<Info> GetSortedExamples();

                protected Double DetectionRate(IEnumerable<Info> sortedExamples, int maxFalsePositives, out Single threshold)
                {
                    int truePositives = 0;
                    int falsePositives = 0;
                    threshold = Single.PositiveInfinity;
                    foreach (var result in sortedExamples)
                    {
                        threshold = result.Score;
                        if (result.Label > 0)
                            ++truePositives;
                        else if (++falsePositives > maxFalsePositives)
                            break;
                    }
                    Contracts.Assert(truePositives <= NumAnomalies);

                    return (Double)truePositives / NumAnomalies;
                }

                public void UpdateCounts(Single label)
                {
                    NumUpdates++;
                    if (label > 0)
                        NumAnomalies++;
                }

                public virtual void FinishFirstPass()
                {
                }

                private protected Info[] ReverseHeap(Heap<Info> heap)
                {
                    var res = new Info[heap.Count];
                    while (heap.Count > 0)
                        res[heap.Count - 1] = heap.Pop();
                    return res;
                }
            }

            private sealed class OnePassCounters : CountersBase
            {
                private readonly List<Info> _examples;

                protected override long NumExamples { get { return _examples.Count; } }

                public OnePassCounters(int k, Double p)
                    : base(k, p)
                {
                    _examples = new List<Info>();
                }

                protected override void UpdateCore(float label, float score)
                {
                    _examples.Add(new Info(label, score));
                }

                protected override IEnumerable<Info> GetSortedExamples()
                {
                    var max = P * (NumUpdates - NumAnomalies);
                    if (max < K)
                        max = K;
                    if (max < NumAnomalies)
                        max = NumAnomalies;
                    var maxNumFalsePos = (int)Math.Ceiling(max);

                    // Create a heap with maxNumFalsePos label 0 elements.
                    var heap = new Heap<Info>((info1, info2) => info1.Score > info2.Score);
                    int numFalsePos = 0;
                    foreach (var example in _examples)
                    {
                        if (numFalsePos < maxNumFalsePos)
                        {
                            heap.Add(example);
                            if (example.Label <= 0)
                                numFalsePos++;
                            continue;
                        }
                        if (example.Score < heap.Top.Score)
                            continue;

                        heap.Add(example);
                        if (example.Label <= 0)
                        {
                            while (true)
                            {
                                if (heap.Pop().Label <= 0)
                                    break;
                            }
                        }
                    }
                    return ReverseHeap(heap);
                }
            }

            private sealed class TwoPassCounters : CountersBase
            {
                private readonly Heap<Info> _examples;
                private int _numFalsePos;

                private int _maxNumFalsePos;

                protected override long NumExamples { get { return NumUpdates; } }

                public TwoPassCounters(int k, Double p)
                    : base(k, p)
                {
                    _examples = new Heap<Info>((info1, info2) => info1.Score > info2.Score);
                }

                protected override void UpdateCore(Single label, Single score)
                {
                    if (_numFalsePos < _maxNumFalsePos)
                    {
                        _examples.Add(new Info(label, score));
                        if (label <= 0)
                            _numFalsePos++;
                        return;
                    }
                    if (score < _examples.Top.Score)
                        return;

                    _examples.Add(new Info(label, score));
                    if (label <= 0)
                    {
                        while (true)
                        {
                            if (_examples.Pop().Label <= 0)
                                break;
                        }
                    }
                }

                public override void FinishFirstPass()
                {
                    var max = P * (NumUpdates - NumAnomalies);
                    if (max < K)
                        max = K;
                    if (max < NumAnomalies)
                        max = NumAnomalies;
                    _maxNumFalsePos = (int)Math.Ceiling(max);
                }

                protected override IEnumerable<Info> GetSortedExamples()
                {
                    return ReverseHeap(_examples);
                }
            }

            private struct TopExamplesInfo
            {
                public Single Score;
                public Single Label;
                public string Name;
            }

            private readonly Heap<TopExamplesInfo> _topExamples;
            private readonly int _nameIndex;
            private readonly int _topK;
            private readonly int _k;
            private readonly Double _p;
            public readonly CountersBase AggCounters;
            private readonly bool _streaming;
            private readonly UnweightedAucAggregator _aucAggregator;
            public Double Auc;

            public Double DrAtK;
            public Double DrAtP;
            public Double DrAtNumAnomalies;
            public Single ThresholdAtK;
            public Single ThresholdAtP;
            public Single ThresholdAtNumAnomalies;

            private ValueGetter<Single> _labelGetter;
            private ValueGetter<Single> _scoreGetter;
            private ValueGetter<ReadOnlyMemory<char>> _nameGetter;

            public readonly ReadOnlyMemory<char>[] Names;
            public readonly Single[] Scores;
            public readonly Single[] Labels;
            public int NumTopExamples;

            public Aggregator(IHostEnvironment env, int reservoirSize, int topK, int k, Double p, bool streaming, int nameIndex, string stratName)
                : base(env, stratName)
            {
                Host.Assert(topK > 0);
                Host.Assert(nameIndex == -1 || nameIndex >= 0);
                Host.Assert(k > 0);

                _nameIndex = nameIndex;
                _topExamples = new Heap<TopExamplesInfo>((exampleA, exampleB) => exampleA.Score > exampleB.Score, topK);
                _topK = topK;
                _k = k;
                _p = p;
                _streaming = streaming;
                if (_streaming)
                    AggCounters = new OnePassCounters(_k, _p);
                else
                    AggCounters = new TwoPassCounters(_k, _p);
                _aucAggregator = new UnweightedAucAggregator(Host.Rand, reservoirSize);

                Names = new ReadOnlyMemory<char>[_topK];
                Scores = new Single[_topK];
                Labels = new Single[_topK];
            }

            private bool IsMainPass()
            {
                return _streaming ? PassNum == 0 : PassNum == 1;
            }

            protected override void FinishPassCore()
            {
                Host.Assert(!_streaming && PassNum < 2 || PassNum < 1);
                if (!_streaming && PassNum == 0)
                    AggCounters.FinishFirstPass();
            }

            public override bool IsActive()
            {
                return !_streaming && PassNum < 2 || PassNum < 1;
            }

            private void FinishOtherMetrics()
            {
                NumTopExamples = _topExamples.Count;
                while (_topExamples.Count > 0)
                {
                    Names[_topExamples.Count - 1] = _topExamples.Top.Name.AsMemory();
                    Scores[_topExamples.Count - 1] = _topExamples.Top.Score;
                    Labels[_topExamples.Count - 1] = _topExamples.Top.Label;
                    _topExamples.Pop();
                }
            }

            internal override void InitializeNextPass(DataViewRow row, RoleMappedSchema schema)
            {
                Host.Assert(!_streaming && PassNum < 2 || PassNum < 1);
                Host.Assert(schema.Label.HasValue);

                var score = schema.GetUniqueColumn(AnnotationUtils.Const.ScoreValueKind.Score);

                _labelGetter = RowCursorUtils.GetLabelGetter(row, schema.Label.Value.Index);
                _scoreGetter = row.GetGetter<float>(score);
                Host.AssertValue(_labelGetter);
                Host.AssertValue(_scoreGetter);

                if (IsMainPass())
                {
                    Host.Assert(_topExamples.Count == 0);
                    if (_nameIndex < 0)
                    {
                        int rowCounter = 0;
                        _nameGetter = (ref ReadOnlyMemory<char> dst) => dst = (rowCounter++).ToString().AsMemory();
                    }
                    else
                        _nameGetter = row.GetGetter<ReadOnlyMemory<char>>(row.Schema[_nameIndex]);
                }
            }

            public override void ProcessRow()
            {
                Single label = 0;
                _labelGetter(ref label);
                if (Single.IsNaN(label))
                {
                    if (PassNum == 0)
                        NumUnlabeledInstances++;
                    return;
                }

                Single score = 0;
                _scoreGetter(ref score);
                if (!FloatUtils.IsFinite(score))
                {
                    if (PassNum == 0)
                        NumBadScores++;
                    return;
                }

                if (PassNum == 0)
                    AggCounters.UpdateCounts(label);

                if (!IsMainPass())
                    return;

                _aucAggregator.ProcessRow(label, score);
                AggCounters.Update(label, score);

                var name = default(ReadOnlyMemory<char>);
                _nameGetter(ref name);
                if (_topExamples.Count >= _topK)
                {
                    var min = _topExamples.Top;
                    if (score < min.Score)
                        return;
                    _topExamples.Pop();
                }
                _topExamples.Add(new TopExamplesInfo() { Score = score, Label = label, Name = name.ToString() });
            }

            public void Finish()
            {
                Contracts.Assert(!IsActive());

                _aucAggregator.Finish();
                Double unweighted;
                Auc = _aucAggregator.ComputeWeightedAuc(out unweighted);
                DrAtK = AggCounters.GetMetrics(_k, _p, out DrAtP, out DrAtNumAnomalies, out ThresholdAtK, out ThresholdAtP, out ThresholdAtNumAnomalies);
                FinishOtherMetrics();
            }
        }

        /// <summary>
        /// Evaluates scored anomaly detection data.
        /// </summary>
        /// <param name="data">The scored data.</param>
        /// <param name="label">The name of the label column in <paramref name="data"/>.</param>
        /// <param name="score">The name of the score column in <paramref name="data"/>.</param>
        /// <param name="predictedLabel">The name of the predicted label column in <paramref name="data"/>.</param>
        /// <returns>The evaluation results for these outputs.</returns>
        internal AnomalyDetectionMetrics Evaluate(IDataView data, string label = DefaultColumnNames.Label, string score = DefaultColumnNames.Score,
            string predictedLabel = DefaultColumnNames.PredictedLabel)
        {
            Host.CheckValue(data, nameof(data));
            Host.CheckNonEmpty(label, nameof(label));
            Host.CheckNonEmpty(score, nameof(score));
            Host.CheckNonEmpty(predictedLabel, nameof(predictedLabel));

            var roles = new RoleMappedData(data, opt: false,
                RoleMappedSchema.ColumnRole.Label.Bind(label),
                RoleMappedSchema.CreatePair(AnnotationUtils.Const.ScoreValueKind.Score, score),
                RoleMappedSchema.CreatePair(AnnotationUtils.Const.ScoreValueKind.PredictedLabel, predictedLabel));

            var resultDict = ((IEvaluator)this).Evaluate(roles);
            Host.Assert(resultDict.ContainsKey(MetricKinds.OverallMetrics));
            var overall = resultDict[MetricKinds.OverallMetrics];

            AnomalyDetectionMetrics result;
            using (var cursor = overall.GetRowCursorForAllColumns())
            {
                var moved = cursor.MoveNext();
                Host.Assert(moved);
                result = new AnomalyDetectionMetrics(Host, cursor);
                moved = cursor.MoveNext();
                Host.Assert(!moved);
            }
            return result;
        }

    }

    [BestFriend]
    internal sealed class AnomalyDetectionMamlEvaluator : MamlEvaluatorBase
    {
        public sealed class Arguments : ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Expected number of false positives")]
            public int K = 10;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Expected false positive rate")]
            public Double P = 0.01;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of top-scored predictions to display", ShortName = "n")]
            public int NumTopResults = 50;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to calculate metrics in one pass")]
            public bool Stream = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of samples to use for AUC calculation. If 0, AUC is not computed. If -1, the whole dataset is used", ShortName = "numauc")]
            public int MaxAucExamples = -1;
        }

        private const string FoldDrAtKFormat = "Detection rate at {0} false positives";
        private const string FoldDrAtPFormat = "Detection rate at {0} false positive rate";
        private const string FoldDrAtNumAnomaliesFormat = "Detection rate at {0} positive predictions";

        private readonly AnomalyDetectionEvaluator _evaluator;
        private readonly int _topScored;
        private readonly int _k;
        private readonly Double _p;

        private protected override IEvaluator Evaluator => _evaluator;

        public AnomalyDetectionMamlEvaluator(IHostEnvironment env, Arguments args)
            : base(args, env, AnnotationUtils.Const.ScoreColumnKind.AnomalyDetection, "AnomalyDetectionMamlEvaluator")
        {
            var evalArgs = new AnomalyDetectionEvaluator.Arguments();
            evalArgs.K = _k = args.K;
            evalArgs.P = _p = args.P;
            evalArgs.NumTopResults = args.NumTopResults;
            _topScored = args.NumTopResults;
            evalArgs.Stream = args.Stream;
            evalArgs.MaxAucExamples = args.MaxAucExamples;
            _evaluator = new AnomalyDetectionEvaluator(Host, evalArgs);
        }

        private protected override void PrintFoldResultsCore(IChannel ch, Dictionary<string, IDataView> metrics)
        {
            IDataView top;
            if (!metrics.TryGetValue(AnomalyDetectionEvaluator.TopKResults, out top))
                throw Host.Except("Did not find the top-k results data view");
            var sb = new StringBuilder();
            using (var cursor = top.GetRowCursorForAllColumns())
            {
                DataViewSchema.Column? column = top.Schema.GetColumnOrNull(AnomalyDetectionEvaluator.TopKResultsColumns.Instance);
                if (!column.HasValue)
                    throw Host.ExceptSchemaMismatch(nameof(top.Schema), "instance", AnomalyDetectionEvaluator.TopKResultsColumns.Instance);
                var instanceGetter = cursor.GetGetter<ReadOnlyMemory<char>>(column.Value);

                column = top.Schema.GetColumnOrNull(AnomalyDetectionEvaluator.TopKResultsColumns.AnomalyScore);
                if (!column.HasValue)
                    throw Host.ExceptSchemaMismatch(nameof(top.Schema), "anomaly score", AnomalyDetectionEvaluator.TopKResultsColumns.AnomalyScore);
                var scoreGetter = cursor.GetGetter<Single>(column.Value);

                column = top.Schema.GetColumnOrNull(AnomalyDetectionEvaluator.TopKResultsColumns.Label);
                if (!column.HasValue)
                    throw Host.ExceptSchemaMismatch(nameof(top.Schema), "label", AnomalyDetectionEvaluator.TopKResultsColumns.Label);
                var labelGetter = cursor.GetGetter<Single>(column.Value);

                bool hasRows = false;
                while (cursor.MoveNext())
                {
                    if (!hasRows)
                    {
                        sb.AppendFormat("{0} Top-scored Results", _topScored);
                        sb.AppendLine();
                        sb.AppendLine("=================================================");
                        sb.AppendLine("Instance    Anomaly Score     Labeled");
                        hasRows = true;
                    }
                    var name = default(ReadOnlyMemory<char>);
                    Single score = 0;
                    Single label = 0;
                    instanceGetter(ref name);
                    scoreGetter(ref score);
                    labelGetter(ref label);
                    sb.AppendFormat("{0,-10}{1,12:G4}{2,12}", name, score, label);
                    sb.AppendLine();
                }
            }
            if (sb.Length > 0)
                ch.Info(MessageSensitivity.UserData, sb.ToString());

            IDataView overall;
            if (!metrics.TryGetValue(MetricKinds.OverallMetrics, out overall))
                throw Host.Except("No overall metrics found");

            // Find the number of anomalies, and the thresholds.
            DataViewSchema.Column? numAnom = overall.Schema.GetColumnOrNull(AnomalyDetectionEvaluator.OverallMetrics.NumAnomalies);
            if (numAnom == null || !numAnom.HasValue)
                throw Host.ExceptSchemaMismatch(nameof(overall.Schema), "number of anomalies", AnomalyDetectionEvaluator.OverallMetrics.NumAnomalies);

            int stratCol;
            var hasStrat = overall.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.StratCol, out stratCol);
            int stratVal;
            bool hasStratVals = overall.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.StratVal, out stratVal);
            Contracts.Assert(hasStrat == hasStratVals);
            long numAnomalies = 0;
            using (var cursor = overall.GetRowCursor(overall.Schema.Where(col => col.Name.Equals(AnomalyDetectionEvaluator.OverallMetrics.NumAnomalies)||
                (hasStrat && col.Name.Equals(MetricKinds.ColumnNames.StratCol)))))
            {
                var numAnomGetter = cursor.GetGetter<long>(numAnom.Value);
                ValueGetter<uint> stratGetter = null;
                if (hasStrat)
                {
                    var type = cursor.Schema[stratCol].Type;
                    stratGetter = RowCursorUtils.GetGetterAs<uint>(type, cursor, stratCol);
                }
                bool foundRow = false;
                while (cursor.MoveNext())
                {
                    uint strat = 0;
                    if (stratGetter != null)
                        stratGetter(ref strat);
                    if (strat > 0)
                        continue;
                    if (foundRow)
                        throw Host.Except("Found multiple non-stratified rows in overall results data view");
                    foundRow = true;
                    numAnomGetter(ref numAnomalies);
                }
            }

            var kFormatName = string.Format(FoldDrAtKFormat, _k);
            var pFormatName = string.Format(FoldDrAtPFormat, _p);
            var numAnomName = string.Format(FoldDrAtNumAnomaliesFormat, numAnomalies);

            (string name, string source)[] cols =
            {
                (kFormatName, AnomalyDetectionEvaluator.OverallMetrics.DrAtK),
                (pFormatName, AnomalyDetectionEvaluator.OverallMetrics.DrAtPFpr),
                (numAnomName, AnomalyDetectionEvaluator.OverallMetrics.DrAtNumPos)
            };

            // List of columns to keep, note that the order specified determines the order of the output
            var colsToKeep = new List<string>();
            colsToKeep.Add(kFormatName);
            colsToKeep.Add(pFormatName);
            colsToKeep.Add(numAnomName);
            colsToKeep.Add(AnomalyDetectionEvaluator.OverallMetrics.ThreshAtK);
            colsToKeep.Add(AnomalyDetectionEvaluator.OverallMetrics.ThreshAtP);
            colsToKeep.Add(AnomalyDetectionEvaluator.OverallMetrics.ThreshAtNumPos);
            colsToKeep.Add(BinaryClassifierEvaluator.Auc);

            overall = new ColumnCopyingTransformer(Host, cols).Transform(overall);
            IDataView fold = ColumnSelectingTransformer.CreateKeep(Host, overall, colsToKeep.ToArray());

            string weightedFold;
            ch.Info(MetricWriter.GetPerFoldResults(Host, fold, out weightedFold));
        }

        private protected override IDataView GetOverallResultsCore(IDataView overall)
        {
            return ColumnSelectingTransformer.CreateDrop(Host,
                                                    overall,
                                                    AnomalyDetectionEvaluator.OverallMetrics.NumAnomalies,
                                                    AnomalyDetectionEvaluator.OverallMetrics.ThreshAtK,
                                                    AnomalyDetectionEvaluator.OverallMetrics.ThreshAtP,
                                                    AnomalyDetectionEvaluator.OverallMetrics.ThreshAtNumPos);
        }

        private protected override IEnumerable<string> GetPerInstanceColumnsToSave(RoleMappedSchema schema)
        {
            Host.CheckValue(schema, nameof(schema));
            Host.CheckParam(schema.Label.HasValue, nameof(schema), "Data must contain a label column");

            // The anomaly detection evaluator outputs the label and the score.
            yield return schema.Label.Value.Name;
            var scoreCol = EvaluateUtils.GetScoreColumn(Host, schema.Schema, ScoreCol, nameof(Arguments.ScoreColumn),
                AnnotationUtils.Const.ScoreColumnKind.AnomalyDetection);
            yield return scoreCol.Name;

            // No additional output columns.
        }

        public override IEnumerable<MetricColumn> GetOverallMetricColumns()
        {
            yield return new MetricColumn("DrAtK", AnomalyDetectionEvaluator.OverallMetrics.DrAtK, canBeWeighted: false);
            yield return new MetricColumn("DrAtPFpr", AnomalyDetectionEvaluator.OverallMetrics.DrAtPFpr, canBeWeighted: false);
            yield return new MetricColumn("DrAtNumPos", AnomalyDetectionEvaluator.OverallMetrics.DrAtNumPos, canBeWeighted: false);
        }
    }

    internal static partial class Evaluate
    {
        [TlcModule.EntryPoint(Name = "Models.AnomalyDetectionEvaluator", Desc = "Evaluates an anomaly detection scored dataset.")]
        public static CommonOutputs.CommonEvaluateOutput AnomalyDetection(IHostEnvironment env, AnomalyDetectionMamlEvaluator.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("EvaluateAnomalyDetection");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            string label;
            string weight;
            string name;
            MatchColumns(host, input, out label, out weight, out name);
            IMamlEvaluator evaluator = new AnomalyDetectionMamlEvaluator(host, input);
            var data = new RoleMappedData(input.Data, label, null, null, weight, name);
            var metrics = evaluator.Evaluate(data);

            var warnings = ExtractWarnings(host, metrics);
            var overallMetrics = ExtractOverallMetrics(host, metrics, evaluator);
            var perInstanceMetrics = evaluator.GetPerInstanceMetrics(data);

            return new CommonOutputs.CommonEvaluateOutput()
            {
                Warnings = warnings,
                OverallMetrics = overallMetrics,
                PerInstanceMetrics = perInstanceMetrics
            };
        }
    }
}
