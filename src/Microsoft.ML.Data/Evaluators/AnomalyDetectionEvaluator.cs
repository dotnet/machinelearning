// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;

[assembly: LoadableClass(typeof(AnomalyDetectionEvaluator), typeof(AnomalyDetectionEvaluator), typeof(AnomalyDetectionEvaluator.Arguments), typeof(SignatureEvaluator),
    "Anomaly Detection Evaluator", AnomalyDetectionEvaluator.LoadName, "AnomalyDetection", "Anomaly")]

[assembly: LoadableClass(typeof(AnomalyDetectionMamlEvaluator), typeof(AnomalyDetectionMamlEvaluator), typeof(AnomalyDetectionMamlEvaluator.Arguments), typeof(SignatureMamlEvaluator),
    "Anomaly Detection Evaluator", AnomalyDetectionEvaluator.LoadName, "AnomalyDetection", "Anomaly")]

namespace Microsoft.ML.Runtime.Data
{
    using Float = System.Single;

    public sealed class AnomalyDetectionEvaluator : EvaluatorBase<AnomalyDetectionEvaluator.Aggregator>
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

        protected override void CheckScoreAndLabelTypes(RoleMappedSchema schema)
        {
            var score = schema.GetUniqueColumn(MetadataUtils.Const.ScoreValueKind.Score);
            var t = score.Type;
            if (t != NumberType.Float)
                throw Host.Except("Score column '{0}' has type '{1}' but must be R4", score, t).MarkSensitive(MessageSensitivity.Schema);
            Host.Check(schema.Label != null, "Could not find the label column");
            t = schema.Label.Type;
            if (t != NumberType.Float && t.KeyCount != 2)
                throw Host.Except("Label column '{0}' has type '{1}' but must be R4 or a 2-value key", schema.Label.Name, t).MarkSensitive(MessageSensitivity.Schema);
        }

        protected override Aggregator GetAggregatorCore(RoleMappedSchema schema, string stratName)
        {
            return new Aggregator(Host, _aucCount, _numTopResults, _k, _p, _streaming, schema.Name == null ? -1 : schema.Name.Index, stratName);
        }

        public override IDataTransform GetPerInstanceMetrics(RoleMappedData data)
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

        protected override void GetAggregatorConsolidationFuncs(Aggregator aggregator, AggregatorDictionaryBase[] dictionaries,
            out Action<uint, DvText, Aggregator> addAgg, out Func<Dictionary<string, IDataView>> consolidate)
        {
            var stratCol = new List<uint>();
            var stratVal = new List<DvText>();
            var auc = new List<Double>();
            var drAtK = new List<Double>();
            var drAtP = new List<Double>();
            var drAtNumAnomalies = new List<Double>();
            var thresholdAtK = new List<Single>();
            var thresholdAtP = new List<Single>();
            var thresholdAtNumAnomalies = new List<Single>();
            var numAnoms = new List<DvInt8>();

            var scores = new List<Single>();
            var labels = new List<Single>();
            var names = new List<DvText>();
            var topKStratCol = new List<uint>();
            var topKStratVal = new List<DvText>();

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
                        overallDvBldr.AddColumn(MetricKinds.ColumnNames.StratCol, GetKeyValueGetter(dictionaries), 0, dictionaries.Length, stratCol.ToArray());
                        overallDvBldr.AddColumn(MetricKinds.ColumnNames.StratVal, TextType.Instance, stratVal.ToArray());
                    }
                    overallDvBldr.AddColumn(BinaryClassifierEvaluator.Auc, NumberType.R8, auc.ToArray());
                    overallDvBldr.AddColumn(OverallMetrics.DrAtK, NumberType.R8, drAtK.ToArray());
                    overallDvBldr.AddColumn(OverallMetrics.DrAtPFpr, NumberType.R8, drAtP.ToArray());
                    overallDvBldr.AddColumn(OverallMetrics.DrAtNumPos, NumberType.R8, drAtNumAnomalies.ToArray());
                    overallDvBldr.AddColumn(OverallMetrics.ThreshAtK, NumberType.R4, thresholdAtK.ToArray());
                    overallDvBldr.AddColumn(OverallMetrics.ThreshAtP, NumberType.R4, thresholdAtP.ToArray());
                    overallDvBldr.AddColumn(OverallMetrics.ThreshAtNumPos, NumberType.R4, thresholdAtNumAnomalies.ToArray());
                    overallDvBldr.AddColumn(OverallMetrics.NumAnomalies, NumberType.I8, numAnoms.ToArray());

                    var topKdvBldr = new ArrayDataViewBuilder(Host);
                    if (hasStrats)
                    {
                        topKdvBldr.AddColumn(MetricKinds.ColumnNames.StratCol, GetKeyValueGetter(dictionaries), 0, dictionaries.Length, topKStratCol.ToArray());
                        topKdvBldr.AddColumn(MetricKinds.ColumnNames.StratVal, TextType.Instance, topKStratVal.ToArray());
                    }
                    topKdvBldr.AddColumn(TopKResultsColumns.Instance, TextType.Instance, names.ToArray());
                    topKdvBldr.AddColumn(TopKResultsColumns.AnomalyScore, NumberType.R4, scores.ToArray());
                    topKdvBldr.AddColumn(TopKResultsColumns.Label, NumberType.R4, labels.ToArray());

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
                protected struct Info
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

                protected IEnumerable<Info> ReverseHeap(Heap<Info> heap)
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
            private ValueGetter<DvText> _nameGetter;

            public readonly DvText[] Names;
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

                Names = new DvText[_topK];
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
                    Names[_topExamples.Count - 1] = new DvText(_topExamples.Top.Name);
                    Scores[_topExamples.Count - 1] = _topExamples.Top.Score;
                    Labels[_topExamples.Count - 1] = _topExamples.Top.Label;
                    _topExamples.Pop();
                }
            }

            public override void InitializeNextPass(IRow row, RoleMappedSchema schema)
            {
                Host.Assert(!_streaming && PassNum < 2 || PassNum < 1);
                Host.AssertValue(schema.Label);

                var score = schema.GetUniqueColumn(MetadataUtils.Const.ScoreValueKind.Score);

                _labelGetter = RowCursorUtils.GetLabelGetter(row, schema.Label.Index);
                _scoreGetter = row.GetGetter<float>(score.Index);
                Host.AssertValue(_labelGetter);
                Host.AssertValue(_scoreGetter);

                if (IsMainPass())
                {
                    Host.Assert(_topExamples.Count == 0);
                    if (_nameIndex < 0)
                    {
                        int rowCounter = 0;
                        _nameGetter = (ref DvText dst) => dst = new DvText((rowCounter++).ToString());
                    }
                    else
                        _nameGetter = row.GetGetter<DvText>(_nameIndex);
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

                var name = default(DvText);
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
    }

    public sealed class AnomalyDetectionMamlEvaluator : MamlEvaluatorBase
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

        protected override IEvaluator Evaluator => _evaluator;

        public AnomalyDetectionMamlEvaluator(IHostEnvironment env, Arguments args)
            : base(args, env, MetadataUtils.Const.ScoreColumnKind.AnomalyDetection, "AnomalyDetectionMamlEvaluator")
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

        protected override void PrintFoldResultsCore(IChannel ch, Dictionary<string, IDataView> metrics)
        {
            IDataView top;
            if (!metrics.TryGetValue(AnomalyDetectionEvaluator.TopKResults, out top))
                throw Host.Except("Did not find the top-k results data view");
            var sb = new StringBuilder();
            using (var cursor = top.GetRowCursor(col => true))
            {
                int index;
                if (!top.Schema.TryGetColumnIndex(AnomalyDetectionEvaluator.TopKResultsColumns.Instance, out index))
                    throw Host.Except("Data view does not contain the 'Instance' column");
                var instanceGetter = cursor.GetGetter<DvText>(index);
                if (!top.Schema.TryGetColumnIndex(AnomalyDetectionEvaluator.TopKResultsColumns.AnomalyScore, out index))
                    throw Host.Except("Data view does not contain the 'Anomaly Score' column");
                var scoreGetter = cursor.GetGetter<Single>(index);
                if (!top.Schema.TryGetColumnIndex(AnomalyDetectionEvaluator.TopKResultsColumns.Label, out index))
                    throw Host.Except("Data view does not contain the 'Label' column");
                var labelGetter = cursor.GetGetter<Single>(index);

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
                    var name = default(DvText);
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
            int numAnomIndex;
            if (!overall.Schema.TryGetColumnIndex(AnomalyDetectionEvaluator.OverallMetrics.NumAnomalies, out numAnomIndex))
                throw Host.Except("Could not find the 'NumAnomalies' column");

            int stratCol;
            var hasStrat = overall.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.StratCol, out stratCol);
            int stratVal;
            bool hasStratVals = overall.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.StratVal, out stratVal);
            Contracts.Assert(hasStrat == hasStratVals);
            DvInt8 numAnomalies = 0;
            using (var cursor = overall.GetRowCursor(col => col == numAnomIndex ||
                (hasStrat && col == stratCol)))
            {
                var numAnomGetter = cursor.GetGetter<DvInt8>(numAnomIndex);
                ValueGetter<uint> stratGetter = null;
                if (hasStrat)
                {
                    var type = cursor.Schema.GetColumnType(stratCol);
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

            var args = new ChooseColumnsTransform.Arguments();
            var cols = new List<ChooseColumnsTransform.Column>()
                {
                    new ChooseColumnsTransform.Column()
                    {
                        Name = string.Format(FoldDrAtKFormat, _k),
                        Source = AnomalyDetectionEvaluator.OverallMetrics.DrAtK
                    },
                    new ChooseColumnsTransform.Column()
                    {
                        Name = string.Format(FoldDrAtPFormat, _p),
                        Source = AnomalyDetectionEvaluator.OverallMetrics.DrAtPFpr
                    },
                    new ChooseColumnsTransform.Column()
                    {
                        Name = string.Format(FoldDrAtNumAnomaliesFormat, numAnomalies),
                        Source=AnomalyDetectionEvaluator.OverallMetrics.DrAtNumPos
                    },
                    new ChooseColumnsTransform.Column()
                    {
                        Name=AnomalyDetectionEvaluator.OverallMetrics.ThreshAtK
                    },
                    new ChooseColumnsTransform.Column()
                    {
                        Name=AnomalyDetectionEvaluator.OverallMetrics.ThreshAtP
                    },
                    new ChooseColumnsTransform.Column()
                    {
                        Name=AnomalyDetectionEvaluator.OverallMetrics.ThreshAtNumPos
                    },
                    new ChooseColumnsTransform.Column()
                    {
                        Name = BinaryClassifierEvaluator.Auc
                    }
                };

            args.Column = cols.ToArray();
            IDataView fold = new ChooseColumnsTransform(Host, args, overall);
            string weightedFold;
            ch.Info(MetricWriter.GetPerFoldResults(Host, fold, out weightedFold));
        }

        protected override IDataView GetOverallResultsCore(IDataView overall)
        {
            var args = new DropColumnsTransform.Arguments();
            args.Column = new[]
            {
                AnomalyDetectionEvaluator.OverallMetrics.NumAnomalies,
                AnomalyDetectionEvaluator.OverallMetrics.ThreshAtK,
                AnomalyDetectionEvaluator.OverallMetrics.ThreshAtP,
                AnomalyDetectionEvaluator.OverallMetrics.ThreshAtNumPos
            };
            return new DropColumnsTransform(Host, args, overall);
        }

        protected override IEnumerable<string> GetPerInstanceColumnsToSave(RoleMappedSchema schema)
        {
            Host.CheckValue(schema, nameof(schema));
            Host.CheckValue(schema.Label, nameof(schema), "Data must contain a label column");

            // The anomaly detection evaluator outputs the label and the score.
            yield return schema.Label.Name;
            var scoreInfo = EvaluateUtils.GetScoreColumnInfo(Host, schema.Schema, ScoreCol, nameof(Arguments.ScoreColumn),
                MetadataUtils.Const.ScoreColumnKind.AnomalyDetection);
            yield return scoreInfo.Name;

            // No additional output columns.
        }

        public override IEnumerable<MetricColumn> GetOverallMetricColumns()
        {
            yield return new MetricColumn("DrAtK", AnomalyDetectionEvaluator.OverallMetrics.DrAtK, canBeWeighted: false);
            yield return new MetricColumn("DrAtPFpr", AnomalyDetectionEvaluator.OverallMetrics.DrAtPFpr, canBeWeighted: false);
            yield return new MetricColumn("DrAtNumPos", AnomalyDetectionEvaluator.OverallMetrics.DrAtNumPos, canBeWeighted: false);
        }
    }

    public static partial class Evaluate
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
            var evaluator = new AnomalyDetectionMamlEvaluator(host, input);
            var data = TrainUtils.CreateExamples(input.Data, label, null, null, weight, name);
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
