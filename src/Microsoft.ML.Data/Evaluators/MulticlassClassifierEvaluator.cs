// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

[assembly: LoadableClass(typeof(MultiClassClassifierEvaluator), typeof(MultiClassClassifierEvaluator), typeof(MultiClassClassifierEvaluator.Arguments), typeof(SignatureEvaluator),
    "Multi-Class Classifier Evaluator", MultiClassClassifierEvaluator.LoadName, "MultiClassClassifier", "MultiClass")]

[assembly: LoadableClass(typeof(MultiClassMamlEvaluator), typeof(MultiClassMamlEvaluator), typeof(MultiClassMamlEvaluator.Arguments), typeof(SignatureMamlEvaluator),
    "Multi-Class Classifier Evaluator", MultiClassClassifierEvaluator.LoadName, "MultiClassClassifier", "MultiClass")]

// This is for deserialization of the per-instance transform.
[assembly: LoadableClass(typeof(MultiClassPerInstanceEvaluator), null, typeof(SignatureLoadRowMapper),
    "", MultiClassPerInstanceEvaluator.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    public sealed class MultiClassClassifierEvaluator : RowToRowEvaluatorBase<MultiClassClassifierEvaluator.Aggregator>
    {
        public sealed class Arguments
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Output top K accuracy", ShortName = "topkacc")]
            public int? OutputTopKAcc;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Use the textual class label names in the report, if available", ShortName = "n")]
            public bool Names = true;
        }

        public const string AccuracyMicro = "Accuracy(micro-avg)";
        public const string AccuracyMacro = "Accuracy(macro-avg)";
        public const string TopKAccuracy = "Top K accuracy";
        public const string PerClassLogLoss = "Per class log-loss";
        public const string LogLoss = "Log-loss";
        public const string LogLossReduction = "Log-loss reduction";

        public enum Metrics
        {
            [EnumValueDisplay(MultiClassClassifierEvaluator.AccuracyMicro)]
            AccuracyMicro,
            [EnumValueDisplay(MultiClassClassifierEvaluator.AccuracyMacro)]
            AccuracyMacro,
            [EnumValueDisplay(MultiClassClassifierEvaluator.LogLoss)]
            LogLoss,
            [EnumValueDisplay(MultiClassClassifierEvaluator.LogLossReduction)]
            LogLossReduction,
        }

        public const string LoadName = "MultiClassClassifierEvaluator";

        private readonly int? _outputTopKAcc;
        private readonly bool _names;

        public MultiClassClassifierEvaluator(IHostEnvironment env, Arguments args)
            : base(env, LoadName)
        {
            Host.AssertValue(args, "args");
            Host.CheckUserArg(args.OutputTopKAcc == null || args.OutputTopKAcc > 0, nameof(args.OutputTopKAcc));
            _outputTopKAcc = args.OutputTopKAcc;
            _names = args.Names;
        }

        protected override void CheckScoreAndLabelTypes(RoleMappedSchema schema)
        {
            var score = schema.GetUniqueColumn(MetadataUtils.Const.ScoreValueKind.Score);
            var t = score.Type;
            if (t.VectorSize < 2 || t.ItemType != NumberType.Float)
                throw Host.Except("Score column '{0}' has type {1} but must be a vector of two or more items of type R4", score.Name, t);
            Host.Check(schema.Label != null, "Could not find the label column");
            t = schema.Label.Type;
            if (t != NumberType.Float && t.KeyCount <= 0)
                throw Host.Except("Label column '{0}' has type {1} but must be a float or a known-cardinality key", schema.Label.Name, t);
        }

        protected override Aggregator GetAggregatorCore(RoleMappedSchema schema, string stratName)
        {
            var score = schema.GetUniqueColumn(MetadataUtils.Const.ScoreValueKind.Score);
            Host.Assert(score.Type.VectorSize > 0);
            int numClasses = score.Type.VectorSize;
            var classNames = GetClassNames(schema);
            return new Aggregator(Host, classNames, numClasses, schema.Weight != null, _outputTopKAcc, stratName);
        }

        private ReadOnlyMemory<char>[] GetClassNames(RoleMappedSchema schema)
        {
            ReadOnlyMemory<char>[] names;
            // Get the label names from the score column if they exist, or use the default names.
            var scoreInfo = schema.GetUniqueColumn(MetadataUtils.Const.ScoreValueKind.Score);
            var mdType = schema.Schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.SlotNames, scoreInfo.Index);
            var labelNames = default(VBuffer<ReadOnlyMemory<char>>);
            if (mdType != null && mdType.IsKnownSizeVector && mdType.ItemType.IsText)
            {
                schema.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, scoreInfo.Index, ref labelNames);
                names = new ReadOnlyMemory<char>[labelNames.Length];
                labelNames.CopyTo(names);
            }
            else
            {
                var score = schema.GetColumns(MetadataUtils.Const.ScoreValueKind.Score);
                Host.Assert(Utils.Size(score) == 1);
                Host.Assert(score[0].Type.VectorSize > 0);
                int numClasses = score[0].Type.VectorSize;
                names = Enumerable.Range(0, numClasses).Select(i => i.ToString().AsMemory()).ToArray();
            }
            return names;
        }

        protected override IRowMapper CreatePerInstanceRowMapper(RoleMappedSchema schema)
        {
            Host.CheckParam(schema.Label != null, nameof(schema), "Schema must contain a label column");
            var scoreInfo = schema.GetUniqueColumn(MetadataUtils.Const.ScoreValueKind.Score);
            int numClasses = scoreInfo.Type.VectorSize;
            return new MultiClassPerInstanceEvaluator(Host, schema.Schema, scoreInfo, schema.Label.Name);
        }

        public override IEnumerable<MetricColumn> GetOverallMetricColumns()
        {
            yield return new MetricColumn("AccuracyMicro", AccuracyMicro);
            yield return new MetricColumn("AccuracyMacro", AccuracyMacro);
            yield return new MetricColumn("TopKAccuracy", TopKAccuracy);
            yield return new MetricColumn("LogLoss<class name>", PerClassLogLoss, MetricColumn.Objective.Minimize,
                isVector: true, namePattern: new Regex(string.Format(@"^{0}(?<class>.+)", LogLoss), RegexOptions.IgnoreCase),
                groupName: "class", nameFormat: string.Format("{0} (class {{0}})", PerClassLogLoss));
            yield return new MetricColumn("LogLoss", LogLoss, MetricColumn.Objective.Minimize);
            yield return new MetricColumn("LogLossReduction", LogLossReduction);
        }

        protected override void GetAggregatorConsolidationFuncs(Aggregator aggregator, AggregatorDictionaryBase[] dictionaries,
            out Action<uint, ReadOnlyMemory<char>, Aggregator> addAgg, out Func<Dictionary<string, IDataView>> consolidate)
        {
            var stratCol = new List<uint>();
            var stratVal = new List<ReadOnlyMemory<char>>();
            var isWeighted = new List<bool>();
            var microAcc = new List<double>();
            var macroAcc = new List<double>();
            var logLoss = new List<double>();
            var logLossRed = new List<double>();
            var topKAcc = new List<double>();
            var perClassLogLoss = new List<double[]>();
            var counts = new List<double[]>();
            var weights = new List<double[]>();
            var confStratCol = new List<uint>();
            var confStratVal = new List<ReadOnlyMemory<char>>();

            bool hasStrats = Utils.Size(dictionaries) > 0;
            bool hasWeight = aggregator.Weighted;

            addAgg =
                (stratColKey, stratColVal, agg) =>
                {
                    Host.Check(agg.Weighted == hasWeight, "All aggregators must either be weighted or unweighted");
                    Host.Check((agg.UnweightedCounters.OutputTopKAcc > 0) == (aggregator.UnweightedCounters.OutputTopKAcc > 0),
                        "All aggregators must either compute top-k accuracy or not compute top-k accuracy");

                    stratCol.Add(stratColKey);
                    stratVal.Add(stratColVal);
                    isWeighted.Add(false);
                    microAcc.Add(agg.UnweightedCounters.MicroAvgAccuracy);
                    macroAcc.Add(agg.UnweightedCounters.MacroAvgAccuracy);
                    logLoss.Add(agg.UnweightedCounters.LogLoss);
                    logLossRed.Add(agg.UnweightedCounters.Reduction);
                    if (agg.UnweightedCounters.OutputTopKAcc > 0)
                        topKAcc.Add(agg.UnweightedCounters.TopKAccuracy);
                    perClassLogLoss.Add(agg.UnweightedCounters.PerClassLogLoss);

                    confStratCol.AddRange(agg.UnweightedCounters.ConfusionTable.Select(x => stratColKey));
                    confStratVal.AddRange(agg.UnweightedCounters.ConfusionTable.Select(x => stratColVal));
                    counts.AddRange(agg.UnweightedCounters.ConfusionTable);

                    if (agg.Weighted)
                    {
                        stratCol.Add(stratColKey);
                        stratVal.Add(stratColVal);
                        isWeighted.Add(true);
                        microAcc.Add(agg.WeightedCounters.MicroAvgAccuracy);
                        macroAcc.Add(agg.WeightedCounters.MacroAvgAccuracy);
                        logLoss.Add(agg.WeightedCounters.LogLoss);
                        logLossRed.Add(agg.WeightedCounters.Reduction);
                        if (agg.WeightedCounters.OutputTopKAcc > 0)
                            topKAcc.Add(agg.WeightedCounters.TopKAccuracy);
                        perClassLogLoss.Add(agg.WeightedCounters.PerClassLogLoss);
                        weights.AddRange(agg.WeightedCounters.ConfusionTable);
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
                    overallDvBldr.AddColumn(AccuracyMicro, NumberType.R8, microAcc.ToArray());
                    overallDvBldr.AddColumn(AccuracyMacro, NumberType.R8, macroAcc.ToArray());
                    overallDvBldr.AddColumn(LogLoss, NumberType.R8, logLoss.ToArray());
                    overallDvBldr.AddColumn(LogLossReduction, NumberType.R8, logLossRed.ToArray());
                    if (aggregator.UnweightedCounters.OutputTopKAcc > 0)
                        overallDvBldr.AddColumn(TopKAccuracy, NumberType.R8, topKAcc.ToArray());
                    overallDvBldr.AddColumn(PerClassLogLoss, aggregator.GetSlotNames, NumberType.R8, perClassLogLoss.ToArray());

                    var confDvBldr = new ArrayDataViewBuilder(Host);
                    if (hasStrats)
                    {
                        confDvBldr.AddColumn(MetricKinds.ColumnNames.StratCol, GetKeyValueGetter(dictionaries), 0, dictionaries.Length, confStratCol.ToArray());
                        confDvBldr.AddColumn(MetricKinds.ColumnNames.StratVal, TextType.Instance, confStratVal.ToArray());
                    }
                    ValueGetter<VBuffer<ReadOnlyMemory<char>>> getSlotNames =
                        (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                            dst = new VBuffer<ReadOnlyMemory<char>>(aggregator.ClassNames.Length, aggregator.ClassNames);
                    confDvBldr.AddColumn(MetricKinds.ColumnNames.Count, getSlotNames, NumberType.R8, counts.ToArray());

                    if (hasWeight)
                        confDvBldr.AddColumn(MetricKinds.ColumnNames.Weight, getSlotNames, NumberType.R8, weights.ToArray());

                    var result = new Dictionary<string, IDataView>
                    {
                        { MetricKinds.OverallMetrics, overallDvBldr.GetDataView() },
                        { MetricKinds.ConfusionMatrix, confDvBldr.GetDataView() }
                    };
                    return result;
                };
        }

        public sealed class Aggregator : AggregatorBase
        {
            public sealed class Counters
            {
                private readonly int _numClasses;
                public readonly int? OutputTopKAcc;

                private double _totalLogLoss;
                private double _numInstances;
                private double _numCorrect;
                private double _numCorrectTopK;
                private readonly double[] _sumWeightsOfClass;
                private readonly double[] _totalPerClassLogLoss;
                public readonly double[][] ConfusionTable;

                public double MicroAvgAccuracy { get { return _numInstances > 0 ? _numCorrect / _numInstances : 0; } }
                public double MacroAvgAccuracy
                {
                    get
                    {
                        if (_numInstances == 0)
                            return 0;
                        double macroAvgAccuracy = 0;
                        int countOfNonEmptyClasses = 0;
                        for (int i = 0; i < _numClasses; ++i)
                        {
                            if (_sumWeightsOfClass[i] > 0)
                            {
                                countOfNonEmptyClasses++;
                                macroAvgAccuracy += ConfusionTable[i][i] / _sumWeightsOfClass[i];
                            }
                        }

                        return countOfNonEmptyClasses > 0 ? macroAvgAccuracy / countOfNonEmptyClasses : 0;
                    }
                }

                public double LogLoss { get { return _numInstances > 0 ? _totalLogLoss / _numInstances : 0; } }

                public double Reduction
                {
                    get
                    {
                        // reduction -- prior log loss is entropy
                        double entropy = 0;
                        for (int i = 0; i < _numClasses; ++i)
                        {
                            if (_sumWeightsOfClass[i] != 0)
                                entropy += _sumWeightsOfClass[i] * Math.Log(_sumWeightsOfClass[i] / _numInstances);
                        }
                        entropy /= -_numInstances;
                        return 100 * (entropy - LogLoss) / entropy;
                    }
                }

                public double TopKAccuracy { get { return _numInstances > 0 ? _numCorrectTopK / _numInstances : 0; } }

                // The per class average log loss is calculated by dividing the weighted sum of the log loss of examples
                // in each class by the total weight of examples in that class.
                public double[] PerClassLogLoss
                {
                    get
                    {
                        var res = new double[_totalPerClassLogLoss.Length];
                        for (int i = 0; i < _totalPerClassLogLoss.Length; i++)
                            res[i] = _sumWeightsOfClass[i] > 0 ? _totalPerClassLogLoss[i] / _sumWeightsOfClass[i] : 0;
                        return res;
                    }
                }

                public Counters(int numClasses, int? outputTopKAcc)
                {
                    _numClasses = numClasses;
                    OutputTopKAcc = outputTopKAcc;

                    _sumWeightsOfClass = new double[numClasses];
                    _totalPerClassLogLoss = new double[numClasses];
                    ConfusionTable = new double[numClasses][];
                    for (int i = 0; i < ConfusionTable.Length; i++)
                        ConfusionTable[i] = new double[numClasses];
                }

                public void Update(int[] indices, double loglossCurr, int label, float weight)
                {
                    Contracts.Assert(Utils.Size(indices) == _numClasses);

                    int assigned = indices[0];

                    _numInstances += weight;

                    if (label < _numClasses)
                        _sumWeightsOfClass[label] += weight;

                    _totalLogLoss += loglossCurr * weight;

                    if (label < _numClasses)
                        _totalPerClassLogLoss[label] += loglossCurr * weight;

                    if (assigned == label)
                    {
                        _numCorrect += weight;
                        ConfusionTable[label][label] += weight;
                        _numCorrectTopK += weight;
                    }
                    else if (label < _numClasses)
                    {
                        if (OutputTopKAcc > 0)
                        {
                            int idx = Array.IndexOf(indices, label);
                            if (0 <= idx && idx < OutputTopKAcc)
                                _numCorrectTopK += weight;
                        }
                        ConfusionTable[label][assigned] += weight;
                    }
                }
            }

            private ValueGetter<float> _labelGetter;
            private ValueGetter<VBuffer<float>> _scoreGetter;
            private ValueGetter<float> _weightGetter;

            private VBuffer<float> _scores;
            private readonly float[] _scoresArr;
            private int[] _indicesArr;

            private const float Epsilon = (float)1e-15;

            public readonly Counters UnweightedCounters;
            public readonly Counters WeightedCounters;
            public readonly bool Weighted;

            private long _numUnknownClassInstances;
            private long _numNegOrNonIntegerLabels;

            public readonly ReadOnlyMemory<char>[] ClassNames;

            public Aggregator(IHostEnvironment env, ReadOnlyMemory<char>[] classNames, int scoreVectorSize, bool weighted, int? outputTopKAcc, string stratName)
                : base(env, stratName)
            {
                Host.Assert(outputTopKAcc == null || outputTopKAcc > 0);
                Host.Assert(scoreVectorSize > 0);
                Host.Assert(Utils.Size(classNames) == scoreVectorSize);

                _scoresArr = new float[scoreVectorSize];
                UnweightedCounters = new Counters(scoreVectorSize, outputTopKAcc);
                Weighted = weighted;
                WeightedCounters = Weighted ? new Counters(scoreVectorSize, outputTopKAcc) : null;
                ClassNames = classNames;
            }

            public override void InitializeNextPass(IRow row, RoleMappedSchema schema)
            {
                Host.Assert(PassNum < 1);
                Host.AssertValue(schema.Label);

                var score = schema.GetUniqueColumn(MetadataUtils.Const.ScoreValueKind.Score);
                Host.Assert(score.Type.VectorSize == _scoresArr.Length);
                _labelGetter = RowCursorUtils.GetLabelGetter(row, schema.Label.Index);
                _scoreGetter = row.GetGetter<VBuffer<float>>(score.Index);
                Host.AssertValue(_labelGetter);
                Host.AssertValue(_scoreGetter);

                if (schema.Weight != null)
                    _weightGetter = row.GetGetter<float>(schema.Weight.Index);
            }

            public override void ProcessRow()
            {
                float label = 0;
                _labelGetter(ref label);
                if (float.IsNaN(label))
                {
                    NumUnlabeledInstances++;
                    return;
                }
                if (label < 0 || label != (int)label)
                {
                    _numNegOrNonIntegerLabels++;
                    return;
                }

                _scoreGetter(ref _scores);
                Host.Check(_scores.Length == _scoresArr.Length);

                if (VBufferUtils.HasNaNs(in _scores) || VBufferUtils.HasNonFinite(in _scores))
                {
                    NumBadScores++;
                    return;
                }
                _scores.CopyTo(_scoresArr);
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

                // Sort classes by prediction strength.
                // Use stable OrderBy instead of Sort(), which may give different results on different machines.
                if (Utils.Size(_indicesArr) < _scoresArr.Length)
                    _indicesArr = new int[_scoresArr.Length];
                int j = 0;
                foreach (var index in Enumerable.Range(0, _scoresArr.Length).OrderByDescending(i => _scoresArr[i]))
                    _indicesArr[j++] = index;

                var intLabel = (int)label;

                // log-loss
                double logloss;
                if (intLabel < _scoresArr.Length)
                {
                    // REVIEW: This assumes that the predictions are probabilities, not just relative scores
                    // for the classes. Is this a correct assumption?
                    float p = Math.Min(1, Math.Max(Epsilon, _scoresArr[intLabel]));
                    logloss = -Math.Log(p);
                }
                else
                {
                    // Penalize logloss if the label was not seen during training
                    logloss = -Math.Log(Epsilon);
                    _numUnknownClassInstances++;
                }

                UnweightedCounters.Update(_indicesArr, logloss, intLabel, 1);
                if (WeightedCounters != null)
                    WeightedCounters.Update(_indicesArr, logloss, intLabel, weight);
            }

            protected override List<string> GetWarningsCore()
            {
                var warnings = base.GetWarningsCore();
                if (_numUnknownClassInstances > 0)
                {
                    warnings.Add(string.Format(
                        "Found {0} test instances with class values not seen in the training set. LogLoss is reported higher than usual because of these instances.",
                        _numUnknownClassInstances));
                }
                if (_numNegOrNonIntegerLabels > 0)
                {
                    warnings.Add(string.Format(
                        "Found {0} test instances with labels that are either negative or non integers. These instances were ignored",
                        _numNegOrNonIntegerLabels));
                }
                return warnings;
            }

            public void GetSlotNames(ref VBuffer<ReadOnlyMemory<char>> slotNames)
            {
                var values = slotNames.Values;
                if (Utils.Size(values) < ClassNames.Length)
                    values = new ReadOnlyMemory<char>[ClassNames.Length];

                for (int i = 0; i < ClassNames.Length; i++)
                    values[i] = string.Format("(class {0})", ClassNames[i]).AsMemory();
                slotNames = new VBuffer<ReadOnlyMemory<char>>(ClassNames.Length, values);
            }
        }

        public sealed class Result
        {
            /// <summary>
            /// Gets the micro-average accuracy of the model.
            /// </summary>
            /// <remarks>
            /// The micro-average is the fraction of instances predicted correctly.
            ///
            /// The micro-average metric weighs each class according to the number of instances that belong
            /// to it in the dataset.
            /// </remarks>
            public double AccuracyMicro { get; }

            /// <summary>
            /// Gets the macro-average accuracy of the model.
            /// </summary>
            /// <remarks>
            /// The macro-average is computed by taking the average over all the classes of the fraction
            /// of correct predictions in this class (the number of correctly predicted instances in the class,
            /// divided by the total number of instances in the class).
            ///
            /// The macro-average metric gives the same weight to each class, no matter how many instances from
            /// that class the dataset contains.
            /// </remarks>
            public double AccuracyMacro { get; }

            /// <summary>
            /// Gets the average log-loss of the classifier.
            /// </summary>
            /// <remarks>
            /// The log-loss metric, is computed as follows:
            /// LL = - (1/m) * sum( log(p[i]))
            /// where m is the number of instances in the test set.
            /// p[i] is the probability returned by the classifier if the instance belongs to class 1,
            /// and 1 minus the probability returned by the classifier if the instance belongs to class 0.
            /// </remarks>
            public double LogLoss { get; }

            /// <summary>
            /// Gets the log-loss reduction (also known as relative log-loss, or reduction in information gain - RIG)
            /// of the classifier.
            /// </summary>
            /// <remarks>
            /// The log-loss reduction is scaled relative to a classifier that predicts the prior for every example:
            /// (LL(prior) - LL(classifier)) / LL(prior)
            /// This metric can be interpreted as the advantage of the classifier over a random prediction.
            /// For example, if the RIG equals 20, it can be interpreted as "the probability of a correct prediction is
            /// 20% better than random guessing".
            /// </remarks>
            public double LogLossReduction { get; private set; }

            /// <summary>
            /// If positive, this is the top-K for which the <see cref="TopKAccuracy"/> is calculated.
            /// </summary>
            public int TopK { get; }

            /// <summary>
            /// If <see cref="TopK"/> is positive, this is the relative number of examples where
            /// the true label is one of the top k predicted labels by the predictor.
            /// </summary>
            public double TopKAccuracy { get; }

            /// <summary>
            /// Gets the log-loss of the classifier for each class.
            /// </summary>
            /// <remarks>
            /// The log-loss metric, is computed as follows:
            /// LL = - (1/m) * sum( log(p[i]))
            /// where m is the number of instances in the test set.
            /// p[i] is the probability returned by the classifier if the instance belongs to the class,
            /// and 1 minus the probability returned by the classifier if the instance does not belong to the class.
            /// </remarks>
            public double[] PerClassLogLoss { get; }

            internal Result(IExceptionContext ectx, IRow overallResult, int topK)
            {
                double FetchDouble(string name) => RowCursorUtils.Fetch<double>(ectx, overallResult, name);
                AccuracyMicro = FetchDouble(MultiClassClassifierEvaluator.AccuracyMicro);
                AccuracyMacro = FetchDouble(MultiClassClassifierEvaluator.AccuracyMacro);
                LogLoss = FetchDouble(MultiClassClassifierEvaluator.LogLoss);
                LogLossReduction = FetchDouble(MultiClassClassifierEvaluator.LogLossReduction);
                TopK = topK;
                if (topK > 0)
                    TopKAccuracy = FetchDouble(MultiClassClassifierEvaluator.TopKAccuracy);

                var perClassLogLoss = RowCursorUtils.Fetch<VBuffer<double>>(ectx, overallResult, MultiClassClassifierEvaluator.PerClassLogLoss);
                PerClassLogLoss = new double[perClassLogLoss.Length];
                perClassLogLoss.CopyTo(PerClassLogLoss);
            }
        }

        /// <summary>
        /// Evaluates scored multiclass classification data.
        /// </summary>
        /// <param name="data">The scored data.</param>
        /// <param name="label">The name of the label column in <paramref name="data"/>.</param>
        /// <param name="score">The name of the score column in <paramref name="data"/>.</param>
        /// <param name="predictedLabel">The name of the predicted label column in <paramref name="data"/>.</param>
        /// <returns>The evaluation results for these outputs.</returns>
        public Result Evaluate(IDataView data, string label, string score, string predictedLabel)
        {
            Host.CheckValue(data, nameof(data));
            Host.CheckNonEmpty(label, nameof(label));
            Host.CheckNonEmpty(score, nameof(score));
            Host.CheckNonEmpty(predictedLabel, nameof(predictedLabel));

            var roles = new RoleMappedData(data, opt: false,
                RoleMappedSchema.ColumnRole.Label.Bind(label),
                RoleMappedSchema.CreatePair(MetadataUtils.Const.ScoreValueKind.Score, score),
                RoleMappedSchema.CreatePair(MetadataUtils.Const.ScoreValueKind.PredictedLabel, predictedLabel));

            var resultDict = Evaluate(roles);
            Host.Assert(resultDict.ContainsKey(MetricKinds.OverallMetrics));
            var overall = resultDict[MetricKinds.OverallMetrics];

            Result result;
            using (var cursor = overall.GetRowCursor(i => true))
            {
                var moved = cursor.MoveNext();
                Host.Assert(moved);
                result = new Result(Host, cursor, _outputTopKAcc ?? 0);
                moved = cursor.MoveNext();
                Host.Assert(!moved);
            }
            return result;
        }

    }

    public sealed class MultiClassPerInstanceEvaluator : PerInstanceEvaluatorBase
    {
        public const string LoaderSignature = "MulticlassPerInstance";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "MLTIINST",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Serialize the class names
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(MultiClassPerInstanceEvaluator).Assembly.FullName);
        }

        private const int AssignedCol = 0;
        private const int LogLossCol = 1;
        private const int SortedScoresCol = 2;
        private const int SortedClassesCol = 3;

        private const uint VerInitial = 0x00010001;

        public const string Assigned = "Assigned";
        public const string LogLoss = "Log-loss";
        public const string SortedScores = "SortedScores";
        public const string SortedClasses = "SortedClasses";

        private const float Epsilon = (float)1e-15;

        private readonly int _numClasses;
        private readonly ReadOnlyMemory<char>[] _classNames;
        private readonly ColumnType[] _types;

        public MultiClassPerInstanceEvaluator(IHostEnvironment env, Schema schema, ColumnInfo scoreInfo, string labelCol)
            : base(env, schema, Contracts.CheckRef(scoreInfo, nameof(scoreInfo)).Name, labelCol)
        {
            CheckInputColumnTypes(schema);

            _numClasses = scoreInfo.Type.VectorSize;
            _types = new ColumnType[4];

            if (schema.HasSlotNames(ScoreIndex, _numClasses))
            {
                var classNames = default(VBuffer<ReadOnlyMemory<char>>);
                schema.GetMetadata(MetadataUtils.Kinds.SlotNames, ScoreIndex, ref classNames);
                _classNames = new ReadOnlyMemory<char>[_numClasses];
                classNames.CopyTo(_classNames);
            }
            else
                _classNames = Utils.BuildArray(_numClasses, i => i.ToString().AsMemory());

            var key = new KeyType(DataKind.U4, 0, _numClasses);
            _types[AssignedCol] = key;
            _types[LogLossCol] = NumberType.R8;
            _types[SortedScoresCol] = new VectorType(NumberType.R4, _numClasses);
            _types[SortedClassesCol] = new VectorType(key, _numClasses);
        }

        private MultiClassPerInstanceEvaluator(IHostEnvironment env, ModelLoadContext ctx, ISchema schema)
            : base(env, ctx, schema)
        {
            CheckInputColumnTypes(schema);

            // *** Binary format **
            // base
            // int: number of classes
            // int[]: Ids of the class names

            _numClasses = ctx.Reader.ReadInt32();
            Host.CheckDecode(_numClasses > 0);
            if (ctx.Header.ModelVerWritten > VerInitial)
            {
                _classNames = new ReadOnlyMemory<char>[_numClasses];
                for (int i = 0; i < _numClasses; i++)
                    _classNames[i] = ctx.LoadNonEmptyString().AsMemory();
            }
            else
                _classNames = Utils.BuildArray(_numClasses, i => i.ToString().AsMemory());

            _types = new ColumnType[4];
            var key = new KeyType(DataKind.U4, 0, _numClasses);
            _types[AssignedCol] = key;
            _types[LogLossCol] = NumberType.R8;
            _types[SortedScoresCol] = new VectorType(NumberType.R4, _numClasses);
            _types[SortedClassesCol] = new VectorType(key, _numClasses);
        }

        public static MultiClassPerInstanceEvaluator Create(IHostEnvironment env, ModelLoadContext ctx, ISchema schema)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new MultiClassPerInstanceEvaluator(env, ctx, schema);
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format **
            // base
            // int: number of classes
            // int[]: Ids of the class names

            base.Save(ctx);
            Host.Assert(_numClasses > 0);
            ctx.Writer.Write(_numClasses);
            for (int i = 0; i < _numClasses; i++)
                ctx.SaveNonEmptyString(_classNames[i].ToString());
        }

        public override Func<int, bool> GetDependencies(Func<int, bool> activeOutput)
        {
            Host.Assert(ScoreIndex >= 0);
            Host.Assert(LabelIndex >= 0);

            // The score column is needed if any of the outputs are active. The label column is needed only
            // if the log-loss output is active.
            return
                col =>
                    col == LabelIndex && activeOutput(LogLossCol) ||
                    col == ScoreIndex && (activeOutput(AssignedCol) || activeOutput(SortedScoresCol) ||
                    activeOutput(SortedClassesCol) || activeOutput(LogLossCol));
        }

        public override Delegate[] CreateGetters(IRow input, Func<int, bool> activeOutput, out Action disposer)
        {
            disposer = null;

            var getters = new Delegate[4];

            if (!activeOutput(AssignedCol) && !activeOutput(SortedClassesCol) && !activeOutput(SortedScoresCol) && !activeOutput(LogLossCol))
                return getters;

            long cachedPosition = -1;
            VBuffer<float> scores = default(VBuffer<float>);
            float label = 0;
            var scoresArr = new float[_numClasses];
            int[] sortedIndices = new int[_numClasses];

            var labelGetter = activeOutput(LogLossCol) ? RowCursorUtils.GetLabelGetter(input, LabelIndex) :
                (ref float dst) => dst = float.NaN;
            var scoreGetter = input.GetGetter<VBuffer<float>>(ScoreIndex);
            Action updateCacheIfNeeded =
                () =>
                {
                    if (cachedPosition != input.Position)
                    {
                        labelGetter(ref label);
                        scoreGetter(ref scores);
                        scores.CopyTo(scoresArr);
                        int j = 0;
                        foreach (var index in Enumerable.Range(0, scoresArr.Length).OrderByDescending(i => scoresArr[i]))
                            sortedIndices[j++] = index;
                        cachedPosition = input.Position;
                    }
                };

            if (activeOutput(AssignedCol))
            {
                ValueGetter<uint> assignedFn =
                    (ref uint dst) =>
                    {
                        updateCacheIfNeeded();
                        dst = (uint)sortedIndices[0] + 1;
                    };
                getters[AssignedCol] = assignedFn;
            }

            if (activeOutput(SortedScoresCol))
            {
                ValueGetter<VBuffer<float>> topKScoresFn =
                    (ref VBuffer<float> dst) =>
                    {
                        updateCacheIfNeeded();
                        var values = dst.Values;
                        if (Utils.Size(values) < _numClasses)
                            values = new float[_numClasses];
                        for (int i = 0; i < _numClasses; i++)
                            values[i] = scores.GetItemOrDefault(sortedIndices[i]);
                        dst = new VBuffer<float>(_numClasses, values);
                    };
                getters[SortedScoresCol] = topKScoresFn;
            }

            if (activeOutput(SortedClassesCol))
            {
                ValueGetter<VBuffer<uint>> topKClassesFn =
                    (ref VBuffer<uint> dst) =>
                    {
                        updateCacheIfNeeded();
                        var values = dst.Values;
                        if (Utils.Size(values) < _numClasses)
                            values = new uint[_numClasses];
                        for (int i = 0; i < _numClasses; i++)
                            values[i] = (uint)sortedIndices[i] + 1;
                        dst = new VBuffer<uint>(_numClasses, values);
                    };
                getters[SortedClassesCol] = topKClassesFn;
            }

            if (activeOutput(LogLossCol))
            {
                ValueGetter<double> logLossFn =
                    (ref double dst) =>
                    {
                        updateCacheIfNeeded();
                        if (float.IsNaN(label))
                        {
                            dst = double.NaN;
                            return;
                        }

                        int intLabel = (int)label;
                        if (intLabel < _numClasses)
                        {
                            float p = Math.Min(1, Math.Max(Epsilon, scoresArr[intLabel]));
                            dst = -Math.Log(p);
                            return;
                        }
                        // Penalize logloss if the label was not seen during training
                        dst = -Math.Log(Epsilon);
                    };
                getters[LogLossCol] = logLossFn;
            }
            return getters;
        }

        public override Schema.Column[] GetOutputColumns()
        {
            var infos = new Schema.Column[4];

            var assignedColKeyValues = new Schema.Metadata.Builder();
            assignedColKeyValues.AddKeyValues(_numClasses, TextType.Instance, CreateKeyValueGetter());
            infos[AssignedCol] = new Schema.Column(Assigned, _types[AssignedCol], assignedColKeyValues.GetMetadata());

            infos[LogLossCol] = new Schema.Column(LogLoss, _types[LogLossCol], null);

            var sortedScores = new Schema.Metadata.Builder();
            sortedScores.AddSlotNames(_numClasses, CreateSlotNamesGetter(_numClasses, "Score"));

            var sortedClasses = new Schema.Metadata.Builder();
            sortedClasses.AddSlotNames(_numClasses, CreateSlotNamesGetter(_numClasses, "Class"));
            sortedClasses.AddKeyValues(_numClasses, TextType.Instance, CreateKeyValueGetter());

            infos[SortedScoresCol] = new Schema.Column(SortedScores, _types[SortedScoresCol], sortedScores.GetMetadata());
            infos[SortedClassesCol] = new Schema.Column(SortedClasses, _types[SortedClassesCol], sortedClasses.GetMetadata());
            return infos;
        }

        // REVIEW: Figure out how to avoid having the column name in each slot name.
        private ValueGetter<VBuffer<ReadOnlyMemory<char>>> CreateSlotNamesGetter(int numTopClasses, string suffix)
        {
            return
                (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                {
                    var values = dst.Values;
                    if (Utils.Size(values) < numTopClasses)
                        values = new ReadOnlyMemory<char>[numTopClasses];
                    for (int i = 1; i <= numTopClasses; i++)
                        values[i - 1] = string.Format("#{0} {1}", i, suffix).AsMemory();
                    dst = new VBuffer<ReadOnlyMemory<char>>(numTopClasses, values);
                };
        }

        private ValueGetter<VBuffer<ReadOnlyMemory<char>>> CreateKeyValueGetter()
        {
            return
                (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                {
                    var values = dst.Values;
                    if (Utils.Size(values) < _numClasses)
                        values = new ReadOnlyMemory<char>[_numClasses];
                    for (int i = 0; i < _numClasses; i++)
                        values[i] = _classNames[i];
                    dst = new VBuffer<ReadOnlyMemory<char>>(_numClasses, values);
                };
        }

        private void CheckInputColumnTypes(ISchema schema)
        {
            Host.AssertNonEmpty(ScoreCol);
            Host.AssertNonEmpty(LabelCol);

            var t = schema.GetColumnType(ScoreIndex);
            if (t.VectorSize < 2 || t.ItemType != NumberType.Float)
                throw Host.Except("Score column '{0}' has type '{1}' but must be a vector of two or more items of type R4", ScoreCol, t);
            t = schema.GetColumnType(LabelIndex);
            if (t != NumberType.Float && t.KeyCount <= 0)
                throw Host.Except("Label column '{0}' has type '{1}' but must be a float or a known-cardinality key", LabelCol, t);
        }
    }

    public sealed class MultiClassMamlEvaluator : MamlEvaluatorBase
    {
        public class Arguments : ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Output top-K accuracy.", ShortName = "topkacc")]
            public int? OutputTopKAcc;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Output top-K classes.", ShortName = "topk")]
            public int NumTopClassesToOutput = 3;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum number of classes in confusion matrix.", ShortName = "nccf")]
            public int NumClassesConfusionMatrix = 10;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Output per class statistics and confusion matrix.", ShortName = "opcs")]
            public bool OutputPerClassStatistics = false;
        }

        private const string TopKAccuracyFormat = "Top-{0}-accuracy";

        private readonly bool _outputPerClass;
        private readonly int _numTopClasses;
        private readonly int _numConfusionTableClasses;
        private readonly int? _outputTopKAcc;
        private readonly MultiClassClassifierEvaluator _evaluator;

        protected override IEvaluator Evaluator { get { return _evaluator; } }

        public MultiClassMamlEvaluator(IHostEnvironment env, Arguments args)
            : base(args, env, MetadataUtils.Const.ScoreColumnKind.MultiClassClassification, "MultiClassMamlEvaluator")
        {
            Host.CheckValue(args, nameof(args));
            // REVIEW: why do we need to insist on at least 2?
            Host.CheckUserArg(2 <= args.NumTopClassesToOutput, nameof(args.NumTopClassesToOutput));
            Host.CheckUserArg(2 <= args.NumClassesConfusionMatrix, nameof(args.NumClassesConfusionMatrix));
            Host.CheckUserArg(args.OutputTopKAcc == null || args.OutputTopKAcc > 0, nameof(args.OutputTopKAcc));
            Host.CheckUserArg(2 <= args.NumClassesConfusionMatrix, nameof(args.NumClassesConfusionMatrix));

            _numTopClasses = args.NumTopClassesToOutput;
            _outputPerClass = args.OutputPerClassStatistics;
            _numConfusionTableClasses = args.NumClassesConfusionMatrix;
            _outputTopKAcc = args.OutputTopKAcc;

            var evalArgs = new MultiClassClassifierEvaluator.Arguments
            {
                OutputTopKAcc = _outputTopKAcc
            };
            _evaluator = new MultiClassClassifierEvaluator(Host, evalArgs);
        }

        protected override void PrintFoldResultsCore(IChannel ch, Dictionary<string, IDataView> metrics)
        {
            Host.AssertValue(metrics);

            if (!metrics.TryGetValue(MetricKinds.OverallMetrics, out IDataView fold))
                throw ch.Except("No overall metrics found");

            if (!metrics.TryGetValue(MetricKinds.ConfusionMatrix, out IDataView conf))
                throw ch.Except("No confusion matrix found");

            // Change the name of the Top-k-accuracy column.
            if (_outputTopKAcc != null)
                fold = ChangeTopKAccColumnName(fold);

            // Drop the per-class information.
            if (!_outputPerClass)
                fold = DropPerClassColumn(fold);

            var unweightedConf = MetricWriter.GetConfusionTable(Host, conf, out string weightedConf, false, _numConfusionTableClasses);
            var unweightedFold = MetricWriter.GetPerFoldResults(Host, fold, out string weightedFold);
            ch.Assert(string.IsNullOrEmpty(weightedConf) == string.IsNullOrEmpty(weightedFold));
            if (!string.IsNullOrEmpty(weightedConf))
            {
                ch.Info(weightedConf);
                ch.Info(weightedFold);
            }
            ch.Info(unweightedConf);
            ch.Info(unweightedFold);
        }

        protected override IDataView CombineOverallMetricsCore(IDataView[] metrics)
        {
            var overallList = new List<IDataView>();

            for (int i = 0; i < metrics.Length; i++)
            {
                var idv = metrics[i];
                if (!_outputPerClass)
                    idv = DropPerClassColumn(idv);

                overallList.Add(idv);
            }
            var views = overallList.ToArray();

            if (_outputPerClass)
            {
                EvaluateUtils.ReconcileSlotNames<double>(Host, views, MultiClassClassifierEvaluator.PerClassLogLoss, NumberType.R8,
                    def: double.NaN);
                for (int i = 0; i < overallList.Count; i++)
                {
                    var idv = views[i];

                    // Find the old per-class log-loss column and drop it.
                    for (int col = 0; col < idv.Schema.ColumnCount; col++)
                    {
                        if (idv.Schema.IsHidden(col) &&
                            idv.Schema.GetColumnName(col).Equals(MultiClassClassifierEvaluator.PerClassLogLoss))
                        {
                            idv = new ChooseColumnsByIndexTransform(Host,
                                new ChooseColumnsByIndexTransform.Arguments() { Drop = true, Index = new[] { col } }, idv);
                            break;
                        }
                    }
                    views[i] = idv;
                }
            }
            return base.CombineOverallMetricsCore(views);
        }

        protected override IDataView GetOverallResultsCore(IDataView overall)
        {
            // Change the name of the Top-k-accuracy column.
            if (_outputTopKAcc != null)
                overall = ChangeTopKAccColumnName(overall);
            return overall;
        }

        private IDataView ChangeTopKAccColumnName(IDataView input)
        {
            input = new ColumnsCopyingTransformer(Host, (MultiClassClassifierEvaluator.TopKAccuracy, string.Format(TopKAccuracyFormat, _outputTopKAcc))).Transform(input);
            return SelectColumnsTransform.CreateDrop(Host, input, MultiClassClassifierEvaluator.TopKAccuracy );
        }

        private IDataView DropPerClassColumn(IDataView input)
        {
            if (input.Schema.TryGetColumnIndex(MultiClassClassifierEvaluator.PerClassLogLoss, out int perClassCol))
            {
                input = SelectColumnsTransform.CreateDrop(Host, input, MultiClassClassifierEvaluator.PerClassLogLoss);
            }
            return input;
        }

        public override IEnumerable<MetricColumn> GetOverallMetricColumns()
        {
            yield return new MetricColumn("AccuracyMicro", MultiClassClassifierEvaluator.AccuracyMicro);
            yield return new MetricColumn("AccuracyMacro", MultiClassClassifierEvaluator.AccuracyMacro);
            yield return new MetricColumn("TopKAccuracy", string.Format(TopKAccuracyFormat, _outputTopKAcc));
            if (_outputPerClass)
            {
                yield return new MetricColumn("LogLoss<class name>",
                    MultiClassClassifierEvaluator.PerClassLogLoss, MetricColumn.Objective.Minimize, isVector: true,
                    namePattern: new Regex(string.Format(@"^{0}(?<class>.+)", MultiClassClassifierEvaluator.LogLoss), RegexOptions.IgnoreCase));
            }
            yield return new MetricColumn("LogLoss", MultiClassClassifierEvaluator.LogLoss, MetricColumn.Objective.Minimize);
            yield return new MetricColumn("LogLossReduction", MultiClassClassifierEvaluator.LogLossReduction);
        }

        protected override IEnumerable<string> GetPerInstanceColumnsToSave(RoleMappedSchema schema)
        {
            Host.CheckValue(schema, nameof(schema));
            Host.CheckParam(schema.Label != null, nameof(schema), "Schema must contain a label column");

            // Output the label column.
            yield return schema.Label.Name;

            // Return the output columns.
            yield return MultiClassPerInstanceEvaluator.Assigned;
            yield return MultiClassPerInstanceEvaluator.LogLoss;
            yield return MultiClassPerInstanceEvaluator.SortedScores;
            yield return MultiClassPerInstanceEvaluator.SortedClasses;
        }

        // Multi-class evaluator adds four per-instance columns: "Assigned", "Top scores", "Top classes" and "Log-loss".
        protected override IDataView GetPerInstanceMetricsCore(IDataView perInst, RoleMappedSchema schema)
        {
            // If the label column is a key without key values, convert it to I8, just for saving the per-instance
            // text file, since if there are different key counts the columns cannot be appended.
            if (!perInst.Schema.TryGetColumnIndex(schema.Label.Name, out int labelCol))
                throw Host.Except("Could not find column '{0}'", schema.Label.Name);
            var labelType = perInst.Schema.GetColumnType(labelCol);
            if (labelType.IsKey && (!perInst.Schema.HasKeyNames(labelCol, labelType.KeyCount) || labelType.RawKind != DataKind.U4))
            {
                perInst = LambdaColumnMapper.Create(Host, "ConvertToDouble", perInst, schema.Label.Name,
                    schema.Label.Name, perInst.Schema.GetColumnType(labelCol), NumberType.R8,
                    (in uint src, ref double dst) => dst = src == 0 ? double.NaN : src - 1 + (double)labelType.AsKey.Min);
            }

            var perInstSchema = perInst.Schema;
            if (perInstSchema.TryGetColumnIndex(MultiClassPerInstanceEvaluator.SortedClasses, out int sortedClassesIndex))
            {
                var type = perInstSchema.GetColumnType(sortedClassesIndex);
                if (_numTopClasses < type.VectorSize)
                {
                    // Wrap with a DropSlots transform to pick only the first _numTopClasses slots.
                    var args = new DropSlotsTransformer.Arguments
                    {
                        Column = new DropSlotsTransformer.Column[]
                        {
                            new DropSlotsTransformer.Column
                            {
                                Name = MultiClassPerInstanceEvaluator.SortedClasses,
                                Slots = new[] {
                                    new DropSlotsTransformer.Range()
                                    {
                                        Min = _numTopClasses
                                    }
                                }
                            }
                        }
                    };
                    perInst = new DropSlotsTransformer(Host, args, perInst);
                }
            }

            // Wrap with a DropSlots transform to pick only the first _numTopClasses slots.
            if (perInst.Schema.TryGetColumnIndex(MultiClassPerInstanceEvaluator.SortedScores, out int sortedScoresIndex))
            {
                var type = perInst.Schema.GetColumnType(sortedScoresIndex);
                if (_numTopClasses < type.VectorSize)
                {
                    var args = new DropSlotsTransformer.Arguments
                    {
                        Column = new DropSlotsTransformer.Column[]
                        {
                            new DropSlotsTransformer.Column()
                            {
                                Name = MultiClassPerInstanceEvaluator.SortedScores,
                                Slots = new[] {
                                    new DropSlotsTransformer.Range()
                                    {
                                        Min = _numTopClasses
                                    }
                                }
                            }
                        }
                    };
                    perInst = new DropSlotsTransformer(Host, args, perInst);
                }
            }
            return perInst;
        }
    }

    public static partial class Evaluate
    {
        [TlcModule.EntryPoint(Name = "Models.ClassificationEvaluator", Desc = "Evaluates a multi class classification scored dataset.")]
        public static CommonOutputs.ClassificationEvaluateOutput MultiClass(IHostEnvironment env, MultiClassMamlEvaluator.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("EvaluateMultiClass");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            MatchColumns(host, input, out string label, out string weight, out string name);
            var evaluator = new MultiClassMamlEvaluator(host, input);
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
    }
}
