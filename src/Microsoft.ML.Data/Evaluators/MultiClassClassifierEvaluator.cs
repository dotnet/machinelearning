// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.FeatureSelection;
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

        internal const string LoadName = "MultiClassClassifierEvaluator";

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

        private protected override void CheckScoreAndLabelTypes(RoleMappedSchema schema)
        {
            var score = schema.GetUniqueColumn(MetadataUtils.Const.ScoreValueKind.Score);
            var t = score.Type;
            if (t.VectorSize < 2 || t.ItemType != NumberType.Float)
                throw Host.ExceptSchemaMismatch(nameof(schema), "score", score.Name, "vector of two or more items of type R4", t.ToString());
            Host.CheckParam(schema.Label.HasValue, nameof(schema), "Could not find the label column");
            t = schema.Label.Value.Type;
            if (t != NumberType.Float && t.KeyCount <= 0)
                throw Host.ExceptSchemaMismatch(nameof(schema), "label", schema.Label.Value.Name, "float or a known-cardinality key", t.ToString());
        }

        private protected override Aggregator GetAggregatorCore(RoleMappedSchema schema, string stratName)
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
            var mdType = schema.Schema[scoreInfo.Index].Metadata.Schema.GetColumnOrNull(MetadataUtils.Kinds.SlotNames)?.Type;
            var labelNames = default(VBuffer<ReadOnlyMemory<char>>);
            if (mdType != null && mdType.IsKnownSizeVector && mdType.ItemType.IsText)
            {
                schema.Schema[scoreInfo.Index].Metadata.GetValue(MetadataUtils.Kinds.SlotNames, ref labelNames);
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

        private protected override IRowMapper CreatePerInstanceRowMapper(RoleMappedSchema schema)
        {
            Host.CheckParam(schema.Label.HasValue, nameof(schema), "Schema must contain a label column");
            var scoreInfo = schema.GetUniqueColumn(MetadataUtils.Const.ScoreValueKind.Score);
            int numClasses = scoreInfo.Type.VectorSize;
            return new MultiClassPerInstanceEvaluator(Host, schema.Schema, scoreInfo, schema.Label.Value.Name);
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

        private protected override void GetAggregatorConsolidationFuncs(Aggregator aggregator, AggregatorDictionaryBase[] dictionaries,
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

            internal override void InitializeNextPass(Row row, RoleMappedSchema schema)
            {
                Host.Assert(PassNum < 1);
                Host.Assert(schema.Label.HasValue);

                var score = schema.GetUniqueColumn(MetadataUtils.Const.ScoreValueKind.Score);
                Host.Assert(score.Type.VectorSize == _scoresArr.Length);
                _labelGetter = RowCursorUtils.GetLabelGetter(row, schema.Label.Value.Index);
                _scoreGetter = row.GetGetter<VBuffer<float>>(score.Index);
                Host.AssertValue(_labelGetter);
                Host.AssertValue(_scoreGetter);

                if (schema.Weight.HasValue)
                    _weightGetter = row.GetGetter<float>(schema.Weight.Value.Index);
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
                var editor = VBufferEditor.Create(ref slotNames, ClassNames.Length);
                for (int i = 0; i < ClassNames.Length; i++)
                    editor.Values[i] = string.Format("(class {0})", ClassNames[i]).AsMemory();
                slotNames = editor.Commit();
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
        public MultiClassClassifierMetrics Evaluate(IDataView data, string label, string score, string predictedLabel)
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

            MultiClassClassifierMetrics result;
            using (var cursor = overall.GetRowCursor(i => true))
            {
                var moved = cursor.MoveNext();
                Host.Assert(moved);
                result = new MultiClassClassifierMetrics(Host, cursor, _outputTopKAcc ?? 0);
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

        public MultiClassPerInstanceEvaluator(IHostEnvironment env, Schema schema, Schema.Column scoreColumn, string labelCol)
            : base(env, schema, scoreColumn.Name, labelCol)
        {
            CheckInputColumnTypes(schema);

            _numClasses = scoreColumn.Type.VectorSize;
            _types = new ColumnType[4];

            if (schema[ScoreIndex].HasSlotNames(_numClasses))
            {
                var classNames = default(VBuffer<ReadOnlyMemory<char>>);
                schema[(int) ScoreIndex].Metadata.GetValue(MetadataUtils.Kinds.SlotNames, ref classNames);
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

        private MultiClassPerInstanceEvaluator(IHostEnvironment env, ModelLoadContext ctx, Schema schema)
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

        public static MultiClassPerInstanceEvaluator Create(IHostEnvironment env, ModelLoadContext ctx, Schema schema)
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

        private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
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

        private protected override Delegate[] CreateGettersCore(Row input, Func<int, bool> activeCols, out Action disposer)
        {
            disposer = null;

            var getters = new Delegate[4];

            if (!activeCols(AssignedCol) && !activeCols(SortedClassesCol) && !activeCols(SortedScoresCol) && !activeCols(LogLossCol))
                return getters;

            long cachedPosition = -1;
            VBuffer<float> scores = default(VBuffer<float>);
            float label = 0;
            var scoresArr = new float[_numClasses];
            int[] sortedIndices = new int[_numClasses];

            var labelGetter = activeCols(LogLossCol) ? RowCursorUtils.GetLabelGetter(input, LabelIndex) :
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

            if (activeCols(AssignedCol))
            {
                ValueGetter<uint> assignedFn =
                    (ref uint dst) =>
                    {
                        updateCacheIfNeeded();
                        dst = (uint)sortedIndices[0] + 1;
                    };
                getters[AssignedCol] = assignedFn;
            }

            if (activeCols(SortedScoresCol))
            {
                ValueGetter<VBuffer<float>> topKScoresFn =
                    (ref VBuffer<float> dst) =>
                    {
                        updateCacheIfNeeded();
                        var editor = VBufferEditor.Create(ref dst, _numClasses);
                        for (int i = 0; i < _numClasses; i++)
                            editor.Values[i] = scores.GetItemOrDefault(sortedIndices[i]);
                        dst = editor.Commit();
                    };
                getters[SortedScoresCol] = topKScoresFn;
            }

            if (activeCols(SortedClassesCol))
            {
                ValueGetter<VBuffer<uint>> topKClassesFn =
                    (ref VBuffer<uint> dst) =>
                    {
                        updateCacheIfNeeded();
                        var editor = VBufferEditor.Create(ref dst, _numClasses);
                        for (int i = 0; i < _numClasses; i++)
                            editor.Values[i] = (uint)sortedIndices[i] + 1;
                        dst = editor.Commit();
                    };
                getters[SortedClassesCol] = topKClassesFn;
            }

            if (activeCols(LogLossCol))
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

        private protected override Schema.DetachedColumn[] GetOutputColumnsCore()
        {
            var infos = new Schema.DetachedColumn[4];

            var assignedColKeyValues = new MetadataBuilder();
            assignedColKeyValues.AddKeyValues(_numClasses, TextType.Instance, CreateKeyValueGetter());
            infos[AssignedCol] = new Schema.DetachedColumn(Assigned, _types[AssignedCol], assignedColKeyValues.GetMetadata());

            infos[LogLossCol] = new Schema.DetachedColumn(LogLoss, _types[LogLossCol], null);

            var sortedScores = new MetadataBuilder();
            sortedScores.AddSlotNames(_numClasses, CreateSlotNamesGetter(_numClasses, "Score"));

            var sortedClasses = new MetadataBuilder();
            sortedClasses.AddSlotNames(_numClasses, CreateSlotNamesGetter(_numClasses, "Class"));
            sortedClasses.AddKeyValues(_numClasses, TextType.Instance, CreateKeyValueGetter());

            infos[SortedScoresCol] = new Schema.DetachedColumn(SortedScores, _types[SortedScoresCol], sortedScores.GetMetadata());
            infos[SortedClassesCol] = new Schema.DetachedColumn(SortedClasses, _types[SortedClassesCol], sortedClasses.GetMetadata());
            return infos;
        }

        // REVIEW: Figure out how to avoid having the column name in each slot name.
        private ValueGetter<VBuffer<ReadOnlyMemory<char>>> CreateSlotNamesGetter(int numTopClasses, string suffix)
        {
            return
                (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                {
                    var editor = VBufferEditor.Create(ref dst, numTopClasses);
                    for (int i = 1; i <= numTopClasses; i++)
                        editor.Values[i - 1] = string.Format("#{0} {1}", i, suffix).AsMemory();
                    dst = editor.Commit();
                };
        }

        private ValueGetter<VBuffer<ReadOnlyMemory<char>>> CreateKeyValueGetter()
        {
            return
                (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                {
                    var editor = VBufferEditor.Create(ref dst, _numClasses);
                    for (int i = 0; i < _numClasses; i++)
                        editor.Values[i] = _classNames[i];
                    dst = editor.Commit();
                };
        }

        private void CheckInputColumnTypes(Schema schema)
        {
            Host.AssertNonEmpty(ScoreCol);
            Host.AssertNonEmpty(LabelCol);

            var t = schema[(int) ScoreIndex].Type;
            if (t.VectorSize < 2 || t.ItemType != NumberType.Float)
                throw Host.Except("Score column '{0}' has type '{1}' but must be a vector of two or more items of type R4", ScoreCol, t);
            t = schema[LabelIndex].Type;
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

        private protected override IEvaluator Evaluator => _evaluator;

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

        private protected override void PrintFoldResultsCore(IChannel ch, Dictionary<string, IDataView> metrics)
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

        private protected override IDataView CombineOverallMetricsCore(IDataView[] metrics)
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
                    for (int col = 0; col < idv.Schema.Count; col++)
                    {
                        if (idv.Schema[col].IsHidden &&
                            idv.Schema[col].Name.Equals(MultiClassClassifierEvaluator.PerClassLogLoss))
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

        private protected override IDataView GetOverallResultsCore(IDataView overall)
        {
            // Change the name of the Top-k-accuracy column.
            if (_outputTopKAcc != null)
                overall = ChangeTopKAccColumnName(overall);
            return overall;
        }

        private IDataView ChangeTopKAccColumnName(IDataView input)
        {
            input = new ColumnCopyingTransformer(Host, (MultiClassClassifierEvaluator.TopKAccuracy, string.Format(TopKAccuracyFormat, _outputTopKAcc))).Transform(input);
            return ColumnSelectingTransformer.CreateDrop(Host, input, MultiClassClassifierEvaluator.TopKAccuracy);
        }

        private IDataView DropPerClassColumn(IDataView input)
        {
            if (input.Schema.TryGetColumnIndex(MultiClassClassifierEvaluator.PerClassLogLoss, out int perClassCol))
            {
                input = ColumnSelectingTransformer.CreateDrop(Host, input, MultiClassClassifierEvaluator.PerClassLogLoss);
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

        private protected override IEnumerable<string> GetPerInstanceColumnsToSave(RoleMappedSchema schema)
        {
            Host.CheckValue(schema, nameof(schema));
            Host.CheckParam(schema.Label.HasValue, nameof(schema), "Schema must contain a label column");

            // Output the label column.
            yield return schema.Label.Value.Name;

            // Return the output columns.
            yield return MultiClassPerInstanceEvaluator.Assigned;
            yield return MultiClassPerInstanceEvaluator.LogLoss;
            yield return MultiClassPerInstanceEvaluator.SortedScores;
            yield return MultiClassPerInstanceEvaluator.SortedClasses;
        }

        // Multi-class evaluator adds four per-instance columns: "Assigned", "Top scores", "Top classes" and "Log-loss".
        private protected override IDataView GetPerInstanceMetricsCore(IDataView perInst, RoleMappedSchema schema)
        {
            // If the label column is a key without key values, convert it to I8, just for saving the per-instance
            // text file, since if there are different key counts the columns cannot be appended.
            string labelName = schema.Label.Value.Name;
            if (!perInst.Schema.TryGetColumnIndex(labelName, out int labelCol))
                throw Host.Except("Could not find column '{0}'", labelName);
            var labelType = perInst.Schema[labelCol].Type;
            if (labelType is KeyType keyType && (!(bool)perInst.Schema[labelCol].HasKeyValues(keyType.KeyCount) || labelType.RawKind != DataKind.U4))
            {
                perInst = LambdaColumnMapper.Create(Host, "ConvertToDouble", perInst, labelName,
                    labelName, perInst.Schema[labelCol].Type, NumberType.R8,
                    (in uint src, ref double dst) => dst = src == 0 ? double.NaN : src - 1 + (double)keyType.Min);
            }

            var perInstSchema = perInst.Schema;
            if (perInstSchema.TryGetColumnIndex(MultiClassPerInstanceEvaluator.SortedClasses, out int sortedClassesIndex))
            {
                var type = perInstSchema[sortedClassesIndex].Type;
                // Wrap with a DropSlots transform to pick only the first _numTopClasses slots.
                if (_numTopClasses < type.VectorSize)
                    perInst = new SlotsDroppingTransformer(Host, MultiClassPerInstanceEvaluator.SortedClasses, min: _numTopClasses).Transform(perInst);
            }

            // Wrap with a DropSlots transform to pick only the first _numTopClasses slots.
            if (perInst.Schema.TryGetColumnIndex(MultiClassPerInstanceEvaluator.SortedScores, out int sortedScoresIndex))
            {
                var type = perInst.Schema[sortedScoresIndex].Type;
                if (_numTopClasses < type.VectorSize)
                    perInst = new SlotsDroppingTransformer(Host, MultiClassPerInstanceEvaluator.SortedScores, min: _numTopClasses).Transform(perInst);
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
            IMamlEvaluator evaluator = new MultiClassMamlEvaluator(host, input);
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
