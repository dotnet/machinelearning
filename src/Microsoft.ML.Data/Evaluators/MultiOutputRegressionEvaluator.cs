// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Numeric;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;

[assembly: LoadableClass(typeof(MultiOutputRegressionEvaluator), typeof(MultiOutputRegressionEvaluator), typeof(MultiOutputRegressionEvaluator.Arguments), typeof(SignatureEvaluator),
    "Multi Output Regression Evaluator", MultiOutputRegressionEvaluator.LoadName, "MultiOutputRegression", "MRE")]

[assembly: LoadableClass(typeof(MultiOutputRegressionMamlEvaluator), typeof(MultiOutputRegressionMamlEvaluator), typeof(MultiOutputRegressionMamlEvaluator.Arguments), typeof(SignatureMamlEvaluator),
    "Multi Output Regression Evaluator", MultiOutputRegressionEvaluator.LoadName, "MultiOutputRegression", "MRE")]

// This is for deserialization from a binary model file.
[assembly: LoadableClass(typeof(MultiOutputRegressionPerInstanceEvaluator), null, typeof(SignatureLoadRowMapper),
    "", MultiOutputRegressionPerInstanceEvaluator.LoaderSignature)]

namespace Microsoft.ML.Data
{
    [BestFriend]
    internal sealed class MultiOutputRegressionEvaluator : RegressionLossEvaluatorBase<MultiOutputRegressionEvaluator.Aggregator>
    {
        public sealed class Arguments : ArgumentsBase
        {
        }

        private const string Dist = "Euclidean-Dist(avg)";
        private const string PerLabelL1 = "Per label L1(avg)";
        private const string PerLabelL2 = "Per label L2(avg)";
        private const string PerLabelRms = "Per label RMS(avg)";
        private const string PerLabelLoss = "Per label LOSS-FN(avg)";

        public const string LoadName = "MultiRegressionEvaluator";

        public MultiOutputRegressionEvaluator(IHostEnvironment env, Arguments args)
            : base(args, env, LoadName)
        {
        }

        private protected override IRowMapper CreatePerInstanceRowMapper(RoleMappedSchema schema)
        {
            Host.CheckParam(schema.Label.HasValue, nameof(schema), "Could not find the label column");
            var scoreCol = schema.GetUniqueColumn(AnnotationUtils.Const.ScoreValueKind.Score);

            return new MultiOutputRegressionPerInstanceEvaluator(Host, schema.Schema, scoreCol.Name, schema.Label.Value.Name);
        }

        private protected override void CheckScoreAndLabelTypes(RoleMappedSchema schema)
        {
            var score = schema.GetUniqueColumn(AnnotationUtils.Const.ScoreValueKind.Score);
            var t = score.Type as VectorType;
            if (t == null || !t.IsKnownSize || t.ItemType != NumberDataViewType.Single)
                throw Host.ExceptSchemaMismatch(nameof(schema), "score", score.Name, "known-size vector of float", t.ToString());
            Host.Check(schema.Label.HasValue, "Could not find the label column");
            t = schema.Label.Value.Type as VectorType;
            if (t == null || !t.IsKnownSize || (t.ItemType != NumberDataViewType.Single && t.ItemType != NumberDataViewType.Double))
                throw Host.ExceptSchemaMismatch(nameof(schema), "label", schema.Label.Value.Name, "known-size vector of float or double", t.ToString());
        }

        private protected override Aggregator GetAggregatorCore(RoleMappedSchema schema, string stratName)
        {
            var score = schema.GetUniqueColumn(AnnotationUtils.Const.ScoreValueKind.Score);
            int vectorSize = score.Type.GetVectorSize();
            Host.Assert(vectorSize > 0);
            return new Aggregator(Host, LossFunction, vectorSize, schema.Weight != null, stratName);
        }

        public override IEnumerable<MetricColumn> GetOverallMetricColumns()
        {
            yield return new MetricColumn("Dist", Dist, MetricColumn.Objective.Minimize);
            yield return new MetricColumn("L1_<label number>", PerLabelL1, MetricColumn.Objective.Minimize,
                isVector: true, namePattern: new Regex(string.Format(@"{0}_(?<label>\d+)\)", L1), RegexOptions.IgnoreCase),
                groupName: "label", nameFormat: string.Format("{0} (Label_{{0}}", PerLabelL1));
            yield return new MetricColumn("L2_<label number>", PerLabelL2, MetricColumn.Objective.Minimize,
                isVector: true, namePattern: new Regex(string.Format(@"{0}_(?<label>\d+)\)", L2), RegexOptions.IgnoreCase),
                groupName: "label", nameFormat: string.Format("{0} (Label_{{0}}", PerLabelL2));
            yield return new MetricColumn("Rms_<label number>", PerLabelRms, MetricColumn.Objective.Minimize,
                isVector: true, namePattern: new Regex(string.Format(@"{0}_(?<label>\d+)\)", Rms), RegexOptions.IgnoreCase),
                groupName: "label", nameFormat: string.Format("{0} (Label_{{0}}", PerLabelRms));
            yield return new MetricColumn("Loss_<label number>", PerLabelLoss, MetricColumn.Objective.Minimize,
                isVector: true, namePattern: new Regex(string.Format(@"{0}_(?<label>\d+)\)", Loss), RegexOptions.IgnoreCase),
                groupName: "label", nameFormat: string.Format("{0} (Label_{{0}}", PerLabelLoss));
        }

        private protected override void GetAggregatorConsolidationFuncs(Aggregator aggregator, AggregatorDictionaryBase[] dictionaries,
            out Action<uint, ReadOnlyMemory<char>, Aggregator> addAgg, out Func<Dictionary<string, IDataView>> consolidate)
        {
            var stratCol = new List<uint>();
            var stratVal = new List<ReadOnlyMemory<char>>();
            var isWeighted = new List<bool>();
            var l1 = new List<Double>();
            var l2 = new List<Double>();
            var dist = new List<Double>();
            var perLabelL1 = new List<Double[]>();
            var perLabelL2 = new List<Double[]>();
            var perLabelRms = new List<Double[]>();
            var perLabelLoss = new List<Double[]>();

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
                    dist.Add(agg.UnweightedCounters.Dist);
                    perLabelL1.Add(agg.UnweightedCounters.PerLabelL1);
                    perLabelL2.Add(agg.UnweightedCounters.PerLabelL2);
                    perLabelRms.Add(agg.UnweightedCounters.PerLabelRms);
                    perLabelLoss.Add(agg.UnweightedCounters.PerLabelLoss);
                    if (agg.Weighted)
                    {
                        stratCol.Add(stratColKey);
                        stratVal.Add(stratColVal);
                        isWeighted.Add(true);
                        l1.Add(agg.WeightedCounters.L1);
                        l2.Add(agg.WeightedCounters.L2);
                        dist.Add(agg.WeightedCounters.Dist);
                        perLabelL1.Add(agg.WeightedCounters.PerLabelL1);
                        perLabelL2.Add(agg.WeightedCounters.PerLabelL2);
                        perLabelRms.Add(agg.WeightedCounters.PerLabelRms);
                        perLabelLoss.Add(agg.WeightedCounters.PerLabelLoss);
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
                    if (hasWeight)
                        overallDvBldr.AddColumn(MetricKinds.ColumnNames.IsWeighted, BooleanDataViewType.Instance, isWeighted.ToArray());
                    overallDvBldr.AddColumn(PerLabelL1, aggregator.GetSlotNames, NumberDataViewType.Double, perLabelL1.ToArray());
                    overallDvBldr.AddColumn(PerLabelL2, aggregator.GetSlotNames, NumberDataViewType.Double, perLabelL2.ToArray());
                    overallDvBldr.AddColumn(PerLabelRms, aggregator.GetSlotNames, NumberDataViewType.Double, perLabelRms.ToArray());
                    overallDvBldr.AddColumn(PerLabelLoss, aggregator.GetSlotNames, NumberDataViewType.Double, perLabelLoss.ToArray());
                    overallDvBldr.AddColumn(L1, NumberDataViewType.Double, l1.ToArray());
                    overallDvBldr.AddColumn(L2, NumberDataViewType.Double, l2.ToArray());
                    overallDvBldr.AddColumn(Dist, NumberDataViewType.Double, dist.ToArray());
                    var result = new Dictionary<string, IDataView>();
                    result.Add(MetricKinds.OverallMetrics, overallDvBldr.GetDataView());
                    return result;
                };
        }

        public sealed class Aggregator : AggregatorBase
        {
            public sealed class Counters
            {
                private readonly Double[] _l1Loss;
                private readonly Double[] _l2Loss;
                private readonly Double[] _fnLoss;
                private Double _sumWeights;
                private Double _sumL1;
                private Double _sumL2;
                private Double _sumEuclidean;

                private readonly IRegressionLoss _lossFunction;

                public Double L1 => _sumWeights > 0 ? _sumL1 / _sumWeights : 0;

                public Double L2 => _sumWeights > 0 ? _sumL2 / _sumWeights : 0;

                public Double Dist => _sumWeights > 0 ? _sumEuclidean / _sumWeights : 0;

                public Double[] PerLabelL1
                {
                    get
                    {
                        var res = new double[_l1Loss.Length];
                        if (_sumWeights == 0)
                            return res;
                        for (int i = 0; i < _l1Loss.Length; i++)
                            res[i] = _l1Loss[i] / _sumWeights;
                        return res;
                    }
                }

                public Double[] PerLabelL2
                {
                    get
                    {
                        var res = new double[_l2Loss.Length];
                        if (_sumWeights == 0)
                            return res;
                        for (int i = 0; i < _l2Loss.Length; i++)
                            res[i] = _l2Loss[i] / _sumWeights;
                        return res;
                    }
                }

                public Double[] PerLabelRms
                {
                    get
                    {
                        var res = new double[_l2Loss.Length];
                        if (_sumWeights == 0)
                            return res;
                        for (int i = 0; i < _l2Loss.Length; i++)
                            res[i] = Math.Sqrt(_l2Loss[i] / _sumWeights);
                        return res;
                    }
                }

                public Double[] PerLabelLoss
                {
                    get
                    {
                        var res = new double[_fnLoss.Length];
                        if (_sumWeights == 0)
                            return res;
                        for (int i = 0; i < _fnLoss.Length; i++)
                            res[i] = _fnLoss[i] / _sumWeights;
                        return res;
                    }
                }

                public Counters(IRegressionLoss lossFunction, int size)
                {
                    Contracts.AssertValue(lossFunction);
                    Contracts.Assert(size > 0);
                    _lossFunction = lossFunction;
                    _l1Loss = new double[size];
                    _l2Loss = new double[size];
                    _fnLoss = new double[size];
                }

                public void Update(ReadOnlySpan<float> score, ReadOnlySpan<float> label, int length, float weight)
                {
                    Contracts.Assert(length == _l1Loss.Length);
                    Contracts.Assert(score.Length >= length);
                    Contracts.Assert(label.Length >= length);

                    Double wht = weight;
                    Double l1 = 0;
                    Double l2 = 0;
                    for (int i = 0; i < length; i++)
                    {
                        Double currL1Loss = Math.Abs((Double)label[i] - score[i]);
                        _l1Loss[i] += currL1Loss * wht;
                        _l2Loss[i] += currL1Loss * currL1Loss * wht;
                        _fnLoss[i] += _lossFunction.Loss(score[i], label[i]) * wht;
                        l1 += currL1Loss;
                        l2 += currL1Loss * currL1Loss;
                    }
                    _sumL1 += l1 * weight;
                    _sumL2 += l2 * weight;
                    _sumEuclidean += Math.Sqrt(l2) * weight;
                    _sumWeights += weight;
                }
            }

            private ValueGetter<VBuffer<float>> _labelGetter;
            private ValueGetter<VBuffer<float>> _scoreGetter;
            private ValueGetter<float> _weightGetter;

            private readonly int _size;

            private VBuffer<float> _label;
            private VBuffer<float> _score;
            private readonly float[] _labelArr;
            private readonly float[] _scoreArr;

            public readonly Counters UnweightedCounters;
            public readonly Counters WeightedCounters;
            public readonly bool Weighted;

            public Aggregator(IHostEnvironment env, IRegressionLoss lossFunction, int size, bool weighted, string stratName)
                : base(env, stratName)
            {
                Host.AssertValue(lossFunction);
                Host.Assert(size > 0);

                _size = size;
                _labelArr = new float[_size];
                _scoreArr = new float[_size];
                UnweightedCounters = new Counters(lossFunction, _size);
                Weighted = weighted;
                WeightedCounters = Weighted ? new Counters(lossFunction, _size) : null;
            }

            internal override void InitializeNextPass(DataViewRow row, RoleMappedSchema schema)
            {
                Contracts.Assert(PassNum < 1);
                Contracts.Assert(schema.Label.HasValue);

                var score = schema.GetUniqueColumn(AnnotationUtils.Const.ScoreValueKind.Score);

                _labelGetter = RowCursorUtils.GetVecGetterAs<float>(NumberDataViewType.Single, row, schema.Label.Value.Index);
                _scoreGetter = row.GetGetter<VBuffer<float>>(score);
                Contracts.AssertValue(_labelGetter);
                Contracts.AssertValue(_scoreGetter);

                if (schema.Weight.HasValue)
                    _weightGetter = row.GetGetter<float>(schema.Weight.Value);
            }

            public override void ProcessRow()
            {
                _labelGetter(ref _label);
                Contracts.Check(_label.Length == _size);
                _scoreGetter(ref _score);
                Contracts.Check(_score.Length == _size);

                if (VBufferUtils.HasNaNs(in _score))
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

                ReadOnlySpan<float> label;
                if (!_label.IsDense)
                {
                    _label.CopyTo(_labelArr);
                    label = _labelArr;
                }
                else
                    label = _label.GetValues();
                ReadOnlySpan<float> score;
                if (!_score.IsDense)
                {
                    _score.CopyTo(_scoreArr);
                    score = _scoreArr;
                }
                else
                    score = _score.GetValues();
                UnweightedCounters.Update(score, label, _size, 1);
                if (WeightedCounters != null)
                    WeightedCounters.Update(score, label, _size, weight);
            }

            public void GetSlotNames(ref VBuffer<ReadOnlyMemory<char>> slotNames)
            {
                var editor = VBufferEditor.Create(ref slotNames, _size);
                for (int i = 0; i < _size; i++)
                    editor.Values[i] = string.Format("(Label_{0})", i).AsMemory();
                slotNames = editor.Commit();
            }
        }
    }

    internal sealed class MultiOutputRegressionPerInstanceEvaluator : PerInstanceEvaluatorBase
    {
        public const string LoaderSignature = "MultiRegPerInstance";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "MREGINST",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(MultiOutputRegressionPerInstanceEvaluator).Assembly.FullName);
        }

        private const int LabelOutput = 0;
        private const int ScoreOutput = 1;
        private const int L1Output = 2;
        private const int L2Output = 3;
        private const int DistCol = 4;

        public const string L1 = "L1-loss";
        public const string L2 = "L2-loss";
        public const string Dist = "Euclidean-Distance";

        private readonly VectorType _labelType;
        private readonly VectorType _scoreType;
        private readonly DataViewSchema.Annotations _labelMetadata;
        private readonly DataViewSchema.Annotations _scoreMetadata;

        public MultiOutputRegressionPerInstanceEvaluator(IHostEnvironment env, DataViewSchema schema, string scoreCol,
            string labelCol)
            : base(env, schema, scoreCol, labelCol)
        {
            CheckInputColumnTypes(schema, out _labelType, out _scoreType, out _labelMetadata, out _scoreMetadata);
        }

        private MultiOutputRegressionPerInstanceEvaluator(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema schema)
            : base(env, ctx, schema)
        {
            CheckInputColumnTypes(schema, out _labelType, out _scoreType, out _labelMetadata, out _scoreMetadata);

            // *** Binary format **
            // base
        }

        public static MultiOutputRegressionPerInstanceEvaluator Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema schema)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new MultiOutputRegressionPerInstanceEvaluator(env, ctx, schema);
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format **
            // base
            base.SaveModel(ctx);
        }

        private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
        {
            return
                col =>
                    (activeOutput(LabelOutput) && col == LabelIndex) ||
                    (activeOutput(ScoreOutput) && col == ScoreIndex) ||
                    (activeOutput(L1Output) || activeOutput(L2Output) || activeOutput(DistCol)) &&
                    (col == ScoreIndex || col == LabelIndex);
        }

        private protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
        {
            var infos = new DataViewSchema.DetachedColumn[5];
            infos[LabelOutput] = new DataViewSchema.DetachedColumn(LabelCol, _labelType, _labelMetadata);
            infos[ScoreOutput] = new DataViewSchema.DetachedColumn(ScoreCol, _scoreType, _scoreMetadata);
            infos[L1Output] = new DataViewSchema.DetachedColumn(L1, NumberDataViewType.Double, null);
            infos[L2Output] = new DataViewSchema.DetachedColumn(L2, NumberDataViewType.Double, null);
            infos[DistCol] = new DataViewSchema.DetachedColumn(Dist, NumberDataViewType.Double, null);
            return infos;
        }

        private protected override Delegate[] CreateGettersCore(DataViewRow input, Func<int, bool> activeCols, out Action disposer)
        {
            Host.Assert(LabelIndex >= 0);
            Host.Assert(ScoreIndex >= 0);

            disposer = null;

            long cachedPosition = -1;
            var label = default(VBuffer<float>);
            var score = default(VBuffer<float>);

            ValueGetter<VBuffer<float>> nullGetter = (ref VBuffer<float> vec) => vec = default(VBuffer<float>);
            var labelGetter = activeCols(LabelOutput) || activeCols(L1Output) || activeCols(L2Output) || activeCols(DistCol)
                ? RowCursorUtils.GetVecGetterAs<float>(NumberDataViewType.Single, input, LabelIndex)
                : nullGetter;
            var scoreGetter = activeCols(ScoreOutput) || activeCols(L1Output) || activeCols(L2Output) || activeCols(DistCol)
                ? input.GetGetter<VBuffer<float>>(input.Schema[ScoreIndex])
                : nullGetter;
            Action updateCacheIfNeeded =
                () =>
                {
                    if (cachedPosition != input.Position)
                    {
                        labelGetter(ref label);
                        scoreGetter(ref score);
                        cachedPosition = input.Position;
                    }
                };

            var getters = new Delegate[5];
            if (activeCols(LabelOutput))
            {
                ValueGetter<VBuffer<float>> labelFn =
                    (ref VBuffer<float> dst) =>
                    {
                        updateCacheIfNeeded();
                        label.CopyTo(ref dst);
                    };
                getters[LabelOutput] = labelFn;
            }
            if (activeCols(ScoreOutput))
            {
                ValueGetter<VBuffer<float>> scoreFn =
                    (ref VBuffer<float> dst) =>
                    {
                        updateCacheIfNeeded();
                        score.CopyTo(ref dst);
                    };
                getters[ScoreOutput] = scoreFn;
            }
            if (activeCols(L1Output))
            {
                ValueGetter<double> l1Fn =
                    (ref double dst) =>
                    {
                        updateCacheIfNeeded();
                        dst = VectorUtils.L1Distance(in label, in score);
                    };
                getters[L1Output] = l1Fn;
            }
            if (activeCols(L2Output))
            {
                ValueGetter<double> l2Fn =
                    (ref double dst) =>
                    {
                        updateCacheIfNeeded();
                        dst = VectorUtils.L2DistSquared(in label, in score);
                    };
                getters[L2Output] = l2Fn;
            }
            if (activeCols(DistCol))
            {
                ValueGetter<double> distFn =
                    (ref double dst) =>
                    {
                        updateCacheIfNeeded();
                        dst = MathUtils.Sqrt(VectorUtils.L2DistSquared(in label, in score));
                    };
                getters[DistCol] = distFn;
            }
            return getters;
        }

        private void CheckInputColumnTypes(DataViewSchema schema, out VectorType labelType, out VectorType scoreType,
            out DataViewSchema.Annotations labelMetadata, out DataViewSchema.Annotations scoreMetadata)
        {
            Host.AssertNonEmpty(ScoreCol);
            Host.AssertNonEmpty(LabelCol);

            var t = schema[LabelIndex].Type as VectorType;
            if (t == null || !t.IsKnownSize || (t.ItemType != NumberDataViewType.Single && t.ItemType != NumberDataViewType.Double))
                throw Host.ExceptSchemaMismatch(nameof(schema), "label", LabelCol, "known-size vector of float or double", t.ToString());
            labelType = new VectorType((PrimitiveDataViewType)t.ItemType, t.Size);
            var slotNamesType = new VectorType(TextDataViewType.Instance, t.Size);
            var builder = new DataViewSchema.Annotations.Builder();
            builder.AddSlotNames(t.Size, CreateSlotNamesGetter(schema, LabelIndex, labelType.Size, "True"));
            labelMetadata = builder.ToAnnotations();

            t = schema[ScoreIndex].Type as VectorType;
            if (t == null || !t.IsKnownSize || t.ItemType != NumberDataViewType.Single)
                throw Host.ExceptSchemaMismatch(nameof(schema), "score", ScoreCol, "known-size vector of float", t.ToString());
            scoreType = new VectorType((PrimitiveDataViewType)t.ItemType, t.Size);
            builder = new DataViewSchema.Annotations.Builder();
            builder.AddSlotNames(t.Size, CreateSlotNamesGetter(schema, ScoreIndex, scoreType.Size, "Predicted"));

            ValueGetter<ReadOnlyMemory<char>> getter = GetScoreColumnKind;
            builder.Add(AnnotationUtils.Kinds.ScoreColumnKind, TextDataViewType.Instance, getter);
            getter = GetScoreValueKind;
            builder.Add(AnnotationUtils.Kinds.ScoreValueKind, TextDataViewType.Instance, getter);
            ValueGetter<uint> uintGetter = GetScoreColumnSetId(schema);
            builder.Add(AnnotationUtils.Kinds.ScoreColumnSetId, AnnotationUtils.ScoreColumnSetIdType, uintGetter);
            scoreMetadata = builder.ToAnnotations();
        }

        private ValueGetter<uint> GetScoreColumnSetId(DataViewSchema schema)
        {
            int c;
            var max = schema.GetMaxAnnotationKind(out c, AnnotationUtils.Kinds.ScoreColumnSetId);
            uint id = checked(max + 1);
            return
                (ref uint dst) => dst = id;
        }

        private void GetScoreColumnKind(ref ReadOnlyMemory<char> dst)
        {
            dst = AnnotationUtils.Const.ScoreColumnKind.MultiOutputRegression.AsMemory();
        }

        private void GetScoreValueKind(ref ReadOnlyMemory<char> dst)
        {
            dst = AnnotationUtils.Const.ScoreValueKind.Score.AsMemory();
        }

        private ValueGetter<VBuffer<ReadOnlyMemory<char>>> CreateSlotNamesGetter(DataViewSchema schema, int column, int length, string prefix)
        {
            var type = schema[column].Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.SlotNames)?.Type;
            if (type != null && type is TextDataViewType)
            {
                return
                    (ref VBuffer<ReadOnlyMemory<char>> dst) => schema[column].Annotations.GetValue(AnnotationUtils.Kinds.SlotNames, ref dst);
            }
            return
                (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                {
                    var editor = VBufferEditor.Create(ref dst, length);
                    for (int i = 0; i < length; i++)
                        editor.Values[i] = string.Format("{0}_{1}", prefix, i).AsMemory();
                    dst = editor.Commit();
                };
        }
    }

    [BestFriend]
    internal sealed class MultiOutputRegressionMamlEvaluator : MamlEvaluatorBase
    {
        public sealed class Arguments : ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "Loss function", ShortName = "loss")]
            public ISupportRegressionLossFactory LossFunction = new SquaredLossFactory();

            [Argument(ArgumentType.AtMostOnce, HelpText = "Supress labels and scores in per-instance outputs?", ShortName = "noScores")]
            public bool SupressScoresAndLabels = false;
        }

        private readonly MultiOutputRegressionEvaluator _evaluator;
        private readonly bool _supressScoresAndLabels;

        private protected override IEvaluator Evaluator => _evaluator;

        public MultiOutputRegressionMamlEvaluator(IHostEnvironment env, Arguments args)
            : base(args, env, AnnotationUtils.Const.ScoreColumnKind.MultiOutputRegression, "RegressionMamlEvaluator")
        {
            Host.CheckUserArg(args.LossFunction != null, nameof(args.LossFunction), "Loss function must be specified");

            _supressScoresAndLabels = args.SupressScoresAndLabels;
            var evalArgs = new MultiOutputRegressionEvaluator.Arguments();
            evalArgs.LossFunction = args.LossFunction;
            _evaluator = new MultiOutputRegressionEvaluator(Host, evalArgs);
        }

        private protected override IEnumerable<string> GetPerInstanceColumnsToSave(RoleMappedSchema schema)
        {
            Host.CheckValue(schema, nameof(schema));
            Host.CheckParam(schema.Label != null, nameof(schema), "Schema must contain a label column");

            // The multi output regression evaluator outputs the label and score column if requested by the user.
            if (!_supressScoresAndLabels)
            {
                yield return schema.Label.Value.Name;

                var scoreCol = EvaluateUtils.GetScoreColumn(Host, schema.Schema, ScoreCol, nameof(Arguments.ScoreColumn),
                    AnnotationUtils.Const.ScoreColumnKind.MultiOutputRegression);
                yield return scoreCol.Name;
            }

            // Return the output columns.
            yield return MultiOutputRegressionPerInstanceEvaluator.L1;
            yield return MultiOutputRegressionPerInstanceEvaluator.L2;
            yield return MultiOutputRegressionPerInstanceEvaluator.Dist;
        }

        // The multi-output regression evaluator prints only the per-label metrics for each fold.
        private protected override void PrintFoldResultsCore(IChannel ch, Dictionary<string, IDataView> metrics)
        {
            IDataView fold;
            if (!metrics.TryGetValue(MetricKinds.OverallMetrics, out fold))
                throw ch.Except("No overall metrics found");

            var isWeightedCol = fold.Schema.GetColumnOrNull(MetricKinds.ColumnNames.IsWeighted);

            int stratCol;
            bool hasStrats = fold.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.StratCol, out stratCol);
            int stratVal;
            bool hasStratVals = fold.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.StratVal, out stratVal);
            ch.Assert(hasStrats == hasStratVals);

            var colCount = fold.Schema.Count;
            var vBufferGetters = new ValueGetter<VBuffer<double>>[colCount];

            using (var cursor = fold.GetRowCursorForAllColumns())
            {
                bool isWeighted = false;
                ValueGetter<bool> isWeightedGetter;
                if (isWeightedCol.HasValue)
                    isWeightedGetter = cursor.GetGetter<bool>(isWeightedCol.Value);
                else
                    isWeightedGetter = (ref bool dst) => dst = false;

                ValueGetter<uint> stratGetter;
                if (hasStrats)
                {
                    var type = cursor.Schema[stratCol].Type;
                    stratGetter = RowCursorUtils.GetGetterAs<uint>(type, cursor, stratCol);
                }
                else
                    stratGetter = (ref uint dst) => dst = 0;

                int labelCount = 0;
                for (int i = 0; i < fold.Schema.Count; i++)
                {
                    var currentColumn = fold.Schema[i];
                    if (currentColumn.IsHidden || (isWeightedCol.HasValue && i == isWeightedCol.Value.Index) ||
                        (hasStrats && (i == stratCol || i == stratVal)))
                    {
                        continue;
                    }

                    var type = fold.Schema[i].Type as VectorType;
                    if (type != null && type.IsKnownSize && type.ItemType == NumberDataViewType.Double)
                    {
                        vBufferGetters[i] = cursor.GetGetter<VBuffer<double>>(currentColumn);
                        if (labelCount == 0)
                            labelCount = type.Size;
                        else
                            ch.Check(labelCount == type.Size, "All vector metrics should contain the same number of slots");
                    }
                }
                var labelNames = new ReadOnlyMemory<char>[labelCount];
                for (int j = 0; j < labelCount; j++)
                    labelNames[j] = string.Format("Label_{0}", j).AsMemory();

                var sb = new StringBuilder();
                sb.AppendLine("Per-label metrics:");
                sb.AppendFormat("{0,12} ", " ");
                for (int i = 0; i < labelCount; i++)
                    sb.AppendFormat(" {0,20}", labelNames[i]);
                sb.AppendLine();

                VBuffer<Double> metricVals = default(VBuffer<Double>);
                bool foundWeighted = !isWeightedCol.HasValue;
                bool foundUnweighted = false;
                uint strat = 0;
                while (cursor.MoveNext())
                {
                    isWeightedGetter(ref isWeighted);
                    if (foundWeighted && isWeighted || foundUnweighted && !isWeighted)
                    {
                        throw ch.Except("Multiple {0} rows found in overall metrics data view",
                            isWeighted ? "weighted" : "unweighted");
                    }
                    if (isWeighted)
                        foundWeighted = true;
                    else
                        foundUnweighted = true;

                    stratGetter(ref strat);
                    if (strat > 0)
                        continue;

                    for (int i = 0; i < colCount; i++)
                    {
                        if (vBufferGetters[i] != null)
                        {
                            vBufferGetters[i](ref metricVals);
                            ch.Assert(metricVals.Length == labelCount);

                            sb.AppendFormat("{0}{1,12}:", isWeighted ? "Weighted " : "", fold.Schema[i].Name);
                            foreach (var metric in metricVals.Items(all: true))
                                sb.AppendFormat(" {0,20:G20}", metric.Value);
                            sb.AppendLine();
                        }
                    }
                    if (foundUnweighted && foundWeighted)
                        break;
                }
                ch.Assert(foundUnweighted && foundWeighted);
                ch.Info(sb.ToString());
            }
        }
    }

    internal static partial class Evaluate
    {
        [TlcModule.EntryPoint(Name = "Models.MultiOutputRegressionEvaluator", Desc = "Evaluates a multi output regression scored dataset.")]
        public static CommonOutputs.CommonEvaluateOutput MultiOutputRegression(IHostEnvironment env, MultiOutputRegressionMamlEvaluator.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("EvaluateMultiOutput");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            string label;
            string weight;
            string name;
            MatchColumns(host, input, out label, out weight, out name);
            IMamlEvaluator evaluator = new MultiOutputRegressionMamlEvaluator(host, input);
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
