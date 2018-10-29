// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Numeric;

[assembly: LoadableClass(typeof(MultiOutputRegressionEvaluator), typeof(MultiOutputRegressionEvaluator), typeof(MultiOutputRegressionEvaluator.Arguments), typeof(SignatureEvaluator),
    "Multi Output Regression Evaluator", MultiOutputRegressionEvaluator.LoadName, "MultiOutputRegression", "MRE")]

[assembly: LoadableClass(typeof(MultiOutputRegressionMamlEvaluator), typeof(MultiOutputRegressionMamlEvaluator), typeof(MultiOutputRegressionMamlEvaluator.Arguments), typeof(SignatureMamlEvaluator),
    "Multi Output Regression Evaluator", MultiOutputRegressionEvaluator.LoadName, "MultiOutputRegression", "MRE")]

// This is for deserialization from a binary model file.
[assembly: LoadableClass(typeof(MultiOutputRegressionPerInstanceEvaluator), null, typeof(SignatureLoadRowMapper),
    "", MultiOutputRegressionPerInstanceEvaluator.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    public sealed class MultiOutputRegressionEvaluator : RegressionLossEvaluatorBase<MultiOutputRegressionEvaluator.Aggregator>
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

        protected override IRowMapper CreatePerInstanceRowMapper(RoleMappedSchema schema)
        {
            Host.CheckParam(schema.Label != null, nameof(schema), "Could not find the label column");
            var scoreInfo = schema.GetUniqueColumn(MetadataUtils.Const.ScoreValueKind.Score);
            Host.AssertValue(scoreInfo);

            return new MultiOutputRegressionPerInstanceEvaluator(Host, schema.Schema, scoreInfo.Name, schema.Label.Name);
        }

        protected override void CheckScoreAndLabelTypes(RoleMappedSchema schema)
        {
            var score = schema.GetUniqueColumn(MetadataUtils.Const.ScoreValueKind.Score);
            var t = score.Type;
            if (t.VectorSize == 0 || t.ItemType != NumberType.Float)
                throw Host.Except("Score column '{0}' has type '{1}' but must be a known length vector of type R4", score.Name, t);
            Host.Check(schema.Label != null, "Could not find the label column");
            t = schema.Label.Type;
            if (!t.IsKnownSizeVector || (t.ItemType != NumberType.R4 && t.ItemType != NumberType.R8))
                throw Host.Except("Label column '{0}' has type '{1}' but must be a known-size vector of R4 or R8", schema.Label.Name, t);
        }

        protected override Aggregator GetAggregatorCore(RoleMappedSchema schema, string stratName)
        {
            var score = schema.GetUniqueColumn(MetadataUtils.Const.ScoreValueKind.Score);
            Host.Assert(score.Type.VectorSize > 0);
            return new Aggregator(Host, LossFunction, score.Type.VectorSize, schema.Weight != null, stratName);
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

        protected override void GetAggregatorConsolidationFuncs(Aggregator aggregator, AggregatorDictionaryBase[] dictionaries,
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
                        overallDvBldr.AddColumn(MetricKinds.ColumnNames.StratCol, GetKeyValueGetter(dictionaries), 0, dictionaries.Length, stratCol.ToArray());
                        overallDvBldr.AddColumn(MetricKinds.ColumnNames.StratVal, TextType.Instance, stratVal.ToArray());
                    }
                    if (hasWeight)
                        overallDvBldr.AddColumn(MetricKinds.ColumnNames.IsWeighted, BoolType.Instance, isWeighted.ToArray());
                    overallDvBldr.AddColumn(PerLabelL1, aggregator.GetSlotNames, NumberType.R8, perLabelL1.ToArray());
                    overallDvBldr.AddColumn(PerLabelL2, aggregator.GetSlotNames, NumberType.R8, perLabelL2.ToArray());
                    overallDvBldr.AddColumn(PerLabelRms, aggregator.GetSlotNames, NumberType.R8, perLabelRms.ToArray());
                    overallDvBldr.AddColumn(PerLabelLoss, aggregator.GetSlotNames, NumberType.R8, perLabelLoss.ToArray());
                    overallDvBldr.AddColumn(L1, NumberType.R8, l1.ToArray());
                    overallDvBldr.AddColumn(L2, NumberType.R8, l2.ToArray());
                    overallDvBldr.AddColumn(Dist, NumberType.R8, dist.ToArray());
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

                public Double L1 { get { return _sumWeights > 0 ? _sumL1 / _sumWeights : 0; } }

                public Double L2 { get { return _sumWeights > 0 ? _sumL2 / _sumWeights : 0; } }

                public Double Dist { get { return _sumWeights > 0 ? _sumEuclidean / _sumWeights : 0; } }

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

                public void Update(Float[] score, Float[] label, int length, Float weight)
                {
                    Contracts.Assert(length == _l1Loss.Length);
                    Contracts.Assert(Utils.Size(score) >= length);
                    Contracts.Assert(Utils.Size(label) >= length);

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

            private ValueGetter<VBuffer<Float>> _labelGetter;
            private ValueGetter<VBuffer<Float>> _scoreGetter;
            private ValueGetter<Float> _weightGetter;

            private readonly int _size;

            private VBuffer<Float> _label;
            private VBuffer<Float> _score;
            private readonly Float[] _labelArr;
            private readonly Float[] _scoreArr;

            public readonly Counters UnweightedCounters;
            public readonly Counters WeightedCounters;
            public readonly bool Weighted;

            public Aggregator(IHostEnvironment env, IRegressionLoss lossFunction, int size, bool weighted, string stratName)
                : base(env, stratName)
            {
                Host.AssertValue(lossFunction);
                Host.Assert(size > 0);

                _size = size;
                _labelArr = new Float[_size];
                _scoreArr = new Float[_size];
                UnweightedCounters = new Counters(lossFunction, _size);
                Weighted = weighted;
                WeightedCounters = Weighted ? new Counters(lossFunction, _size) : null;
            }

            public override void InitializeNextPass(IRow row, RoleMappedSchema schema)
            {
                Contracts.Assert(PassNum < 1);
                Contracts.AssertValue(schema.Label);

                var score = schema.GetUniqueColumn(MetadataUtils.Const.ScoreValueKind.Score);

                _labelGetter = RowCursorUtils.GetVecGetterAs<Float>(NumberType.Float, row, schema.Label.Index);
                _scoreGetter = row.GetGetter<VBuffer<Float>>(score.Index);
                Contracts.AssertValue(_labelGetter);
                Contracts.AssertValue(_scoreGetter);

                if (schema.Weight != null)
                    _weightGetter = row.GetGetter<Float>(schema.Weight.Index);
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

                Float weight = 1;
                if (_weightGetter != null)
                {
                    _weightGetter(ref weight);
                    if (!FloatUtils.IsFinite(weight))
                    {
                        NumBadWeights++;
                        weight = 1;
                    }
                }

                Float[] label;
                if (!_label.IsDense)
                {
                    _label.CopyTo(_labelArr);
                    label = _labelArr;
                }
                else
                    label = _label.Values;
                Float[] score;
                if (!_score.IsDense)
                {
                    _score.CopyTo(_scoreArr);
                    score = _scoreArr;
                }
                else
                    score = _score.Values;
                UnweightedCounters.Update(score, label, _size, 1);
                if (WeightedCounters != null)
                    WeightedCounters.Update(score, label, _size, weight);
            }

            public void GetSlotNames(ref VBuffer<ReadOnlyMemory<char>> slotNames)
            {
                var values = slotNames.Values;
                if (Utils.Size(values) < _size)
                    values = new ReadOnlyMemory<char>[_size];

                for (int i = 0; i < _size; i++)
                    values[i] = string.Format("(Label_{0})", i).AsMemory();
                slotNames = new VBuffer<ReadOnlyMemory<char>>(_size, values);
            }
        }
    }

    public sealed class MultiOutputRegressionPerInstanceEvaluator : PerInstanceEvaluatorBase
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

        private readonly ColumnType _labelType;
        private readonly ColumnType _scoreType;
        private readonly Schema.Metadata _labelMetadata;
        private readonly Schema.Metadata _scoreMetadata;

        public MultiOutputRegressionPerInstanceEvaluator(IHostEnvironment env, Schema schema, string scoreCol,
            string labelCol)
            : base(env, schema, scoreCol, labelCol)
        {
            CheckInputColumnTypes(schema, out _labelType, out _scoreType, out _labelMetadata, out _scoreMetadata);
        }

        private MultiOutputRegressionPerInstanceEvaluator(IHostEnvironment env, ModelLoadContext ctx, Schema schema)
            : base(env, ctx, schema)
        {
            CheckInputColumnTypes(schema, out _labelType, out _scoreType, out _labelMetadata, out _scoreMetadata);

            // *** Binary format **
            // base
        }

        public static MultiOutputRegressionPerInstanceEvaluator Create(IHostEnvironment env, ModelLoadContext ctx, ISchema schema)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new MultiOutputRegressionPerInstanceEvaluator(env, ctx, Schema.Create(schema));
        }

        public override void Save(ModelSaveContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format **
            // base
            base.Save(ctx);
        }

        public override Func<int, bool> GetDependencies(Func<int, bool> activeOutput)
        {
            return
                col =>
                    (activeOutput(LabelOutput) && col == LabelIndex) ||
                    (activeOutput(ScoreOutput) && col == ScoreIndex) ||
                    (activeOutput(L1Output) || activeOutput(L2Output) || activeOutput(DistCol)) &&
                    (col == ScoreIndex || col == LabelIndex);
        }

        public override Schema.Column[] GetOutputColumns()
        {
            var infos = new Schema.Column[5];
            infos[LabelOutput] = new Schema.Column(LabelCol, _labelType, _labelMetadata);
            infos[ScoreOutput] = new Schema.Column(ScoreCol, _scoreType, _scoreMetadata);
            infos[L1Output] = new Schema.Column(L1, NumberType.R8, null);
            infos[L2Output] = new Schema.Column(L2, NumberType.R8, null);
            infos[DistCol] = new Schema.Column(Dist, NumberType.R8, null);
            return infos;
        }

        public override Delegate[] CreateGetters(IRow input, Func<int, bool> activeCols, out Action disposer)
        {
            Host.Assert(LabelIndex >= 0);
            Host.Assert(ScoreIndex >= 0);

            disposer = null;

            long cachedPosition = -1;
            var label = default(VBuffer<Float>);
            var score = default(VBuffer<Float>);

            ValueGetter<VBuffer<Float>> nullGetter = (ref VBuffer<Float> vec) => vec = default(VBuffer<Float>);
            var labelGetter = activeCols(LabelOutput) || activeCols(L1Output) || activeCols(L2Output) || activeCols(DistCol)
                ? RowCursorUtils.GetVecGetterAs<Float>(NumberType.Float, input, LabelIndex)
                : nullGetter;
            var scoreGetter = activeCols(ScoreOutput) || activeCols(L1Output) || activeCols(L2Output) || activeCols(DistCol)
                ? input.GetGetter<VBuffer<Float>>(ScoreIndex)
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
                ValueGetter<VBuffer<Float>> labelFn =
                    (ref VBuffer<Float> dst) =>
                    {
                        updateCacheIfNeeded();
                        label.CopyTo(ref dst);
                    };
                getters[LabelOutput] = labelFn;
            }
            if (activeCols(ScoreOutput))
            {
                ValueGetter<VBuffer<Float>> scoreFn =
                    (ref VBuffer<Float> dst) =>
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

        private void CheckInputColumnTypes(Schema schema, out ColumnType labelType, out ColumnType scoreType,
            out Schema.Metadata labelMetadata, out Schema.Metadata scoreMetadata)
        {
            Host.AssertNonEmpty(ScoreCol);
            Host.AssertNonEmpty(LabelCol);

            var t = schema.GetColumnType(LabelIndex);
            if (!t.IsKnownSizeVector || (t.ItemType != NumberType.R4 && t.ItemType != NumberType.R8))
                throw Host.Except("Label column '{0}' has type '{1}' but must be a known-size vector of R4 or R8", LabelCol, t);
            labelType = new VectorType(t.ItemType.AsPrimitive, t.VectorSize);
            var slotNamesType = new VectorType(TextType.Instance, t.VectorSize);
            var builder = new Schema.Metadata.Builder();
            builder.AddSlotNames(t.VectorSize, CreateSlotNamesGetter(schema, LabelIndex, labelType.VectorSize, "True"));
            labelMetadata = builder.GetMetadata();

            t = schema.GetColumnType(ScoreIndex);
            if (t.VectorSize == 0 || t.ItemType != NumberType.Float)
                throw Host.Except("Score column '{0}' has type '{1}' but must be a known length vector of type R4", ScoreCol, t);
            scoreType = new VectorType(t.ItemType.AsPrimitive, t.VectorSize);
            builder = new Schema.Metadata.Builder();
            builder.AddSlotNames(t.VectorSize, CreateSlotNamesGetter(schema, ScoreIndex, scoreType.VectorSize, "Predicted"));

            ValueGetter<ReadOnlyMemory<char>> getter = GetScoreColumnKind;
            builder.Add(new Schema.Column(MetadataUtils.Kinds.ScoreColumnKind, TextType.Instance, null), getter);
            getter = GetScoreValueKind;
            builder.Add(new Schema.Column(MetadataUtils.Kinds.ScoreValueKind, TextType.Instance, null), getter);
            ValueGetter<uint> uintGetter = GetScoreColumnSetId(schema);
            builder.Add(new Schema.Column(MetadataUtils.Kinds.ScoreColumnSetId, MetadataUtils.ScoreColumnSetIdType, null), uintGetter);
            scoreMetadata = builder.GetMetadata();
        }

        private ValueGetter<uint> GetScoreColumnSetId(Schema schema)
        {
            int c;
            var max = schema.GetMaxMetadataKind(out c, MetadataUtils.Kinds.ScoreColumnSetId);
            uint id = checked(max + 1);
            return
                (ref uint dst) => dst = id;
        }

        private void GetScoreColumnKind(ref ReadOnlyMemory<char> dst)
        {
            dst = MetadataUtils.Const.ScoreColumnKind.MultiOutputRegression.AsMemory();
        }

        private void GetScoreValueKind(ref ReadOnlyMemory<char> dst)
        {
            dst = MetadataUtils.Const.ScoreValueKind.Score.AsMemory();
        }

        private ValueGetter<VBuffer<ReadOnlyMemory<char>>> CreateSlotNamesGetter(ISchema schema, int column, int length, string prefix)
        {
            var type = schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.SlotNames, column);
            if (type != null && type.IsText)
            {
                return
                    (ref VBuffer<ReadOnlyMemory<char>> dst) => schema.GetMetadata(MetadataUtils.Kinds.SlotNames, column, ref dst);
            }
            return
                (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                {
                    var values = dst.Values;
                    if (Utils.Size(values) < length)
                        values = new ReadOnlyMemory<char>[length];
                    for (int i = 0; i < length; i++)
                        values[i] = string.Format("{0}_{1}", prefix, i).AsMemory();
                    dst = new VBuffer<ReadOnlyMemory<char>>(length, values);
                };
        }
    }

    public sealed class MultiOutputRegressionMamlEvaluator : MamlEvaluatorBase
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

        protected override IEvaluator Evaluator { get { return _evaluator; } }

        public MultiOutputRegressionMamlEvaluator(IHostEnvironment env, Arguments args)
            : base(args, env, MetadataUtils.Const.ScoreColumnKind.MultiOutputRegression, "RegressionMamlEvaluator")
        {
            Host.CheckUserArg(args.LossFunction != null, nameof(args.LossFunction), "Loss function must be specified");

            _supressScoresAndLabels = args.SupressScoresAndLabels;
            var evalArgs = new MultiOutputRegressionEvaluator.Arguments();
            evalArgs.LossFunction = args.LossFunction;
            _evaluator = new MultiOutputRegressionEvaluator(Host, evalArgs);
        }

        protected override IEnumerable<string> GetPerInstanceColumnsToSave(RoleMappedSchema schema)
        {
            Host.CheckValue(schema, nameof(schema));
            Host.CheckParam(schema.Label != null, nameof(schema), "Schema must contain a label column");

            // The multi output regression evaluator outputs the label and score column if requested by the user.
            if (!_supressScoresAndLabels)
            {
                yield return schema.Label.Name;

                var scoreInfo = EvaluateUtils.GetScoreColumnInfo(Host, schema.Schema, ScoreCol, nameof(Arguments.ScoreColumn),
                    MetadataUtils.Const.ScoreColumnKind.MultiOutputRegression);
                yield return scoreInfo.Name;
            }

            // Return the output columns.
            yield return MultiOutputRegressionPerInstanceEvaluator.L1;
            yield return MultiOutputRegressionPerInstanceEvaluator.L2;
            yield return MultiOutputRegressionPerInstanceEvaluator.Dist;
        }

        // The multi-output regression evaluator prints only the per-label metrics for each fold.
        protected override void PrintFoldResultsCore(IChannel ch, Dictionary<string, IDataView> metrics)
        {
            IDataView fold;
            if (!metrics.TryGetValue(MetricKinds.OverallMetrics, out fold))
                throw ch.Except("No overall metrics found");

            int isWeightedCol;
            bool needWeighted = fold.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.IsWeighted, out isWeightedCol);

            int stratCol;
            bool hasStrats = fold.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.StratCol, out stratCol);
            int stratVal;
            bool hasStratVals = fold.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.StratVal, out stratVal);
            ch.Assert(hasStrats == hasStratVals);

            var colCount = fold.Schema.ColumnCount;
            var vBufferGetters = new ValueGetter<VBuffer<double>>[colCount];

            using (var cursor = fold.GetRowCursor(col => true))
            {
                bool isWeighted = false;
                ValueGetter<bool> isWeightedGetter;
                if (needWeighted)
                    isWeightedGetter = cursor.GetGetter<bool>(isWeightedCol);
                else
                    isWeightedGetter = (ref bool dst) => dst = false;

                ValueGetter<uint> stratGetter;
                if (hasStrats)
                {
                    var type = cursor.Schema.GetColumnType(stratCol);
                    stratGetter = RowCursorUtils.GetGetterAs<uint>(type, cursor, stratCol);
                }
                else
                    stratGetter = (ref uint dst) => dst = 0;

                int labelCount = 0;
                for (int i = 0; i < fold.Schema.ColumnCount; i++)
                {
                    if (fold.Schema.IsHidden(i) || (needWeighted && i == isWeightedCol) ||
                        (hasStrats && (i == stratCol || i == stratVal)))
                    {
                        continue;
                    }

                    var type = fold.Schema.GetColumnType(i);
                    if (type.IsKnownSizeVector && type.ItemType == NumberType.R8)
                    {
                        vBufferGetters[i] = cursor.GetGetter<VBuffer<double>>(i);
                        if (labelCount == 0)
                            labelCount = type.VectorSize;
                        else
                            ch.Check(labelCount == type.VectorSize, "All vector metrics should contain the same number of slots");
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
                bool foundWeighted = !needWeighted;
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

                            sb.AppendFormat("{0}{1,12}:", isWeighted ? "Weighted " : "", fold.Schema.GetColumnName(i));
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

    public static partial class Evaluate
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
            var evaluator = new MultiOutputRegressionMamlEvaluator(host, input);
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
