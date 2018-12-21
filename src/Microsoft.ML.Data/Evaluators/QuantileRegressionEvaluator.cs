﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using System;
using System.Collections.Generic;
using Float = System.Single;

[assembly: LoadableClass(typeof(QuantileRegressionEvaluator), typeof(QuantileRegressionEvaluator), typeof(QuantileRegressionEvaluator.Arguments), typeof(SignatureEvaluator),
    "Quantile Regression Evaluator", QuantileRegressionEvaluator.LoadName, "QuantileRegression")]

[assembly: LoadableClass(typeof(QuantileRegressionMamlEvaluator), typeof(QuantileRegressionMamlEvaluator), typeof(QuantileRegressionMamlEvaluator.Arguments), typeof(SignatureMamlEvaluator),
    "Quantile Regression Evaluator", QuantileRegressionEvaluator.LoadName, "QuantileRegression")]

// This is for deserialization from a binary model file.
[assembly: LoadableClass(typeof(QuantileRegressionPerInstanceEvaluator), null, typeof(SignatureLoadRowMapper),
    "", QuantileRegressionPerInstanceEvaluator.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    public sealed class QuantileRegressionEvaluator :
        RegressionEvaluatorBase<QuantileRegressionEvaluator.Aggregator, VBuffer<Float>, VBuffer<Double>>
    {
        public sealed class Arguments : ArgumentsBase
        {
        }

        public const string LoadName = "QuantileRegressionEvaluator";

        public QuantileRegressionEvaluator(IHostEnvironment env, Arguments args)
            : base(args, env, LoadName)
        {
        }

        private protected override IRowMapper CreatePerInstanceRowMapper(RoleMappedSchema schema)
        {
            Host.CheckParam(schema.Label.HasValue, nameof(schema), "Must contain a label column");
            var scoreInfo = schema.GetUniqueColumn(MetadataUtils.Const.ScoreValueKind.Score);
            int scoreSize = scoreInfo.Type.VectorSize;
            var type = schema.Schema[scoreInfo.Index].Metadata.Schema.GetColumnOrNull(MetadataUtils.Kinds.SlotNames)?.Type;
            Host.Check(type != null && type.IsKnownSizeVector && type.ItemType.IsText, "Quantile regression score column must have slot names");
            var quantiles = default(VBuffer<ReadOnlyMemory<char>>);
            schema.Schema[scoreInfo.Index].Metadata.GetValue(MetadataUtils.Kinds.SlotNames, ref quantiles);
            Host.Assert(quantiles.IsDense && quantiles.Length == scoreSize);

            return new QuantileRegressionPerInstanceEvaluator(Host, schema.Schema, scoreInfo.Name, schema.Label.Value.Name, scoreSize, quantiles);
        }

        private protected override void CheckScoreAndLabelTypes(RoleMappedSchema schema)
        {
            var score = schema.GetUniqueColumn(MetadataUtils.Const.ScoreValueKind.Score);
            var t = score.Type;
            if (t.VectorSize == 0 || (t.ItemType != NumberType.R4 && t.ItemType != NumberType.R8))
                throw Host.ExceptSchemaMismatch(nameof(schema), "score", score.Name, "vector of type R4 or R8", t.ToString());
            Host.CheckParam(schema.Label.HasValue, nameof(schema), "Must contain a label column");
            t = schema.Label.Value.Type;
            if (t != NumberType.R4)
                throw Host.ExceptSchemaMismatch(nameof(schema), "label", schema.Label.Value.Name, "R4", t.ToString());
        }

        private protected override Aggregator GetAggregatorCore(RoleMappedSchema schema, string stratName)
        {
            var scoreInfo = schema.GetUniqueColumn(MetadataUtils.Const.ScoreValueKind.Score);
            var t = scoreInfo.Type;
            Host.Assert(t.VectorSize > 0 && (t.ItemType == NumberType.R4 || t.ItemType == NumberType.R8));
            var slotNames = default(VBuffer<ReadOnlyMemory<char>>);
            t = schema.Schema[scoreInfo.Index].Metadata.Schema.GetColumnOrNull(MetadataUtils.Kinds.SlotNames)?.Type;
            if (t != null && t.VectorSize == scoreInfo.Type.VectorSize && t.ItemType.IsText)
                schema.Schema[scoreInfo.Index].GetSlotNames(ref slotNames);
            return new Aggregator(Host, LossFunction, schema.Weight != null, scoreInfo.Type.VectorSize, in slotNames, stratName);
        }

        public override IEnumerable<MetricColumn> GetOverallMetricColumns()
        {
            yield return new MetricColumn("L1", L1, MetricColumn.Objective.Minimize, isVector: true);
            yield return new MetricColumn("L2", L2, MetricColumn.Objective.Minimize, isVector: true);
            yield return new MetricColumn("Rms", Rms, MetricColumn.Objective.Minimize, isVector: true);
            yield return new MetricColumn("Loss", Loss, MetricColumn.Objective.Minimize, isVector: true);
            yield return new MetricColumn("RSquared", RSquared, isVector: true);
        }

        public sealed class Aggregator : RegressionAggregatorBase
        {
            private sealed class Counters : CountersBase
            {
                private readonly int _size;

                public override VBuffer<Double> Rms
                {
                    get
                    {
                        var res = new Double[_size];
                        if (SumWeights != 0)
                        {
                            foreach (var i in TotalL2Loss.Items())
                                res[i.Key] = Math.Sqrt(i.Value / SumWeights);
                        }
                        return new VBuffer<Double>(_size, res);
                    }
                }

                public override VBuffer<Double> RSquared
                {
                    get
                    {
                        var res = new Double[_size];
                        if (SumWeights != 0)
                        {
                            foreach (var i in TotalL2Loss.Items())
                                res[i.Key] = 1 - i.Value / (TotalLabelSquaredW - TotalLabelW * TotalLabelW / SumWeights);
                        }
                        return new VBuffer<Double>(_size, res);
                    }
                }

                public Counters(int size)
                {
                    Contracts.Assert(size > 0);
                    _size = size;
                    TotalL1Loss = VBufferUtils.CreateDense<Double>(size);
                    TotalL2Loss = VBufferUtils.CreateDense<Double>(size);
                    TotalLoss = VBufferUtils.CreateDense<Double>(size);
                }

                protected override void UpdateCore(Float label, in VBuffer<Float> score, in VBuffer<Double> loss, Float weight)
                {
                    AddL1AndL2Loss(label, in score, weight);
                    AddCustomLoss(weight, in loss);
                }

                private void AddL1AndL2Loss(Float label, in VBuffer<Float> score, Float weight)
                {
                    Contracts.Check(score.Length == TotalL1Loss.Length, "Vectors must have the same dimensionality.");

                    var totalL1LossEditor = VBufferEditor.CreateFromBuffer(ref TotalL1Loss);
                    var totalL2LossEditor = VBufferEditor.CreateFromBuffer(ref TotalL2Loss);

                    var scoreValues = score.GetValues();
                    if (score.IsDense)
                    {
                        // Both are dense.
                        for (int i = 0; i < scoreValues.Length; i++)
                        {
                            var diff = Math.Abs((Double)label - scoreValues[i]);
                            var weightedDiff = diff * weight;
                            totalL1LossEditor.Values[i] += weightedDiff;
                            totalL2LossEditor.Values[i] += diff * weightedDiff;
                        }
                        return;
                    }

                    // score is sparse, and _totalL1Loss is dense.
                    var scoreIndices = score.GetIndices();
                    for (int i = 0; i < scoreValues.Length; i++)
                    {
                        var diff = Math.Abs((Double)label - scoreValues[i]);
                        var weightedDiff = diff * weight;
                        totalL1LossEditor.Values[scoreIndices[i]] += weightedDiff;
                        totalL2LossEditor.Values[scoreIndices[i]] += diff * weightedDiff;
                    }
                }

                private void AddCustomLoss(Float weight, in VBuffer<Double> loss)
                {
                    Contracts.Check(loss.Length == TotalL1Loss.Length, "Vectors must have the same dimensionality.");

                    var totalLossEditor = VBufferEditor.CreateFromBuffer(ref TotalLoss);

                    var lossValues = loss.GetValues();
                    if (loss.IsDense)
                    {
                        // Both are dense.
                        for (int i = 0; i < lossValues.Length; i++)
                            totalLossEditor.Values[i] += lossValues[i] * weight;
                        return;
                    }

                    // loss is sparse, and _totalL1Loss is dense.
                    var lossIndices = loss.GetIndices();
                    for (int i = 0; i < lossValues.Length; i++)
                        totalLossEditor.Values[lossIndices[i]] += lossValues[i] * weight;
                }

                protected override void Normalize(in VBuffer<Double> src, ref VBuffer<Double> dst)
                {
                    Contracts.Assert(SumWeights > 0);
                    Contracts.Assert(src.IsDense);

                    var editor = VBufferEditor.Create(ref dst, src.Length);
                    var inv = 1 / SumWeights;
                    var srcValues = src.GetValues();
                    for (int i = 0; i < srcValues.Length; i++)
                        editor.Values[i] = srcValues[i] * inv;
                    dst = editor.Commit();
                }

                protected override VBuffer<Double> Zero()
                {
                    return VBufferUtils.CreateDense<Double>(_size);
                }
            }

            private readonly Counters _counters;
            private readonly Counters _weightedCounters;

            private VBuffer<ReadOnlyMemory<char>> _slotNames;

            public override CountersBase UnweightedCounters { get { return _counters; } }

            public override CountersBase WeightedCounters { get { return _weightedCounters; } }

            public Aggregator(IHostEnvironment env, IRegressionLoss lossFunction, bool weighted, int size,
                in VBuffer<ReadOnlyMemory<char>> slotNames, string stratName)
                : base(env, lossFunction, weighted, stratName)
            {
                Host.Assert(size > 0);
                Host.Assert(slotNames.Length == 0 || slotNames.Length == size);
                Score = new VBuffer<float>(size, 0, null, null);
                Loss = new VBuffer<Double>(size, 0, null, null);
                _counters = new Counters(size);
                if (Weighted)
                    _weightedCounters = new Counters(size);
                _slotNames = slotNames;
            }

            protected override void ApplyLossFunction(in VBuffer<float> score, float label, ref VBuffer<Double> loss)
            {
                VBufferUtils.PairManipulator<Float, Double> lossFn =
                    (int slot, Float src, ref Double dst) => dst = LossFunction.Loss(src, label);
                VBufferUtils.ApplyWith(in score, ref loss, lossFn);
            }

            protected override bool IsNaN(in VBuffer<Float> score)
            {
                return VBufferUtils.HasNaNs(in score);
            }

            public override void AddColumn(ArrayDataViewBuilder dvBldr, string metricName, params VBuffer<Double>[] metric)
            {
                Host.AssertValue(dvBldr);
                if (_slotNames.Length > 0)
                {
                    ValueGetter<VBuffer<ReadOnlyMemory<char>>> getSlotNames =
                        (ref VBuffer<ReadOnlyMemory<char>> dst) => dst = _slotNames;
                    dvBldr.AddColumn(metricName, getSlotNames, NumberType.R8, metric);
                }
                else
                    dvBldr.AddColumn(metricName, NumberType.R8, metric);
            }
        }
    }

    public sealed class QuantileRegressionPerInstanceEvaluator : PerInstanceEvaluatorBase
    {
        public const string LoaderSignature = "QuantileRegPerInstance";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "QREGINST",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(QuantileRegressionPerInstanceEvaluator).Assembly.FullName);
        }

        private const int L1Col = 0;
        private const int L2Col = 1;

        public const string L1 = "L1-loss";
        public const string L2 = "L2-loss";

        private readonly int _scoreSize;
        private readonly VBuffer<ReadOnlyMemory<char>> _quantiles;
        private readonly ColumnType _outputType;

        public QuantileRegressionPerInstanceEvaluator(IHostEnvironment env, Schema schema, string scoreCol, string labelCol, int scoreSize, VBuffer<ReadOnlyMemory<char>> quantiles)
            : base(env, schema, scoreCol, labelCol)
        {
            Host.CheckParam(scoreSize > 0, nameof(scoreSize), "must be greater than 0");
            if (!quantiles.IsDense)
                throw Host.ExceptParam(nameof(quantiles), "buffer must be dense");
            if (quantiles.Length != scoreSize)
                throw Host.ExceptParam(nameof(quantiles), "buffer must be of length '{0}'", scoreSize);
            CheckInputColumnTypes(schema);
            _scoreSize = scoreSize;
            _quantiles = quantiles;
            _outputType = new VectorType(NumberType.R8, _scoreSize);
        }

        private QuantileRegressionPerInstanceEvaluator(IHostEnvironment env, ModelLoadContext ctx, Schema schema)
            : base(env, ctx, schema)
        {
            CheckInputColumnTypes(schema);

            // *** Binary format **
            // base
            // int: _scoreSize
            // int[]: Ids of the quantile names

            _scoreSize = ctx.Reader.ReadInt32();
            Host.CheckDecode(_scoreSize > 0);
            ReadOnlyMemory<char>[] quantiles = new ReadOnlyMemory<char>[_scoreSize];
            for (int i = 0; i < _scoreSize; i++)
                quantiles[i] = ctx.LoadNonEmptyString().AsMemory();
            _quantiles = new VBuffer<ReadOnlyMemory<char>>(quantiles.Length, quantiles);
            _outputType = new VectorType(NumberType.R8, _scoreSize);
        }

        public static QuantileRegressionPerInstanceEvaluator Create(IHostEnvironment env, ModelLoadContext ctx, Schema schema)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new QuantileRegressionPerInstanceEvaluator(env, ctx, schema);
        }

        public override void Save(ModelSaveContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format **
            // base
            // int: _scoreSize
            // int[]: Ids of the quantile names

            base.Save(ctx);
            Host.Assert(_scoreSize > 0);
            ctx.Writer.Write(_scoreSize);
            var quantiles = _quantiles.GetValues();
            for (int i = 0; i < _scoreSize; i++)
                ctx.SaveNonEmptyString(quantiles[i].ToString());
        }

        private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
        {
            return
                col => (activeOutput(L1Col) || activeOutput(L2Col)) && (col == ScoreIndex || col == LabelIndex);
        }

        private protected override Schema.DetachedColumn[] GetOutputColumnsCore()
        {
            var infos = new Schema.DetachedColumn[2];

            var slotNamesType = new VectorType(TextType.Instance, _scoreSize);
            var l1Metadata = new MetadataBuilder();
            l1Metadata.AddSlotNames(_scoreSize, CreateSlotNamesGetter(L1));

            var l2Metadata = new MetadataBuilder();
            l2Metadata.AddSlotNames(_scoreSize, CreateSlotNamesGetter(L2));

            infos[L1Col] = new Schema.DetachedColumn(L1, _outputType, l1Metadata.GetMetadata());
            infos[L2Col] = new Schema.DetachedColumn(L2, _outputType, l2Metadata.GetMetadata());
            return infos;
        }

        private ValueGetter<VBuffer<ReadOnlyMemory<char>>> CreateSlotNamesGetter(string prefix)
        {
            return
                (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                {
                    var editor = VBufferEditor.Create(ref dst, _scoreSize);
                    var quantiles = _quantiles.GetValues();
                    for (int i = 0; i < _scoreSize; i++)
                        editor.Values[i] = string.Format("{0} ({1})", prefix, quantiles[i]).AsMemory();
                    dst = editor.Commit();
                };
        }

        private protected override Delegate[] CreateGettersCore(Row input, Func<int, bool> activeCols, out Action disposer)
        {
            Host.Assert(LabelIndex >= 0);
            Host.Assert(ScoreIndex >= 0);

            disposer = null;

            long cachedPosition = -1;
            Float label = 0;
            var score = default(VBuffer<Float>);
            var l1 = VBufferUtils.CreateDense<Double>(_scoreSize);

            ValueGetter<Float> nanGetter = (ref Float value) => value = Single.NaN;
            var labelGetter = activeCols(L1Col) || activeCols(L2Col) ? RowCursorUtils.GetLabelGetter(input, LabelIndex) : nanGetter;
            ValueGetter<VBuffer<Float>> scoreGetter;
            if (activeCols(L1Col) || activeCols(L2Col))
                scoreGetter = input.GetGetter<VBuffer<Float>>(ScoreIndex);
            else
                scoreGetter = (ref VBuffer<Float> dst) => dst = default(VBuffer<Float>);
            Action updateCacheIfNeeded =
                () =>
                {
                    if (cachedPosition != input.Position)
                    {
                        labelGetter(ref label);
                        scoreGetter(ref score);
                        var lab = (Double)label;
                        var l1Editor = VBufferEditor.CreateFromBuffer(ref l1);
                        foreach (var s in score.Items(all: true))
                            l1Editor.Values[s.Key] = Math.Abs(lab - s.Value);
                        cachedPosition = input.Position;
                    }
                };

            var getters = new Delegate[2];
            if (activeCols(L1Col))
            {
                ValueGetter<VBuffer<Double>> l1Fn =
                     (ref VBuffer<Double> dst) =>
                     {
                         updateCacheIfNeeded();
                         l1.CopyTo(ref dst);
                     };
                getters[L1Col] = l1Fn;
            }
            if (activeCols(L2Col))
            {
                VBufferUtils.PairManipulator<Double, Double> sqr =
                    (int slot, Double x, ref Double y) => y = x * x;

                ValueGetter<VBuffer<Double>> l2Fn =
                    (ref VBuffer<Double> dst) =>
                    {
                        updateCacheIfNeeded();
                        VBufferUtils.Resize(ref dst, _scoreSize, 0);
                        VBufferUtils.ApplyWith(in l1, ref dst, sqr);
                    };
                getters[L2Col] = l2Fn;
            }
            return getters;
        }

        private void CheckInputColumnTypes(Schema schema)
        {
            Host.AssertNonEmpty(ScoreCol);
            Host.AssertNonEmpty(LabelCol);

            var t = schema[(int)LabelIndex].Type;
            if (t != NumberType.R4)
                throw Host.Except("Label column '{0}' has type '{1}' but must be R4", LabelCol, t);

            t = schema[ScoreIndex].Type;
            if (t.VectorSize == 0 || (t.ItemType != NumberType.R4 && t.ItemType != NumberType.R8))
            {
                throw Host.Except(
                    "Score column '{0}' has type '{1}' but must be a known length vector of type R4 or R8", ScoreCol, t);
            }
        }
    }

    public sealed class QuantileRegressionMamlEvaluator : MamlEvaluatorBase
    {
        public sealed class Arguments : ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "Loss function", ShortName = "loss")]
            public ISupportRegressionLossFactory LossFunction = new SquaredLossFactory();

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Quantile index to select", ShortName = "ind")]
            public int? Index;
        }

        private readonly int? _index;
        private readonly QuantileRegressionEvaluator _evaluator;

        private protected override IEvaluator Evaluator => _evaluator;

        public QuantileRegressionMamlEvaluator(IHostEnvironment env, Arguments args)
            : base(args, env, MetadataUtils.Const.ScoreColumnKind.QuantileRegression, "QuantilsRegressionMamlEvaluator")
        {
            _index = args.Index;
            Host.CheckUserArg(args.LossFunction != null, nameof(args.LossFunction), "Loss function must be specified.");

            var evalArgs = new QuantileRegressionEvaluator.Arguments();
            evalArgs.LossFunction = args.LossFunction;
            _evaluator = new QuantileRegressionEvaluator(Host, evalArgs);
        }

        private protected override void PrintFoldResultsCore(IChannel ch, Dictionary<string, IDataView> metrics)
        {
            ch.AssertValue(metrics);

            IDataView fold;
            if (!metrics.TryGetValue(MetricKinds.OverallMetrics, out fold))
                throw ch.Except("No overall metrics found");

            // Show only the metrics for the requested index.
            fold = ExtractRelevantIndex(fold);

            string weightedMetrics;
            string unweightedMetrics = MetricWriter.GetPerFoldResults(Host, fold, out weightedMetrics);
            if (!string.IsNullOrEmpty(weightedMetrics))
                ch.Info(weightedMetrics);
            ch.Info(unweightedMetrics);
        }

        private protected override IDataView GetOverallResultsCore(IDataView overall)
        {
            return ExtractRelevantIndex(overall);
        }

        private IDataView ExtractRelevantIndex(IDataView data)
        {
            IDataView output = data;
            for (int i = 0; i < data.Schema.Count; i++)
            {
                var type = data.Schema[i].Type;
                if (type.IsKnownSizeVector && type.ItemType == NumberType.R8)
                {
                    var name = data.Schema[i].Name;
                    var index = _index ?? type.VectorSize / 2;
                    output = LambdaColumnMapper.Create(Host, "Quantile Regression", output, name, name, type, NumberType.R8,
                        (in VBuffer<Double> src, ref Double dst) => dst = src.GetItemOrDefault(index));
                    output = new ChooseColumnsByIndexTransform(Host,
                        new ChooseColumnsByIndexTransform.Arguments() { Drop = true, Index = new[] { i } }, output);
                }
            }
            return output;
        }

        public override IEnumerable<MetricColumn> GetOverallMetricColumns()
        {
            yield return new MetricColumn("L1", QuantileRegressionEvaluator.L1, MetricColumn.Objective.Minimize);
            yield return new MetricColumn("L2", QuantileRegressionEvaluator.L2, MetricColumn.Objective.Minimize);
            yield return new MetricColumn("Rms", QuantileRegressionEvaluator.Rms, MetricColumn.Objective.Minimize);
            yield return new MetricColumn("Loss", QuantileRegressionEvaluator.Loss, MetricColumn.Objective.Minimize);
            yield return new MetricColumn("RSquared", QuantileRegressionEvaluator.RSquared);
        }

        private protected override IEnumerable<string> GetPerInstanceColumnsToSave(RoleMappedSchema schema)
        {
            Host.CheckValue(schema, nameof(schema));
            Host.CheckParam(schema.Label.HasValue, nameof(schema), "Must contain a label column");

            // The quantile regression evaluator outputs the label and score columns.
            yield return schema.Label.Value.Name;
            var scoreCol = EvaluateUtils.GetScoreColumn(Host, schema.Schema, ScoreCol, nameof(Arguments.ScoreColumn),
                MetadataUtils.Const.ScoreColumnKind.QuantileRegression);
            yield return scoreCol.Name;

            // Return the output columns.
            yield return RegressionPerInstanceEvaluator.L1;
            yield return RegressionPerInstanceEvaluator.L2;
        }
    }

    public static partial class Evaluate
    {
        [TlcModule.EntryPoint(Name = "Models.QuantileRegressionEvaluator", Desc = "Evaluates a quantile regression scored dataset.")]
        public static CommonOutputs.CommonEvaluateOutput QuantileRegression(IHostEnvironment env, QuantileRegressionMamlEvaluator.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("EvaluateQuantileRegression");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            string label;
            string weight;
            string name;
            MatchColumns(host, input, out label, out weight, out name);
            IMamlEvaluator evaluator = new QuantileRegressionMamlEvaluator(host, input);
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
