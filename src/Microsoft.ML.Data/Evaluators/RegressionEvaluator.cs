// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(typeof(RegressionEvaluator), typeof(RegressionEvaluator), typeof(RegressionEvaluator.Arguments), typeof(SignatureEvaluator),
    "Regression Evaluator", RegressionEvaluator.LoadName, "Regression")]

[assembly: LoadableClass(typeof(RegressionMamlEvaluator), typeof(RegressionMamlEvaluator), typeof(RegressionMamlEvaluator.Arguments), typeof(SignatureMamlEvaluator),
    "Regression Evaluator", RegressionEvaluator.LoadName, "Regression")]

// This is for deserialization from a binary model file.
[assembly: LoadableClass(typeof(RegressionPerInstanceEvaluator), null, typeof(SignatureLoadRowMapper),
    "", RegressionPerInstanceEvaluator.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    public sealed class RegressionEvaluator :
        RegressionEvaluatorBase<RegressionEvaluator.Aggregator, Float, Double>
    {
        public sealed class Arguments : ArgumentsBase
        {
        }

        public enum Metrics
        {
            [EnumValueDisplay(RegressionEvaluator.L1)]
            L1,
            [EnumValueDisplay(RegressionEvaluator.L2)]
            L2,
            [EnumValueDisplay(RegressionEvaluator.Rms)]
            Rms,
            [EnumValueDisplay(RegressionEvaluator.Loss)]
            Loss,
            [EnumValueDisplay(RegressionEvaluator.RSquared)]
            RSquared,
        }

        public const string LoadName = "RegressionEvaluator";

        public RegressionEvaluator(IHostEnvironment env, Arguments args)
            : base(args, env, LoadName)
        {
        }

        protected override void CheckScoreAndLabelTypes(RoleMappedSchema schema)
        {
            var score = schema.GetUniqueColumn(MetadataUtils.Const.ScoreValueKind.Score);
            var t = score.Type;
            if (t.IsVector || t.ItemType != NumberType.Float)
                throw Host.Except("Score column '{0}' has type '{1}' but must be R4", score, t);
            Host.Check(schema.Label != null, "Could not find the label column");
            t = schema.Label.Type;
            if (t != NumberType.R4)
                throw Host.Except("Label column '{0}' has type '{1}' but must be R4", schema.Label.Name, t);
        }

        protected override Aggregator GetAggregatorCore(RoleMappedSchema schema, string stratName)
        {
            return new Aggregator(Host, LossFunction, schema.Weight != null, stratName);
        }

        protected override IRowMapper CreatePerInstanceRowMapper(RoleMappedSchema schema)
        {
            Contracts.CheckParam(schema.Label != null, nameof(schema), "Could not find the label column");
            var scoreInfo = schema.GetUniqueColumn(MetadataUtils.Const.ScoreValueKind.Score);
            Contracts.AssertValue(scoreInfo);

            return new RegressionPerInstanceEvaluator(Host, schema.Schema, scoreInfo.Name, schema.Label.Name);
        }

        public override IEnumerable<MetricColumn> GetOverallMetricColumns()
        {
            yield return new MetricColumn("L1", L1, MetricColumn.Objective.Minimize);
            yield return new MetricColumn("L2", L2, MetricColumn.Objective.Minimize);
            yield return new MetricColumn("Rms", Rms, MetricColumn.Objective.Minimize);
            yield return new MetricColumn("Loss", Loss, MetricColumn.Objective.Minimize);
            yield return new MetricColumn("RSquared", RSquared);
        }

        public sealed class Aggregator : RegressionAggregatorBase
        {
            private sealed class Counters : CountersBase
            {
                public override double Rms
                {
                    get
                    {
                        return SumWeights > 0 ? Math.Sqrt(TotalL2Loss / SumWeights) : 0;
                    }
                }

                public override double RSquared
                {
                    get
                    {
                        return SumWeights > 0 ? 1 - TotalL2Loss / (TotalLabelSquaredW - TotalLabelW * TotalLabelW / SumWeights) : 0;
                    }
                }

                protected override void UpdateCore(Float label, ref float score, ref double loss, Float weight)
                {
                    Double currL1Loss = Math.Abs((Double)label - score);
                    TotalL1Loss += currL1Loss * weight;
                    TotalL2Loss += currL1Loss * currL1Loss * weight;
                    TotalLoss += loss * weight; // REVIEW: Fix this! += (Double)loss * wht; //Loss as reported by regressor, note it can result in NaN if loss is NaN
                }

                protected override void Normalize(ref double src, ref double dst)
                {
                    dst = src / SumWeights;
                }

                protected override double Zero()
                {
                    return 0;
                }
            }

            private readonly Counters _counters;
            private readonly Counters _weightedCounters;

            public override CountersBase UnweightedCounters { get { return _counters; } }

            public override CountersBase WeightedCounters { get { return _weightedCounters; } }

            public Aggregator(IHostEnvironment env, IRegressionLoss lossFunction, bool weighted, string stratName)
                : base(env, lossFunction, weighted, stratName)
            {
                _counters = new Counters();
                _weightedCounters = Weighted ? new Counters() : null;
            }

            protected override void ApplyLossFunction(ref float score, float label, ref double loss)
            {
                loss = LossFunction.Loss(score, label);
            }

            protected override bool IsNaN(ref Float score)
            {
                return Float.IsNaN(score);
            }

            public override void AddColumn(ArrayDataViewBuilder dvBldr, string metricName, params double[] metric)
            {
                Host.AssertValue(dvBldr);
                dvBldr.AddColumn(metricName, NumberType.R8, metric);
            }
        }

        public sealed class Result
        {
            /// <summary>
            /// Gets the absolute loss of the model.
            /// </summary>
            /// <remarks>
            /// The absolute loss is defined as
            /// L1 = (1/m) * sum( abs( yi - y&apos;i))
            /// where m is the number of instances in the test set.
            /// y'i are the predicted labels for each instance.
            /// yi are the correct labels of each instance.
            /// </remarks>
            public double L1 { get; }

            /// <summary>
            /// Gets the squared loss of the model.
            /// </summary>
            /// <remarks>
            /// The squared loss is defined as
            /// L2 = (1/m) * sum(( yi - y&apos;i)^2)
            /// where m is the number of instances in the test set.
            /// y'i are the predicted labels for each instance.
            /// yi are the correct labels of each instance.
            /// </remarks>
            public double L2 { get; }

            /// <summary>
            /// Gets the root mean square loss (or RMS) which is the square root of the L2 loss.
            /// </summary>
            public double Rms { get; }

            /// <summary>
            /// Gets the user defined loss function.
            /// </summary>
            /// <remarks>
            /// This is the average of a loss function defined by the user,
            /// computed over all the instances in the test set.
            /// </remarks>
            public double LossFn { get; }

            /// <summary>
            /// Gets the R squared value of the model, which is also known as
            /// the coefficient of determination​.
            /// </summary>
            public double RSquared { get; }

            internal Result(IExceptionContext ectx, IRow overallResult)
            {
                double Fetch(string name) => RowCursorUtils.Fetch<double>(ectx, overallResult, name);
                L1 = Fetch(RegressionEvaluator.L1);
                L2 = Fetch(RegressionEvaluator.L2);
                Rms = Fetch(RegressionEvaluator.Rms);
                LossFn = Fetch(RegressionEvaluator.Loss);
                RSquared = Fetch(RegressionEvaluator.RSquared);
            }
        }

        /// <summary>
        /// Evaluates scored regression data.
        /// </summary>
        /// <param name="data">The data to evaluate.</param>
        /// <param name="label">The name of the label column.</param>
        /// <param name="score">The name of the predicted score column.</param>
        /// <returns>The evaluation metrics for these outputs.</returns>
        public Result Evaluate(IDataView data, string label, string score)
        {
            Host.CheckValue(data, nameof(data));
            Host.CheckNonEmpty(label, nameof(label));
            Host.CheckNonEmpty(score, nameof(score));
            var roles = new RoleMappedData(data, opt: false,
                RoleMappedSchema.ColumnRole.Label.Bind(label),
                RoleMappedSchema.CreatePair(MetadataUtils.Const.ScoreValueKind.Score, score));

            var resultDict = Evaluate(roles);
            Host.Assert(resultDict.ContainsKey(MetricKinds.OverallMetrics));
            var overall = resultDict[MetricKinds.OverallMetrics];

            Result result;
            using (var cursor = overall.GetRowCursor(i => true))
            {
                var moved = cursor.MoveNext();
                Host.Assert(moved);
                result = new Result(Host, cursor);
                moved = cursor.MoveNext();
                Host.Assert(!moved);
            }
            return result;
        }
    }

    public sealed class RegressionPerInstanceEvaluator : PerInstanceEvaluatorBase
    {
        public const string LoaderSignature = "RegressionPerInstance";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "REG INST",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(RegressionPerInstanceEvaluator).Assembly.FullName);
        }

        private const int L1Col = 0;
        private const int L2Col = 1;

        public const string L1 = "L1-loss";
        public const string L2 = "L2-loss";

        public RegressionPerInstanceEvaluator(IHostEnvironment env, ISchema schema, string scoreCol, string labelCol)
            : base(env, schema, scoreCol, labelCol)
        {
            CheckInputColumnTypes(schema);
        }

        private RegressionPerInstanceEvaluator(IHostEnvironment env, ModelLoadContext ctx, ISchema schema)
            : base(env, ctx, schema)
        {
            CheckInputColumnTypes(schema);

            // *** Binary format **
            // base
        }

        public static RegressionPerInstanceEvaluator Create(IHostEnvironment env, ModelLoadContext ctx, ISchema schema)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new RegressionPerInstanceEvaluator(env, ctx, schema);
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
                col => (activeOutput(L1Col) || activeOutput(L2Col)) && (col == ScoreIndex || col == LabelIndex);
        }

        public override Schema.Column[] GetOutputColumns()
        {
            var infos = new Schema.Column[2];
            infos[L1Col] = new Schema.Column(L1, NumberType.R8, null);
            infos[L2Col] = new Schema.Column(L2, NumberType.R8, null);
            return infos;
        }

        public override Delegate[] CreateGetters(IRow input, Func<int, bool> activeCols, out Action disposer)
        {
            Host.Assert(LabelIndex >= 0);
            Host.Assert(ScoreIndex >= 0);

            disposer = null;

            long cachedPosition = -1;
            Float label = 0;
            Float score = 0;

            ValueGetter<Float> nan = (ref Float value) => value = Single.NaN;
            var labelGetter = activeCols(L1Col) || activeCols(L2Col) ? RowCursorUtils.GetLabelGetter(input, LabelIndex) : nan;
            ValueGetter<Float> scoreGetter;
            if (activeCols(L1Col) || activeCols(L2Col))
                scoreGetter = input.GetGetter<Float>(ScoreIndex);
            else
                scoreGetter = nan;
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

            var getters = new Delegate[2];
            if (activeCols(L1Col))
            {
                ValueGetter<double> l1Fn =
                    (ref double dst) =>
                    {
                        updateCacheIfNeeded();
                        dst = Math.Abs((Double)label - score);
                    };
                getters[L1Col] = l1Fn;
            }
            if (activeCols(L2Col))
            {
                ValueGetter<double> l2Fn =
                    (ref double dst) =>
                    {
                        updateCacheIfNeeded();
                        dst = Math.Abs((Double)label - score);
                        dst *= dst;
                    };
                getters[L2Col] = l2Fn;
            }
            return getters;
        }

        private void CheckInputColumnTypes(ISchema schema)
        {
            Host.AssertNonEmpty(ScoreCol);
            Host.AssertNonEmpty(LabelCol);

            var t = schema.GetColumnType(LabelIndex);
            if (t != NumberType.R4)
                throw Host.Except("Label column '{0}' has type '{1}' but must be R4", LabelCol, t);

            t = schema.GetColumnType(ScoreIndex);
            if (t.IsVector || t.ItemType != NumberType.Float)
                throw Host.Except("Score column '{0}' has type '{1}' but must be R4", ScoreCol, t);
        }
    }

    public sealed class RegressionMamlEvaluator : MamlEvaluatorBase
    {
        public sealed class Arguments : ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "Loss function", ShortName = "loss")]
            public ISupportRegressionLossFactory LossFunction = new SquaredLossFactory();
        }

        private readonly RegressionEvaluator _evaluator;

        protected override IEvaluator Evaluator { get { return _evaluator; } }

        public RegressionMamlEvaluator(IHostEnvironment env, Arguments args)
            : base(args, env, MetadataUtils.Const.ScoreColumnKind.Regression, "RegressionMamlEvaluator")
        {
            Host.CheckUserArg(args.LossFunction != null, nameof(args.LossFunction), "Loss function must be specified.");

            var evalArgs = new RegressionEvaluator.Arguments();
            evalArgs.LossFunction = args.LossFunction;
            _evaluator = new RegressionEvaluator(Host, evalArgs);
        }

        protected override IEnumerable<string> GetPerInstanceColumnsToSave(RoleMappedSchema schema)
        {
            Host.CheckValue(schema, nameof(schema));
            Host.CheckParam(schema.Label != null, nameof(schema), "Schema must contain a label column");

            // The regression evaluator outputs the label and score columns.
            yield return schema.Label.Name;
            var scoreInfo = EvaluateUtils.GetScoreColumnInfo(Host, schema.Schema, ScoreCol, nameof(Arguments.ScoreColumn),
                MetadataUtils.Const.ScoreColumnKind.Regression);
            yield return scoreInfo.Name;

            // Return the output columns.
            yield return RegressionPerInstanceEvaluator.L1;
            yield return RegressionPerInstanceEvaluator.L2;

            // REVIEW: Identify by metadata.
            int col;
            if (schema.Schema.TryGetColumnIndex("FeatureContributions", out col))
                yield return "FeatureContributions";
        }
    }

    public static partial class Evaluate
    {
        [TlcModule.EntryPoint(Name = "Models.RegressionEvaluator", Desc = "Evaluates a regression scored dataset.")]
        public static CommonOutputs.CommonEvaluateOutput Regression(IHostEnvironment env, RegressionMamlEvaluator.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("EvaluateRegression");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            string label;
            string weight;
            string name;
            MatchColumns(host, input, out label, out weight, out name);
            var evaluator = new RegressionMamlEvaluator(host, input);
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
