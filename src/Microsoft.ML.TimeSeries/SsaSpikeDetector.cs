// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.TimeSeriesProcessing;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.StaticPipe.Runtime;

[assembly: LoadableClass(SsaSpikeDetector.Summary, typeof(IDataTransform), typeof(SsaSpikeDetector), typeof(SsaSpikeDetector.Arguments), typeof(SignatureDataTransform),
    SsaSpikeDetector.UserName, SsaSpikeDetector.LoaderSignature, SsaSpikeDetector.ShortName)]

[assembly: LoadableClass(SsaSpikeDetector.Summary, typeof(IDataTransform), typeof(SsaSpikeDetector), null, typeof(SignatureLoadDataTransform),
    SsaSpikeDetector.UserName, SsaSpikeDetector.LoaderSignature)]

[assembly: LoadableClass(SsaSpikeDetector.Summary, typeof(SsaSpikeDetector), null, typeof(SignatureLoadModel),
    SsaSpikeDetector.UserName, SsaSpikeDetector.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(SsaSpikeDetector), null, typeof(SignatureLoadRowMapper),
   SsaSpikeDetector.UserName, SsaSpikeDetector.LoaderSignature)]

namespace Microsoft.ML.Runtime.TimeSeriesProcessing
{
    /// <summary>
    /// This class implements the spike detector transform based on Singular Spectrum modeling of the time-series.
    /// For the details of the Singular Spectrum Analysis (SSA), refer to http://arxiv.org/pdf/1206.6910.pdf.
    /// </summary>
    public sealed class SsaSpikeDetector : SsaAnomalyDetectionBase
    {
        internal const string Summary = "This transform detects the spikes in a seasonal time-series using Singular Spectrum Analysis (SSA).";
        public const string LoaderSignature = "SsaSpikeDetector";
        public const string UserName = "SSA Spike Detection";
        public const string ShortName = "spike";

        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "The name of the source column.", ShortName = "src",
                SortOrder = 1, Purpose = SpecialPurpose.ColumnName)]
            public string Source;

            [Argument(ArgumentType.Required, HelpText = "The name of the new column.",
                SortOrder = 2)]
            public string Name;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The argument that determines whether to detect positive or negative anomalies, or both.", ShortName = "side",
                SortOrder = 101)]
            public AnomalySide Side = AnomalySide.TwoSided;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The size of the sliding window for computing the p-value.", ShortName = "wnd",
                SortOrder = 102)]
            public int PvalueHistoryLength = 100;

            [Argument(ArgumentType.Required, HelpText = "The number of points from the beginning of the sequence used for training.",
                ShortName = "twnd", SortOrder = 3)]
            public int TrainingWindowSize = 100;

            [Argument(ArgumentType.Required, HelpText = "The confidence for spike detection in the range [0, 100].",
                ShortName = "cnf", SortOrder = 4)]
            public double Confidence = 99;

            [Argument(ArgumentType.Required, HelpText = "An upper bound on the largest relevant seasonality in the input time-series.", ShortName = "swnd", SortOrder = 5)]
            public int SeasonalWindowSize = 10;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The function used to compute the error between the expected and the observed value.", ShortName = "err", SortOrder = 103)]
            public ErrorFunctionUtils.ErrorFunction ErrorFunction = ErrorFunctionUtils.ErrorFunction.SignedDifference;
        }

        private sealed class BaseArguments : SsaArguments
        {
            public BaseArguments(Arguments args)
            {
                Source = args.Source;
                Name = args.Name;
                Side = args.Side;
                WindowSize = args.PvalueHistoryLength;
                InitialWindowSize = args.TrainingWindowSize;
                SeasonalWindowSize = args.SeasonalWindowSize;
                AlertThreshold = 1 - args.Confidence / 100;
                AlertOn = SequentialAnomalyDetectionTransformBase<float, State>.AlertingScore.PValueScore;
                DiscountFactor = 1;
                IsAdaptive = false;
                ErrorFunction = args.ErrorFunction;
                Martingale = MartingaleType.None;
            }
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "SSPKTRNS",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(SsaSpikeDetector).Assembly.FullName);
        }

        // Factory method for SignatureDataTransform.
        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));

            var transform = new SsaSpikeDetector(env, args);
            transform.Fit(input);
            return transform.MakeDataTransform(input);
        }

        internal SsaSpikeDetector(IHostEnvironment env, Arguments args)
            : base(new BaseArguments(args), LoaderSignature, env)
        {
            // This constructor is empty.
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            env.CheckValue(input, nameof(input));

            return new SsaSpikeDetector(env, ctx).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadModel.
        private static SsaSpikeDetector Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new SsaSpikeDetector(env, ctx);
        }

        internal SsaSpikeDetector(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, ctx, LoaderSignature)
        {
            // *** Binary format ***
            // <base>

            Host.CheckDecode(ThresholdScore == AlertingScore.PValueScore);
            Host.CheckDecode(DiscountFactor == 1);
            Host.CheckDecode(IsAdaptive == false);
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            Host.Assert(ThresholdScore == AlertingScore.PValueScore);
            Host.Assert(DiscountFactor == 1);
            Host.Assert(IsAdaptive == false);

            // *** Binary format ***
            // <base>

            base.Save(ctx);
        }

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, Schema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);
    }

    public sealed class SsaSpikeEstimator : IEstimator<ITransformer>
    {
        private readonly IHost _host;
        private readonly string _inputColumnName;
        private readonly string _outputColumnName;
        private readonly int _confidence;
        private readonly int _pvalueHistoryLength;
        private readonly int _trainingWindowSize;
        private readonly int _seasonalityWindowSize;

        public SsaSpikeEstimator(
            IHostEnvironment env,
            int confidence,
            int pvalueHistoryLength,
            int trainingWindowSize,
            int seasonalityWindowSize,
            string input,
            string output)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register("SsaSpikeEstimator");

            _host.CheckNonEmpty(input, nameof(input));
            _host.CheckNonEmpty(output, nameof(output));

            _confidence = confidence;
            _pvalueHistoryLength = pvalueHistoryLength;
            _trainingWindowSize = trainingWindowSize;
            _seasonalityWindowSize = seasonalityWindowSize;
            _inputColumnName = input;
            _outputColumnName = output;
        }

        public ITransformer Fit(IDataView input)
        {
            _host.CheckValue(input, nameof(input));
            var transformer = new SsaSpikeDetector(_host,
                new SsaSpikeDetector.Arguments
                {
                    Confidence = _confidence,
                    PvalueHistoryLength = _pvalueHistoryLength,
                    TrainingWindowSize = _trainingWindowSize,
                    SeasonalWindowSize = _seasonalityWindowSize,
                    Source = _inputColumnName,
                    Name = _outputColumnName
                });
            transformer.Fit(input);
            return transformer;
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var resultDic = inputSchema.Columns.ToDictionary(x => x.Name);

            resultDic[_outputColumnName] = new SchemaShape.Column(
                _outputColumnName, SchemaShape.Column.VectorKind.Vector, NumberType.R8, false);

            return new SchemaShape(resultDic.Values);
        }
    }

    public static class SsaSpikeStaticExtensions
    {
        private sealed class OutColumn : Vector<float>
        {
            public PipelineColumn Input { get; }

            public OutColumn(Vector<float> input,
                int confidence,
                int pvalueHistoryLength,
                int trainingWindowSize,
                int seasonalityWindowSize)
                : base(new Reconciler(confidence, pvalueHistoryLength, trainingWindowSize, seasonalityWindowSize), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly int _confidence;
            private readonly int _pvalueHistoryLength;
            private readonly int _trainingWindowSize;
            private readonly int _seasonalityWindowSize;

            public Reconciler(
                int confidence,
                int pvalueHistoryLength,
                int trainingWindowSize,
                int seasonalityWindowSize)
            {
                _confidence = confidence;
                _pvalueHistoryLength = pvalueHistoryLength;
                _trainingWindowSize = trainingWindowSize;
                _seasonalityWindowSize = seasonalityWindowSize;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);
                var outCol = (OutColumn)toOutput[0];
                return new SsaSpikeEstimator(env,
                    _confidence,
                    _pvalueHistoryLength,
                    _trainingWindowSize,
                    _seasonalityWindowSize,
                    inputNames[outCol.Input], outputNames[outCol]);
            }
        }
    }
}
