﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Model;
using Microsoft.ML.TimeSeries;
using Microsoft.ML.TimeSeriesProcessing;
using static Microsoft.ML.TimeSeriesProcessing.SequentialAnomalyDetectionTransformBase<System.Single, Microsoft.ML.TimeSeriesProcessing.SsaAnomalyDetectionBase.State>;

[assembly: LoadableClass(SsaSpikeDetector.Summary, typeof(IDataTransform), typeof(SsaSpikeDetector), typeof(SsaSpikeDetector.Arguments), typeof(SignatureDataTransform),
    SsaSpikeDetector.UserName, SsaSpikeDetector.LoaderSignature, SsaSpikeDetector.ShortName)]

[assembly: LoadableClass(SsaSpikeDetector.Summary, typeof(IDataTransform), typeof(SsaSpikeDetector), null, typeof(SignatureLoadDataTransform),
    SsaSpikeDetector.UserName, SsaSpikeDetector.LoaderSignature)]

[assembly: LoadableClass(SsaSpikeDetector.Summary, typeof(SsaSpikeDetector), null, typeof(SignatureLoadModel),
    SsaSpikeDetector.UserName, SsaSpikeDetector.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(SsaSpikeDetector), null, typeof(SignatureLoadRowMapper),
   SsaSpikeDetector.UserName, SsaSpikeDetector.LoaderSignature)]

namespace Microsoft.ML.TimeSeriesProcessing
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

        internal SsaSpikeDetector(IHostEnvironment env, Arguments args, IDataView input)
            : base(new BaseArguments(args), LoaderSignature, env)
        {
            Model.Train(new RoleMappedData(input, null, InputColumnName));
        }

        // Factory method for SignatureDataTransform.
        private static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));

            return new SsaSpikeDetector(env, args, input).MakeDataTransform(input);
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

        internal override IStatefulTransformer Clone()
        {
            var clone = (SsaSpikeDetector)MemberwiseClone();
            clone.Model = clone.Model.Clone();
            clone.StateRef = (State)clone.StateRef.Clone();
            clone.StateRef.InitState(clone, Host);
            return clone;
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

    /// <summary>
    /// Estimator for <see cref="SsaSpikeDetector"/>
    /// </summary>
    /// <p>Example code can be found by searching for <i>SsaSpikeDetector</i> in <a href='https://github.com/dotnet/machinelearning'>ML.NET.</a></p>
    /// <example>
    /// <format type="text/markdown">
    /// <![CDATA[
    /// [!code-csharp[MF](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/SsaSpikeDetectorTransform.cs)]
    /// ]]>
    /// </format>
    /// </example>
    public sealed class SsaSpikeEstimator : IEstimator<SsaSpikeDetector>
    {
        private readonly IHost _host;
        private readonly SsaSpikeDetector.Arguments _args;

        /// <summary>
        /// Create a new instance of <see cref="SsaSpikeEstimator"/>
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="confidence">The confidence for spike detection in the range [0, 100].</param>
        /// <param name="pvalueHistoryLength">The size of the sliding window for computing the p-value.</param>
        /// <param name="trainingWindowSize">The number of points from the beginning of the sequence used for training.</param>
        /// <param name="seasonalityWindowSize">An upper bound on the largest relevant seasonality in the input time-series.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.
        /// The vector contains Alert, Raw Score, P-Value as first three values.</param>
        /// <param name="side">The argument that determines whether to detect positive or negative anomalies, or both.</param>
        /// <param name="errorFunction">The function used to compute the error between the expected and the observed value.</param>
        public SsaSpikeEstimator(IHostEnvironment env,
            string outputColumnName,
            int confidence,
            int pvalueHistoryLength,
            int trainingWindowSize,
            int seasonalityWindowSize,
            string inputColumnName = null,
            AnomalySide side = AnomalySide.TwoSided,
            ErrorFunctionUtils.ErrorFunction errorFunction = ErrorFunctionUtils.ErrorFunction.SignedDifference)
            : this(env, new SsaSpikeDetector.Arguments
                {
                    Source = inputColumnName ?? outputColumnName,
                    Name = outputColumnName,
                    Confidence = confidence,
                    PvalueHistoryLength = pvalueHistoryLength,
                    TrainingWindowSize = trainingWindowSize,
                    SeasonalWindowSize = seasonalityWindowSize,
                    Side = side,
                    ErrorFunction = errorFunction
                })
        {
        }

        public SsaSpikeEstimator(IHostEnvironment env, SsaSpikeDetector.Arguments args)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(SsaSpikeEstimator));

            _host.CheckNonEmpty(args.Name, nameof(args.Name));
            _host.CheckNonEmpty(args.Source, nameof(args.Source));

            _args = args;
        }

        public SsaSpikeDetector Fit(IDataView input)
        {
            _host.CheckValue(input, nameof(input));
            return new SsaSpikeDetector(_host, _args, input);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));

            if (!inputSchema.TryFindColumn(_args.Source, out var col))
                throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", _args.Source);
            if (col.ItemType != NumberType.R4)
                throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", _args.Source, NumberType.R4.ToString(), col.GetTypeString());

            var metadata = new List<SchemaShape.Column>() {
                new SchemaShape.Column(MetadataUtils.Kinds.SlotNames, SchemaShape.Column.VectorKind.Vector, TextType.Instance, false)
            };
            var resultDic = inputSchema.ToDictionary(x => x.Name);
            resultDic[_args.Name] = new SchemaShape.Column(
                _args.Name, SchemaShape.Column.VectorKind.Vector, NumberType.R8, false, new SchemaShape(metadata));

            return new SchemaShape(resultDic.Values);
        }
    }
}
