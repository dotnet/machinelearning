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
using static Microsoft.ML.Runtime.TimeSeriesProcessing.SequentialAnomalyDetectionTransformBase<System.Single, Microsoft.ML.Runtime.TimeSeriesProcessing.SsaAnomalyDetectionBase.State>;

[assembly: LoadableClass(SsaChangePointDetector.Summary, typeof(IDataTransform), typeof(SsaChangePointDetector), typeof(SsaChangePointDetector.Arguments), typeof(SignatureDataTransform),
    SsaChangePointDetector.UserName, SsaChangePointDetector.LoaderSignature, SsaChangePointDetector.ShortName)]

[assembly: LoadableClass(SsaChangePointDetector.Summary, typeof(IDataTransform), typeof(SsaChangePointDetector), null, typeof(SignatureLoadDataTransform),
    SsaChangePointDetector.UserName, SsaChangePointDetector.LoaderSignature)]

[assembly: LoadableClass(SsaChangePointDetector.Summary, typeof(SsaChangePointDetector), null, typeof(SignatureLoadModel),
    SsaChangePointDetector.UserName, SsaChangePointDetector.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(SsaChangePointDetector), null, typeof(SignatureLoadRowMapper),
   SsaChangePointDetector.UserName, SsaChangePointDetector.LoaderSignature)]

namespace Microsoft.ML.Runtime.TimeSeriesProcessing
{
    /// <summary>
    /// This class implements the change point detector transform based on Singular Spectrum modeling of the time-series.
    /// For the details of the Singular Spectrum Analysis (SSA), refer to http://arxiv.org/pdf/1206.6910.pdf.
    /// </summary>
    public sealed class SsaChangePointDetector : SsaAnomalyDetectionBase
    {
        internal const string Summary = "This transform detects the change-points in a seasonal time-series using Singular Spectrum Analysis (SSA).";
        public const string LoaderSignature = "SsaChangePointDetector";
        public const string UserName = "SSA Change Point Detection";
        public const string ShortName = "chgpnt";

        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "The name of the source column.", ShortName = "src",
                SortOrder = 1, Purpose = SpecialPurpose.ColumnName)]
            public string Source;

            [Argument(ArgumentType.Required, HelpText = "The name of the new column.",
                SortOrder = 2)]
            public string Name;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The length of the sliding window on p-values for computing the martingale score.", ShortName = "wnd",
                SortOrder = 102)]
            public int ChangeHistoryLength = 20;

            [Argument(ArgumentType.Required, HelpText = "The number of points from the beginning of the sequence used for training.",
                ShortName = "twnd", SortOrder = 3)]
            public int TrainingWindowSize = 100;

            [Argument(ArgumentType.Required, HelpText = "The confidence for change point detection in the range [0, 100].",
                ShortName = "cnf", SortOrder = 4)]
            public double Confidence = 95;

            [Argument(ArgumentType.Required, HelpText = "An upper bound on the largest relevant seasonality in the input time-series.", ShortName = "swnd", SortOrder = 5)]
            public int SeasonalWindowSize = 10;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The function used to compute the error between the expected and the observed value.", ShortName = "err", SortOrder = 103)]
            public ErrorFunctionUtils.ErrorFunction ErrorFunction = ErrorFunctionUtils.ErrorFunction.SignedDifference;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The martingale used for scoring.", ShortName = "mart", SortOrder = 104)]
            public MartingaleType Martingale = SequentialAnomalyDetectionTransformBase<float, State>.MartingaleType.Power;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The epsilon parameter for the Power martingale.",
                ShortName = "eps", SortOrder = 105)]
            public double PowerMartingaleEpsilon = 0.1;
        }

        private sealed class BaseArguments : SsaArguments
        {
            public BaseArguments(Arguments args)
            {
                Source = args.Source;
                Name = args.Name;
                Side = SequentialAnomalyDetectionTransformBase<float, State>.AnomalySide.TwoSided;
                WindowSize = args.ChangeHistoryLength;
                InitialWindowSize = args.TrainingWindowSize;
                SeasonalWindowSize = args.SeasonalWindowSize;
                Martingale = args.Martingale;
                PowerMartingaleEpsilon = args.PowerMartingaleEpsilon;
                AlertOn = SequentialAnomalyDetectionTransformBase<float, State>.AlertingScore.MartingaleScore;
                DiscountFactor = 1;
                IsAdaptive = false;
                ErrorFunction = args.ErrorFunction;
            }
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(modelSignature: "SCHGTRNS",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(SsaChangePointDetector).Assembly.FullName);
        }

        internal SsaChangePointDetector(IHostEnvironment env, Arguments args, IDataView input)
            : this(env, args)
        {
            Model.Train(new RoleMappedData(input, null, InputColumnName));
        }

        // Factory method for SignatureDataTransform.
        private static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));

            return new SsaChangePointDetector(env, args, input).MakeDataTransform(input);
        }

        internal SsaChangePointDetector(IHostEnvironment env, Arguments args)
            : base(new BaseArguments(args), LoaderSignature, env)
        {
            switch (Martingale)
            {
                case MartingaleType.None:
                    AlertThreshold = Double.MaxValue;
                    break;
                case MartingaleType.Power:
                    AlertThreshold = Math.Exp(WindowSize * LogPowerMartigaleBettingFunc(1 - args.Confidence / 100, PowerMartingaleEpsilon));
                    break;
                case MartingaleType.Mixture:
                    AlertThreshold = Math.Exp(WindowSize * LogMixtureMartigaleBettingFunc(1 - args.Confidence / 100));
                    break;
                default:
                    Host.Assert(!Enum.IsDefined(typeof(MartingaleType), Martingale));
                    throw Host.ExceptUserArg(nameof(args.Martingale), "Value not defined.");
            }
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            env.CheckValue(input, nameof(input));

            return new SsaChangePointDetector(env, ctx).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadModel.
        private static SsaChangePointDetector Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new SsaChangePointDetector(env, ctx);
        }

        internal SsaChangePointDetector(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, ctx, LoaderSignature)
        {
            // *** Binary format ***
            // <base>

            Host.CheckDecode(ThresholdScore == AlertingScore.MartingaleScore);
            Host.CheckDecode(Side == AnomalySide.TwoSided);
            Host.CheckDecode(DiscountFactor == 1);
            Host.CheckDecode(IsAdaptive == false);
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            Host.Assert(ThresholdScore == AlertingScore.MartingaleScore);
            Host.Assert(Side == AnomalySide.TwoSided);
            Host.Assert(DiscountFactor == 1);
            Host.Assert(IsAdaptive == false);

            // *** Binary format ***
            // <base>

            base.Save(ctx);
        }

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);
    }

    /// <summary>
    /// Estimator for <see cref="SsaChangePointDetector"/>
    /// </summary>
    public sealed class SsaChangePointEstimator : IEstimator<SsaChangePointDetector>
    {
        private readonly IHost _host;
        private readonly SsaChangePointDetector.Arguments _args;

        /// <summary>
        /// Create a new instance of <see cref="SsaChangePointEstimator"/>
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="inputColumn">Name of the input column.</param>
        /// <param name="outputColumn">Name of the output column.</param>
        /// <param name="confidence">The confidence for change point detection in the range [0, 100].</param>
        /// <param name="trainingWindowSize">The number of points from the beginning of the sequence used for training.</param>
        /// <param name="changeHistoryLength">The size of the sliding window for computing the p-value.</param>
        /// <param name="seasonalityWindowSize">An upper bound on the largest relevant seasonality in the input time-series.</param>
        /// <param name="errorFunction">The function used to compute the error between the expected and the observed value.</param>
        /// <param name="martingale">The martingale used for scoring.</param>
        /// <param name="eps">The epsilon parameter for the Power martingale.</param>
        /// <p>Example code can be found by searching for <i>SsaChangePointDetector</i> in <a href='https://github.com/dotnet/machinelearning'>ML.NET.</a></p>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[MF](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Timeseries.cs "Timeseries examples for changepoint detection.")]
        /// ]]>
        /// </format>
        public SsaChangePointEstimator(IHostEnvironment env, string inputColumn, string outputColumn,
            int confidence, int changeHistoryLength, int trainingWindowSize, int seasonalityWindowSize,
            ErrorFunctionUtils.ErrorFunction errorFunction = ErrorFunctionUtils.ErrorFunction.SignedDifference,
            MartingaleType martingale = MartingaleType.Power, double eps = 0.1)
            : this(env, new SsaChangePointDetector.Arguments
                {
                    Name = outputColumn,
                    Source = inputColumn,
                    Confidence = confidence,
                    ChangeHistoryLength = changeHistoryLength,
                    TrainingWindowSize = trainingWindowSize,
                    SeasonalWindowSize = seasonalityWindowSize,
                    Martingale = martingale,
                    PowerMartingaleEpsilon = eps,
                    ErrorFunction = errorFunction
                })
        {
        }

        public SsaChangePointEstimator(IHostEnvironment env, SsaChangePointDetector.Arguments args)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(SsaChangePointEstimator));

            _host.CheckNonEmpty(args.Name, nameof(args.Name));
            _host.CheckNonEmpty(args.Source, nameof(args.Source));

            _args = args;
        }

        public SsaChangePointDetector Fit(IDataView input)
        {
            _host.CheckValue(input, nameof(input));
            return new SsaChangePointDetector(_host, _args, input);
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
            var resultDic = inputSchema.Columns.ToDictionary(x => x.Name);
            resultDic[_args.Name] = new SchemaShape.Column(
                _args.Name, SchemaShape.Column.VectorKind.Vector, NumberType.R8, false, new SchemaShape(metadata));

            return new SchemaShape(resultDic.Values);
        }
    }
}
