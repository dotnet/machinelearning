// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.TimeSeries;

[assembly: LoadableClass(SsaChangePointDetector.Summary, typeof(IDataTransform), typeof(SsaChangePointDetector), typeof(SsaChangePointDetector.Options), typeof(SignatureDataTransform),
    SsaChangePointDetector.UserName, SsaChangePointDetector.LoaderSignature, SsaChangePointDetector.ShortName)]

[assembly: LoadableClass(SsaChangePointDetector.Summary, typeof(IDataTransform), typeof(SsaChangePointDetector), null, typeof(SignatureLoadDataTransform),
    SsaChangePointDetector.UserName, SsaChangePointDetector.LoaderSignature)]

[assembly: LoadableClass(SsaChangePointDetector.Summary, typeof(SsaChangePointDetector), null, typeof(SignatureLoadModel),
    SsaChangePointDetector.UserName, SsaChangePointDetector.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(SsaChangePointDetector), null, typeof(SignatureLoadRowMapper),
   SsaChangePointDetector.UserName, SsaChangePointDetector.LoaderSignature)]

namespace Microsoft.ML.Transforms.TimeSeries
{
    /// <summary>
    /// <see cref="ITransformer"/> produced by fitting the <see cref="IDataView"/> to an <see cref="SsaChangePointEstimator" /> .
    /// </summary>
    public sealed class SsaChangePointDetector : SsaAnomalyDetectionBaseWrapper, IStatefulTransformer
    {
        internal const string Summary = "This transform detects the change-points in a seasonal time-series using Singular Spectrum Analysis (SSA).";
        internal const string LoaderSignature = "SsaChangePointDetector";
        internal const string UserName = "SSA Change Point Detection";
        internal const string ShortName = "chgpnt";

        internal sealed class Options : TransformInputBase
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
            public ErrorFunction ErrorFunction = ErrorFunction.SignedDifference;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The martingale used for scoring.", ShortName = "mart", SortOrder = 104)]
            public MartingaleType Martingale = MartingaleType.Power;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The epsilon parameter for the Power martingale.",
                ShortName = "eps", SortOrder = 105)]
            public double PowerMartingaleEpsilon = 0.1;
        }

        private sealed class BaseArguments : SsaOptions
        {
            public BaseArguments(Options options)
            {
                Source = options.Source;
                Name = options.Name;
                Side = AnomalySide.TwoSided;
                WindowSize = options.ChangeHistoryLength;
                InitialWindowSize = options.TrainingWindowSize;
                SeasonalWindowSize = options.SeasonalWindowSize;
                Martingale = options.Martingale;
                PowerMartingaleEpsilon = options.PowerMartingaleEpsilon;
                AlertOn = AlertingScore.MartingaleScore;
                DiscountFactor = 1;
                IsAdaptive = false;
                ErrorFunction = options.ErrorFunction;
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

        internal SsaChangePointDetector(IHostEnvironment env, Options options, IDataView input)
            : this(env, options)
        {
            InternalTransform.Model.Train(new RoleMappedData(input, null, InternalTransform.InputColumnName));
        }

        // Factory method for SignatureDataTransform.
        private static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));

            return new SsaChangePointDetector(env, options, input).MakeDataTransform(input);
        }

        IStatefulTransformer IStatefulTransformer.Clone()
        {
            var clone = (SsaChangePointDetector)MemberwiseClone();
            clone.InternalTransform.Model = clone.InternalTransform.Model.Clone();
            clone.InternalTransform.StateRef = (SsaAnomalyDetectionBase.State)clone.InternalTransform.StateRef.Clone();
            clone.InternalTransform.StateRef.InitState(clone.InternalTransform, InternalTransform.Host);
            return clone;
        }

        internal SsaChangePointDetector(IHostEnvironment env, Options options)
            : base(new BaseArguments(options), LoaderSignature, env)
        {
            switch (InternalTransform.Martingale)
            {
                case MartingaleType.None:
                    InternalTransform.AlertThreshold = Double.MaxValue;
                    break;
                case MartingaleType.Power:
                    InternalTransform.AlertThreshold = Math.Exp(InternalTransform.WindowSize * InternalTransform.LogPowerMartigaleBettingFunc(1 - options.Confidence / 100, InternalTransform.PowerMartingaleEpsilon));
                    break;
                case MartingaleType.Mixture:
                    InternalTransform.AlertThreshold = Math.Exp(InternalTransform.WindowSize * InternalTransform.LogMixtureMartigaleBettingFunc(1 - options.Confidence / 100));
                    break;
                default:
                    InternalTransform.Host.Assert(!Enum.IsDefined(typeof(MartingaleType), InternalTransform.Martingale));
                    throw InternalTransform.Host.ExceptUserArg(nameof(options.Martingale), "Value not defined.");
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

            InternalTransform.Host.CheckDecode(InternalTransform.ThresholdScore == AlertingScore.MartingaleScore);
            InternalTransform.Host.CheckDecode(InternalTransform.Side == AnomalySide.TwoSided);
            InternalTransform.Host.CheckDecode(InternalTransform.DiscountFactor == 1);
            InternalTransform.Host.CheckDecode(InternalTransform.IsAdaptive == false);
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            InternalTransform.Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            InternalTransform.Host.Assert(InternalTransform.ThresholdScore == AlertingScore.MartingaleScore);
            InternalTransform.Host.Assert(InternalTransform.Side == AnomalySide.TwoSided);
            InternalTransform.Host.Assert(InternalTransform.DiscountFactor == 1);
            InternalTransform.Host.Assert(InternalTransform.IsAdaptive == false);

            // *** Binary format ***
            // <base>

            base.SaveModel(ctx);
        }

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);
    }

    /// <summary>
    /// The <see cref="IEstimator{ITransformer}"/> for detecting a signal change through <a href="http://arxiv.org/pdf/1206.6910.pdf">Singular Spectrum Analysis (SSA)</a> of time series.
    /// </summary>
    public sealed class SsaChangePointEstimator : IEstimator<SsaChangePointDetector>
    {
        private readonly IHost _host;
        private readonly SsaChangePointDetector.Options _options;

        /// <summary>
        /// Create a new instance of <see cref="SsaChangePointEstimator"/>
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// Column is a vector of type double and size 4. The vector contains Alert, Raw Score, P-Value and Martingale score as first four values.</param>
        /// <param name="confidence">The confidence for change point detection in the range [0, 100].</param>
        /// <param name="trainingWindowSize">The number of points from the beginning of the sequence used for training.</param>
        /// <param name="changeHistoryLength">The size of the sliding window for computing the p-value.</param>
        /// <param name="seasonalityWindowSize">An upper bound on the largest relevant seasonality in the input time-series.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="errorFunction">The function used to compute the error between the expected and the observed value.</param>
        /// <param name="martingale">The martingale used for scoring.</param>
        /// <param name="eps">The epsilon parameter for the Power martingale.</param>
        internal SsaChangePointEstimator(IHostEnvironment env, string outputColumnName,
            int confidence,
            int changeHistoryLength,
            int trainingWindowSize,
            int seasonalityWindowSize,
            string inputColumnName = null,
            ErrorFunction errorFunction = ErrorFunction.SignedDifference,
            MartingaleType martingale = MartingaleType.Power,
            double eps = 0.1)
            : this(env, new SsaChangePointDetector.Options
                {
                    Name = outputColumnName,
                    Source = inputColumnName ?? outputColumnName,
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

        internal SsaChangePointEstimator(IHostEnvironment env, SsaChangePointDetector.Options options)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(SsaChangePointEstimator));

            _host.CheckNonEmpty(options.Name, nameof(options.Name));
            _host.CheckNonEmpty(options.Source, nameof(options.Source));

            _options = options;
        }

        /// <summary>
        /// Train and return a transformer.
        /// </summary>
        public SsaChangePointDetector Fit(IDataView input)
        {
            _host.CheckValue(input, nameof(input));
            return new SsaChangePointDetector(_host, _options, input);
        }

        /// <summary>
        /// Schema propagation for transformers.
        /// Returns the output schema of the data, if the input schema is like the one provided.
        /// </summary>
        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));

            if (!inputSchema.TryFindColumn(_options.Source, out var col))
                throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", _options.Source);
            if (col.ItemType != NumberDataViewType.Single)
                throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", _options.Source, "float", col.GetTypeString());

            var metadata = new List<SchemaShape.Column>() {
                new SchemaShape.Column(AnnotationUtils.Kinds.SlotNames, SchemaShape.Column.VectorKind.Vector, TextDataViewType.Instance, false)
            };
            var resultDic = inputSchema.ToDictionary(x => x.Name);
            resultDic[_options.Name] = new SchemaShape.Column(
                _options.Name, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Double, false, new SchemaShape(metadata));

            return new SchemaShape(resultDic.Values);
        }
    }
}
