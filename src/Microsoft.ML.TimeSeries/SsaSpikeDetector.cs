// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.TimeSeries;

[assembly: LoadableClass(SsaSpikeDetector.Summary, typeof(IDataTransform), typeof(SsaSpikeDetector), typeof(SsaSpikeDetector.Options), typeof(SignatureDataTransform),
    SsaSpikeDetector.UserName, SsaSpikeDetector.LoaderSignature, SsaSpikeDetector.ShortName)]

[assembly: LoadableClass(SsaSpikeDetector.Summary, typeof(IDataTransform), typeof(SsaSpikeDetector), null, typeof(SignatureLoadDataTransform),
    SsaSpikeDetector.UserName, SsaSpikeDetector.LoaderSignature)]

[assembly: LoadableClass(SsaSpikeDetector.Summary, typeof(SsaSpikeDetector), null, typeof(SignatureLoadModel),
    SsaSpikeDetector.UserName, SsaSpikeDetector.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(SsaSpikeDetector), null, typeof(SignatureLoadRowMapper),
   SsaSpikeDetector.UserName, SsaSpikeDetector.LoaderSignature)]

namespace Microsoft.ML.Transforms.TimeSeries
{
    /// <summary>
    /// <see cref="ITransformer"/> produced by fitting the <see cref="IDataView"/> to an <see cref="SsaSpikeEstimator" />.
    /// </summary>
    public sealed class SsaSpikeDetector : SsaAnomalyDetectionBaseWrapper, IStatefulTransformer
    {
        internal const string Summary = "This transform detects the spikes in a seasonal time-series using Singular Spectrum Analysis (SSA).";
        internal const string LoaderSignature = "SsaSpikeDetector";
        internal const string UserName = "SSA Spike Detection";
        internal const string ShortName = "spike";

        internal sealed class Options : TransformInputBase
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
            public ErrorFunction ErrorFunction = ErrorFunction.SignedDifference;
        }

        private sealed class BaseArguments : SsaOptions
        {
            public BaseArguments(Options options)
            {
                Source = options.Source;
                Name = options.Name;
                Side = options.Side;
                WindowSize = options.PvalueHistoryLength;
                InitialWindowSize = options.TrainingWindowSize;
                SeasonalWindowSize = options.SeasonalWindowSize;
                AlertThreshold = 1 - options.Confidence / 100;
                AlertOn = AlertingScore.PValueScore;
                DiscountFactor = 1;
                IsAdaptive = false;
                ErrorFunction = options.ErrorFunction;
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

        internal SsaSpikeDetector(IHostEnvironment env, Options options, IDataView input)
            : base(new BaseArguments(options), LoaderSignature, env)
        {
            InternalTransform.Model.Train(new RoleMappedData(input, null, InternalTransform.InputColumnName));
        }

        // Factory method for SignatureDataTransform.
        private static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));

            return new SsaSpikeDetector(env, options, input).MakeDataTransform(input);
        }

        internal SsaSpikeDetector(IHostEnvironment env, Options options)
            : base(new BaseArguments(options), LoaderSignature, env)
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

        IStatefulTransformer IStatefulTransformer.Clone()
        {
            var clone = (SsaSpikeDetector)MemberwiseClone();
            clone.InternalTransform.Model = clone.InternalTransform.Model.Clone();
            clone.InternalTransform.StateRef = (SsaAnomalyDetectionBase.State)clone.InternalTransform.StateRef.Clone();
            clone.InternalTransform.StateRef.InitState(clone.InternalTransform, InternalTransform.Host);
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

            InternalTransform.Host.CheckDecode(InternalTransform.ThresholdScore == AlertingScore.PValueScore);
            InternalTransform.Host.CheckDecode(InternalTransform.DiscountFactor == 1);
            InternalTransform.Host.CheckDecode(InternalTransform.IsAdaptive == false);
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            InternalTransform.Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            InternalTransform.Host.Assert(InternalTransform.ThresholdScore == AlertingScore.PValueScore);
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
    /// The <see cref="IEstimator{ITransformer}"/> for detecting a signal spike through <a href="http://arxiv.org/pdf/1206.6910.pdf">Singular Spectrum Analysis (SSA)</a> of time series.
    /// </summary>
    public sealed class SsaSpikeEstimator : IEstimator<SsaSpikeDetector>
    {
        private readonly IHost _host;
        private readonly SsaSpikeDetector.Options _options;

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
        internal SsaSpikeEstimator(IHostEnvironment env,
            string outputColumnName,
            int confidence,
            int pvalueHistoryLength,
            int trainingWindowSize,
            int seasonalityWindowSize,
            string inputColumnName = null,
            AnomalySide side = AnomalySide.TwoSided,
            ErrorFunction errorFunction = ErrorFunction.SignedDifference)
            : this(env, new SsaSpikeDetector.Options
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

        internal SsaSpikeEstimator(IHostEnvironment env, SsaSpikeDetector.Options options)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(SsaSpikeEstimator));

            _host.CheckNonEmpty(options.Name, nameof(options.Name));
            _host.CheckNonEmpty(options.Source, nameof(options.Source));

            _options = options;
        }

        /// <summary>
        /// Train and return a transformer.
        /// </summary>
        public SsaSpikeDetector Fit(IDataView input)
        {
            _host.CheckValue(input, nameof(input));
            return new SsaSpikeDetector(_host, _options, input);
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
