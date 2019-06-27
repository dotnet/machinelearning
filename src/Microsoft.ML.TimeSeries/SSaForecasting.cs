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

[assembly: LoadableClass(SsaForecasting.Summary, typeof(IDataTransform), typeof(SsaForecasting), typeof(SsaForecasting.Options), typeof(SignatureDataTransform),
    SsaForecasting.UserName, SsaForecasting.LoaderSignature, SsaForecasting.ShortName)]

[assembly: LoadableClass(SsaForecasting.Summary, typeof(IDataTransform), typeof(SsaForecasting), null, typeof(SignatureLoadDataTransform),
    SsaForecasting.UserName, SsaForecasting.LoaderSignature)]

[assembly: LoadableClass(SsaForecasting.Summary, typeof(SsaForecasting), null, typeof(SignatureLoadModel),
    SsaForecasting.UserName, SsaForecasting.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(SsaForecasting), null, typeof(SignatureLoadRowMapper),
   SsaForecasting.UserName, SsaForecasting.LoaderSignature)]

namespace Microsoft.ML.Transforms.TimeSeries
{
    /// <summary>
    /// <see cref="ITransformer"/> resulting from fitting a <see cref="SsaForecastingEstimator"/>.
    /// </summary>
    public sealed class SsaForecasting : SsaForecastingBaseWrapper, IStatefulTransformer, IForecastTransformer
    {
        internal const string Summary = "This transform forecasts using Singular Spectrum Analysis (SSA).";
        internal const string LoaderSignature = "SsaForecasting";
        internal const string UserName = "SSA Forecasting";
        internal const string ShortName = "ssafcst";

        internal sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "The name of the source column.", ShortName = "src",
                SortOrder = 1, Purpose = SpecialPurpose.ColumnName)]
            public string Source;

            [Argument(ArgumentType.Required, HelpText = "The name of the new column.",
                SortOrder = 2)]
            public string Name;

            [Argument(ArgumentType.Required, HelpText = "The name of the new column", ShortName = "cnfminname",
                SortOrder = 2)]
            public string ForecastingConfidenceIntervalMinOutputColumnName;

            [Argument(ArgumentType.Required, HelpText = "The name of the new column", ShortName = "cnfmaxnname",
                SortOrder = 2)]
            public string ForecastingConfidenceIntervalMaxOutputColumnName;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The discount factor in [0, 1]", ShortName = "disc", SortOrder = 12)]
            public float DiscountFactor = 1;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The flag determing whether the model is adaptive", ShortName = "adp", SortOrder = 14)]
            public bool IsAdaptive = false;
            public int WindowSize;
            public RankSelectionMethod RankSelectionMethod;
            public int? Rank;
            public int? MaxRank;
            public bool ShouldStablize;
            public bool ShouldMaintainInfo;
            public GrowthRatio? MaxGrowth;
            public int SeriesLength;
            public int TrainSize;
            public int Horizon;
            public float ConfidenceLevel;
        }

        private sealed class BaseArguments : SsaForecastingOptions
        {
            public BaseArguments(Options options)
            {
                Source = options.Source;
                Name = options.Name;
                ForecastingConfidenceIntervalMinOutputColumnName = options.ForecastingConfidenceIntervalMinOutputColumnName;
                ForecastingConfidenceIntervalMaxOutputColumnName = options.ForecastingConfidenceIntervalMaxOutputColumnName;
                WindowSize = options.WindowSize;
                DiscountFactor = options.DiscountFactor;
                IsAdaptive = options.IsAdaptive;
                RankSelectionMethod = options.RankSelectionMethod;
                Rank = options.Rank;
                ShouldStablize = options.ShouldStablize;
                MaxGrowth = options.MaxGrowth;
                SeriesLength = options.SeriesLength;
                TrainSize = options.TrainSize;
                Horizon = options.Horizon;
                ConfidenceLevel = options.ConfidenceLevel;
            }
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "FRCSTRNS",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(SsaForecasting).Assembly.FullName);
        }

        internal SsaForecasting(IHostEnvironment env, Options options, IDataView input)
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

            return new SsaForecasting(env, options, input).MakeDataTransform(input);
        }

        internal SsaForecasting(IHostEnvironment env, Options options)
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

            return new SsaForecasting(env, ctx).MakeDataTransform(input);
        }

        IStatefulTransformer IStatefulTransformer.Clone()
        {
            var clone = (SsaForecasting)MemberwiseClone();
            clone.InternalTransform.Model = clone.InternalTransform.Model.Clone();
            clone.InternalTransform.StateRef = (SsaForecastingBase.State)clone.InternalTransform.StateRef.Clone();
            clone.InternalTransform.StateRef.InitState(clone.InternalTransform, InternalTransform.Host);
            return clone;
        }

        // Factory method for SignatureLoadModel.
        private static SsaForecasting Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new SsaForecasting(env, ctx);
        }

        internal SsaForecasting(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, ctx, LoaderSignature)
        {
            // *** Binary format ***
            // <base>
            InternalTransform.Host.CheckDecode(InternalTransform.IsAdaptive == false);
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            InternalTransform.Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

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
    /// Detect spikes in time series using Singular Spectrum Analysis.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    /// To create this estimator, use [DetectSpikeBySsa](xref:Microsoft.ML.TimeSeriesCatalog.DetectSpikeBySsa(Microsoft.ML.TransformsCatalog,System.String,System.String,System.Int32,System.Int32,System.Int32,System.Int32,Microsoft.ML.Transforms.TimeSeries.AnomalySide,Microsoft.ML.Transforms.TimeSeries.ErrorFunction))
    ///
    /// [!include[io](~/../docs/samples/docs/api-reference/io-time-series-spike.md)]
    ///
    /// ###  Estimator Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Does this estimator need to look at the data to train its parameters? | Yes |
    /// | Input column data type | <xref:System.Single> |
    /// | Output column data type | 3-element vector of <xref:System.Double> |
    ///
    /// [!include[io](~/../docs/samples/docs/api-reference/time-series-props.md)]
    ///
    /// [!include[io](~/../docs/samples/docs/api-reference/time-series-ssa.md)]
    ///
    /// [!include[io](~/../docs/samples/docs/api-reference/time-series-pvalue.md)]
    ///
    /// Check the See Also section for links to usage examples.
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="Microsoft.ML.TimeSeriesCatalog.DetectSpikeBySsa(Microsoft.ML.TransformsCatalog,System.String,System.String,System.Int32,System.Int32,System.Int32,System.Int32,Microsoft.ML.Transforms.TimeSeries.AnomalySide,Microsoft.ML.Transforms.TimeSeries.ErrorFunction)" />
    public sealed class SsaForecastingEstimator : IEstimator<SsaForecasting>
    {
        private readonly IHost _host;
        private readonly SsaForecasting.Options _options;

        /// <summary>
        /// Create a new instance of <see cref="SsaForecastingEstimator"/>
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="discountFactor"></param>
        /// <param name="isAdaptive"></param>
        /// <param name="windowSize"></param>
        /// <param name="rankSelectionMethod"></param>
        /// <param name="rank"></param>
        /// <param name="maxRank"></param>
        /// <param name="shouldComputeForecastIntervals"></param>
        /// <param name="shouldStablize"></param>
        /// <param name="shouldMaintainInfo"></param>
        /// <param name="maxGrowth"></param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.
        /// The vector contains Alert, Raw Score, P-Value as first three values.</param>
        /// <param name="forecastingConfidenceIntervalMinOutputColumnName"></param>
        /// <param name="forecastingConfidenceIntervalMaxOutputColumnName"></param>
        /// <param name="confidenceLevel"></param>
        internal SsaForecastingEstimator(IHostEnvironment env,
            string outputColumnName,
            float discountFactor,
            bool isAdaptive,
            int windowSize,
            RankSelectionMethod rankSelectionMethod,
            int? rank,
            int? maxRank,
            bool shouldComputeForecastIntervals,
            bool shouldStablize,
            bool shouldMaintainInfo,
            GrowthRatio? maxGrowth,
            string inputColumnName = null,
            string forecastingConfidenceIntervalMinOutputColumnName = null,
            string forecastingConfidenceIntervalMaxOutputColumnName = null,
            float confidenceLevel = 0.95f)
            : this(env, new SsaForecasting.Options
            {
                Source = inputColumnName ?? outputColumnName,
                Name = outputColumnName,
                DiscountFactor = discountFactor,
                IsAdaptive = isAdaptive,
                WindowSize = windowSize,
                RankSelectionMethod = rankSelectionMethod,
                Rank = rank,
                MaxRank = maxRank,
                ShouldStablize = shouldStablize,
                ShouldMaintainInfo = shouldMaintainInfo,
                MaxGrowth = maxGrowth,
                ConfidenceLevel = confidenceLevel,
                ForecastingConfidenceIntervalMinOutputColumnName = forecastingConfidenceIntervalMinOutputColumnName,
                ForecastingConfidenceIntervalMaxOutputColumnName = forecastingConfidenceIntervalMaxOutputColumnName
            })
        {
        }

        internal SsaForecastingEstimator(IHostEnvironment env, SsaForecasting.Options options)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(SsaForecastingEstimator));

            _host.CheckNonEmpty(options.Name, nameof(options.Name));
            _host.CheckNonEmpty(options.Source, nameof(options.Source));

            _options = options;
        }

        /// <summary>
        /// Train and return a transformer.
        /// </summary>
        public SsaForecasting Fit(IDataView input)
        {
            _host.CheckValue(input, nameof(input));
            return new SsaForecasting(_host, _options, input);
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
                throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", _options.Source, "Single", col.GetTypeString());

            var metadata = new List<SchemaShape.Column>() {
                new SchemaShape.Column(AnnotationUtils.Kinds.SlotNames, SchemaShape.Column.VectorKind.Vector, TextDataViewType.Instance, false)
            };
            var resultDic = inputSchema.ToDictionary(x => x.Name);
            resultDic[_options.Name] = new SchemaShape.Column(
                _options.Name, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Single, false, new SchemaShape(metadata));

            if (!string.IsNullOrEmpty(_options.ForecastingConfidenceIntervalMaxOutputColumnName))
            {
                resultDic[_options.ForecastingConfidenceIntervalMinOutputColumnName] = new SchemaShape.Column(
                    _options.ForecastingConfidenceIntervalMinOutputColumnName, SchemaShape.Column.VectorKind.Vector,
                    NumberDataViewType.Single, false, new SchemaShape(metadata));

                resultDic[_options.ForecastingConfidenceIntervalMaxOutputColumnName] = new SchemaShape.Column(
                    _options.ForecastingConfidenceIntervalMaxOutputColumnName, SchemaShape.Column.VectorKind.Vector,
                    NumberDataViewType.Single, false, new SchemaShape(metadata));
            }

            return new SchemaShape(resultDic.Values);
        }
    }
}
