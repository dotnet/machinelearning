// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.TimeSeries;

[assembly: LoadableClass(SsaForecastingTransformer.Summary, typeof(IDataTransform), typeof(SsaForecastingTransformer), typeof(SsaForecastingTransformer.Options), typeof(SignatureDataTransform),
    SsaForecastingTransformer.UserName, SsaForecastingTransformer.LoaderSignature, SsaForecastingTransformer.ShortName)]

[assembly: LoadableClass(SsaForecastingTransformer.Summary, typeof(IDataTransform), typeof(SsaForecastingTransformer), null, typeof(SignatureLoadDataTransform),
    SsaForecastingTransformer.UserName, SsaForecastingTransformer.LoaderSignature)]

[assembly: LoadableClass(SsaForecastingTransformer.Summary, typeof(SsaForecastingTransformer), null, typeof(SignatureLoadModel),
    SsaForecastingTransformer.UserName, SsaForecastingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(SsaForecastingTransformer), null, typeof(SignatureLoadRowMapper),
   SsaForecastingTransformer.UserName, SsaForecastingTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms.TimeSeries
{
    /// <summary>
    /// <see cref="ITransformer"/> resulting from fitting a <see cref="SsaForecastingEstimator"/>.
    /// </summary>
    public sealed class SsaForecastingTransformer : SsaForecastingBaseWrapper, IStatefulTransformer
    {
        internal const string Summary = "This transform forecasts using Singular Spectrum Analysis (SSA).";
        internal const string LoaderSignature = "SsaForecasting";
        internal const string UserName = "SSA Forecasting";
        internal const string ShortName = "ssafcst";

        internal sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "The name of the source column.", ShortName = "src", SortOrder = 1, Purpose = SpecialPurpose.ColumnName)]
            public string Source;

            [Argument(ArgumentType.Required, HelpText = "The name of the new column.", SortOrder = 2)]
            public string Name;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The name of the confidence interval lower bound column.", ShortName = "cnfminname", SortOrder = 3)]
            public string ConfidenceLowerBoundColumn;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The name of the confidence interval upper bound column.", ShortName = "cnfmaxnname", SortOrder = 3)]
            public string ConfidenceUpperBoundColumn;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The discount factor in [0,1] used for online updates.", ShortName = "disc", SortOrder = 5)]
            public float DiscountFactor = 1;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The flag determing whether the model is adaptive", ShortName = "adp", SortOrder = 6)]
            public bool IsAdaptive = false;

            [Argument(ArgumentType.Required, HelpText = "The length of the window on the series for building the trajectory matrix (parameter L).", SortOrder = 2)]
            public int WindowSize;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The rank selection method.", SortOrder = 3)]
            public RankSelectionMethod RankSelectionMethod = RankSelectionMethod.Exact;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The desired rank of the subspace used for SSA projection (parameter r). This parameter should be in the range in [1, windowSize]. " +
                "If set to null, the rank is automatically determined based on prediction error minimization.", SortOrder = 3)]
            public int? Rank = null;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The maximum rank considered during the rank selection process. If not provided (i.e. set to null), it is set to windowSize - 1.", SortOrder = 3)]
            public int? MaxRank = null;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The flag determining whether the model should be stabilized.", SortOrder = 3)]
            public bool ShouldStabilize = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The flag determining whether the meta information for the model needs to be maintained.", SortOrder = 3)]
            public bool ShouldMaintainInfo = false;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The maximum growth on the exponential trend.", SortOrder = 3)]
            public GrowthRatio? MaxGrowth = null;

            [Argument(ArgumentType.Required, HelpText = "The length of series that is kept in buffer for modeling (parameter N).", SortOrder = 2)]
            public int SeriesLength;

            [Argument(ArgumentType.Required, HelpText = "The length of series from the beginning used for training.", SortOrder = 2)]
            public int TrainSize;

            [Argument(ArgumentType.Required, HelpText = "The number of values to forecast.", SortOrder = 2)]
            public int Horizon;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The confidence level in [0, 1) for forecasting.", SortOrder = 2)]
            public float ConfidenceLevel = 0.95f;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Set this to true horizon will change at prediction time.", SortOrder = 2)]
            public bool VariableHorizon;
        }

        private sealed class BaseArguments : SsaForecastingOptions
        {
            public BaseArguments(Options options)
            {
                Source = options.Source;
                Name = options.Name;
                ConfidenceLowerBoundColumn = options.ConfidenceLowerBoundColumn;
                ConfidenceUpperBoundColumn = options.ConfidenceUpperBoundColumn;
                WindowSize = options.WindowSize;
                DiscountFactor = options.DiscountFactor;
                IsAdaptive = options.IsAdaptive;
                RankSelectionMethod = options.RankSelectionMethod;
                Rank = options.Rank;
                ShouldStablize = options.ShouldStabilize;
                MaxGrowth = options.MaxGrowth;
                SeriesLength = options.SeriesLength;
                TrainSize = options.TrainSize;
                Horizon = options.Horizon;
                ConfidenceLevel = options.ConfidenceLevel;
                VariableHorizon = options.VariableHorizon;
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
                loaderAssemblyName: typeof(SsaForecastingTransformer).Assembly.FullName);
        }

        internal SsaForecastingTransformer(IHostEnvironment env, Options options, IDataView input)
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

            return new SsaForecastingTransformer(env, options, input).MakeDataTransform(input);
        }

        internal SsaForecastingTransformer(IHostEnvironment env, Options options)
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

            return new SsaForecastingTransformer(env, ctx).MakeDataTransform(input);
        }

        IStatefulTransformer IStatefulTransformer.Clone()
        {
            var clone = (SsaForecastingTransformer)MemberwiseClone();
            clone.InternalTransform.Model = clone.InternalTransform.Model.Clone();
            clone.InternalTransform.StateRef = (SsaForecastingBase.State)clone.InternalTransform.StateRef.Clone();
            clone.InternalTransform.StateRef.InitState(clone.InternalTransform, InternalTransform.Host);
            return clone;
        }

        // Factory method for SignatureLoadModel.
        internal static SsaForecastingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new SsaForecastingTransformer(env, ctx);
        }

        private SsaForecastingTransformer(IHostEnvironment env, ModelLoadContext ctx)
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
    /// Forecasts using Singular Spectrum Analysis.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    /// To create this estimator, use [ForecastBySsa](xref:Microsoft.ML.TimeSeriesCatalog.ForecastBySsa(Microsoft.ML.ForecastingCatalog,System.String,System.String,System.Int32,System.Int32,System.Int32,System.Int32,System.Boolean,System.Single,Microsoft.ML.Transforms.TimeSeries.RankSelectionMethod,System.Int32?,System.Int32?,System.Boolean,System.Boolean,Microsoft.ML.Transforms.TimeSeries.GrowthRatio?,System.String,System.String,System.Single,System.Boolean))
    ///
    /// [!include[io](~/../docs/samples/docs/api-reference/io-time-series-ssa-forecast.md)]
    ///
    /// ###  Estimator Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Does this estimator need to look at the data to train its parameters? | Yes |
    /// | Input column data type | <xref:System.Single> |
    /// | Output column data type | Vector of <xref:System.Single> |
    /// | Exportable to ONNX | No |
    ///
    /// [!include[io](~/../docs/samples/docs/api-reference/time-series-props.md)]
    ///
    /// [!include[io](~/../docs/samples/docs/api-reference/time-series-ssa.md)]
    ///
    /// Check the See Also section for links to usage examples.
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="Microsoft.ML.TimeSeriesCatalog.ForecastBySsa(Microsoft.ML.ForecastingCatalog,System.String,System.String,System.Int32,System.Int32,System.Int32,System.Int32,System.Boolean,System.Single,Microsoft.ML.Transforms.TimeSeries.RankSelectionMethod,System.Int32?,System.Int32?,System.Boolean,System.Boolean,Microsoft.ML.Transforms.TimeSeries.GrowthRatio?,System.String,System.String,System.Single,System.Boolean)" />
    public sealed class SsaForecastingEstimator : IEstimator<SsaForecastingTransformer>
    {
        private readonly IHost _host;
        private readonly SsaForecastingTransformer.Options _options;

        /// <summary>
        /// Create a new instance of <see cref="SsaForecastingEstimator"/>
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.
        /// The vector contains Alert, Raw Score, P-Value as first three values.</param>
        /// <param name="windowSize">The length of the window on the series for building the trajectory matrix (parameter L).</param>
        /// <param name="seriesLength">The length of series that is kept in buffer for modeling (parameter N).</param>
        /// <param name="trainSize">The length of series from the beginning used for training.</param>
        /// <param name="horizon">The number of values to forecast.</param>
        /// <param name="isAdaptive">The flag determing whether the model is adaptive.</param>
        /// <param name="discountFactor">The discount factor in [0,1] used for online updates.</param>
        /// <param name="rankSelectionMethod">The rank selection method.</param>
        /// <param name="rank">The desired rank of the subspace used for SSA projection (parameter r). This parameter should be in the range in [1, windowSize].
        /// If set to null, the rank is automatically determined based on prediction error minimization.</param>
        /// <param name="maxRank">The maximum rank considered during the rank selection process. If not provided (i.e. set to null), it is set to windowSize - 1.</param>
        /// <param name="shouldStabilize">The flag determining whether the model should be stabilized.</param>
        /// <param name="shouldMaintainInfo">The flag determining whether the meta information for the model needs to be maintained.</param>
        /// <param name="maxGrowth">The maximum growth on the exponential trend.</param>
        /// <param name="confidenceLowerBoundColumn">The name of the confidence interval lower bound column. If not specified then confidence intervals will not be calculated.</param>
        /// <param name="confidenceUpperBoundColumn">The name of the confidence interval upper bound column. If not specified then confidence intervals will not be calculated.</param>
        /// <param name="confidenceLevel">The confidence level for forecasting.</param>
        /// <param name="variableHorizon">Set this to true if horizon will change after training.</param>
        internal SsaForecastingEstimator(IHostEnvironment env,
            string outputColumnName,
            string inputColumnName,
            int windowSize,
            int seriesLength,
            int trainSize,
            int horizon,
            bool isAdaptive = false,
            float discountFactor = 1,
            RankSelectionMethod rankSelectionMethod = RankSelectionMethod.Exact,
            int? rank = null,
            int? maxRank = null,
            bool shouldStabilize = true,
            bool shouldMaintainInfo = false,
            GrowthRatio? maxGrowth = null,
            string confidenceLowerBoundColumn = null,
            string confidenceUpperBoundColumn = null,
            float confidenceLevel = 0.95f,
            bool variableHorizon = false)
            : this(env, new SsaForecastingTransformer.Options
            {
                Source = inputColumnName ?? outputColumnName,
                Name = outputColumnName,
                DiscountFactor = discountFactor,
                IsAdaptive = isAdaptive,
                WindowSize = windowSize,
                RankSelectionMethod = rankSelectionMethod,
                Rank = rank,
                MaxRank = maxRank,
                ShouldStabilize = shouldStabilize,
                ShouldMaintainInfo = shouldMaintainInfo,
                MaxGrowth = maxGrowth,
                ConfidenceLevel = confidenceLevel,
                ConfidenceLowerBoundColumn = confidenceLowerBoundColumn,
                ConfidenceUpperBoundColumn = confidenceUpperBoundColumn,
                SeriesLength = seriesLength,
                TrainSize = trainSize,
                VariableHorizon = variableHorizon,
                Horizon = horizon
            })
        {
        }

        internal SsaForecastingEstimator(IHostEnvironment env, SsaForecastingTransformer.Options options)
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
        public SsaForecastingTransformer Fit(IDataView input)
        {
            _host.CheckValue(input, nameof(input));
            return new SsaForecastingTransformer(_host, _options, input);
        }

        /// <summary>
        /// Schema propagation for transformers.
        /// Returns the output schema of the data, if the input schema is like the one provided.
        /// Creates three output columns if confidence intervals are requested otherwise
        /// just one.
        /// </summary>
        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));

            if (!inputSchema.TryFindColumn(_options.Source, out var col))
                throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", _options.Source);
            if (col.ItemType != NumberDataViewType.Single)
                throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", _options.Source, "Single", col.GetTypeString());

            var resultDic = inputSchema.ToDictionary(x => x.Name);
            resultDic[_options.Name] = new SchemaShape.Column(
                _options.Name, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Single, false);

            if (!string.IsNullOrEmpty(_options.ConfidenceUpperBoundColumn))
            {
                resultDic[_options.ConfidenceLowerBoundColumn] = new SchemaShape.Column(
                    _options.ConfidenceLowerBoundColumn, SchemaShape.Column.VectorKind.Vector,
                    NumberDataViewType.Single, false);

                resultDic[_options.ConfidenceUpperBoundColumn] = new SchemaShape.Column(
                    _options.ConfidenceUpperBoundColumn, SchemaShape.Column.VectorKind.Vector,
                    NumberDataViewType.Single, false);
            }

            return new SchemaShape(resultDic.Values);
        }
    }
}
