// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;

[assembly: LoadableClass(FeatureContributionCalculatingTransformer.Summary, typeof(FeatureContributionCalculatingTransformer), null, typeof(SignatureLoadModel),
    FeatureContributionCalculatingTransformer.FriendlyName, FeatureContributionCalculatingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(FeatureContributionCalculatingTransformer), null, typeof(SignatureLoadRowMapper),
   FeatureContributionCalculatingTransformer.FriendlyName, FeatureContributionCalculatingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(void), typeof(FeatureContributionEntryPoint), null, typeof(SignatureEntryPointModule), FeatureContributionCalculatingTransformer.LoaderSignature)]

namespace Microsoft.ML.Data
{
    /// <summary>
    /// The FeatureContributionCalculationTransformer computes model-specific contribution scores for each feature.
    /// See the list of currently supported models below.
    /// </summary>
    /// <remarks>
    /// Feature Contribution Calculation is currently supported for the following models:
    ///     Regression:
    ///         OrdinaryLeastSquares, StochasticDualCoordinateAscent (SDCA), OnlineGradientDescent, PoissonRegression,
    ///         GeneralizedAdditiveModels (GAM), LightGbm, FastTree, FastForest, FastTreeTweedie
    ///     Binary Classification:
    ///         AveragedPerceptron, LinearSupportVectorMachines, LogisticRegression, StochasticDualCoordinateAscent (SDCA),
    ///         StochasticGradientDescent (SGD), SymbolicStochasticGradientDescent, GeneralizedAdditiveModels (GAM),
    ///         FastForest, FastTree, LightGbm
    ///     Ranking:
    ///         FastTree, LightGbm
    ///
    /// See the sample below for an example of how to compute feature importance using the FeatureContributionCalculatingTransformer.
    /// </remarks>
    /// <example>
    /// <format type="text/markdown">
    /// <![CDATA[
    /// [!code-csharp[FCT](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/FeatureContributionCalculationTransform.cs)]
    /// ]]>
    /// </format>
    /// </example>
    public sealed class FeatureContributionCalculatingTransformer : OneToOneTransformerBase
    {
        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "The predictor model to apply to data", SortOrder = 1)]
            public PredictorModel PredictorModel;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Name of feature column", SortOrder = 2)]
            public string FeatureColumn = DefaultColumnNames.Features;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of top contributions", SortOrder = 3)]
            public int Top = FeatureContributionCalculatingEstimator.Defaults.Top;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of bottom contributions", SortOrder = 4)]
            public int Bottom = FeatureContributionCalculatingEstimator.Defaults.Bottom;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether or not output of Features contribution should be normalized", ShortName = "norm", SortOrder = 5)]
            public bool Normalize = FeatureContributionCalculatingEstimator.Defaults.Normalize;
        }

        // Apparently, loader signature is limited in length to 24 characters.
        internal const string Summary = "For each data point, calculates the contribution of individual features to the model prediction.";
        internal const string FriendlyName = "Feature Contribution Calculation";
        internal const string LoaderSignature = "FeatureContribution";

        public readonly int Top;
        public readonly int Bottom;
        public readonly bool Normalize;

        private readonly IFeatureContributionMapper _predictor;

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "FCC TRAN",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(FeatureContributionCalculatingTransformer).Assembly.FullName);
        }

        /// <summary>
        /// Feature Contribution Calculation computes model-specific contribution scores for each feature.
        /// Note that this functionality is not supported by all the models. See <see cref="FeatureContributionCalculatingTransformer"/> for a list of the suported models.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="modelParameters">Trained model parameters that support Feature Contribution Calculation and which will be used for scoring.</param>
        /// <param name="featureColumn">The name of the feature column that will be used as input.</param>
        /// <param name="top">The number of features with highest positive contributions for each data sample that will be retained in the FeatureContribution column.
        /// Note that if there are fewer features with positive contributions than <paramref name="top"/>, the rest will be returned as zeros.</param>
        /// <param name="bottom">The number of features with least negative contributions for each data sample that will be retained in the FeatureContribution column.
        /// Note that if there are fewer features with negative contributions than <paramref name="bottom"/>, the rest will be returned as zeros.</param>
        /// <param name="normalize">Whether the feature contributions should be normalized to the [-1, 1] interval.</param>
        public FeatureContributionCalculatingTransformer(IHostEnvironment env, ICalculateFeatureContribution modelParameters,
            string featureColumn = DefaultColumnNames.Features,
            int top = FeatureContributionCalculatingEstimator.Defaults.Top,
            int bottom = FeatureContributionCalculatingEstimator.Defaults.Bottom,
            bool normalize = FeatureContributionCalculatingEstimator.Defaults.Normalize)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(FeatureContributionCalculatingTransformer)), new[] { (input: featureColumn, output: DefaultColumnNames.FeatureContributions) })
        {
            Host.CheckValue(modelParameters, nameof(modelParameters));
            Host.CheckNonEmpty(featureColumn, nameof(featureColumn));
            if (top < 0)
                throw Host.Except($"Number of top contribution must be non negative");
            if (bottom < 0)
                throw Host.Except($"Number of bottom contribution must be non negative");

            // If a predictor implements ICalculateFeatureContribution, it also implements the internal interface IFeatureContributionMapper.
            // This is how we keep the implementation of feature contribution calculation internal.
            _predictor = modelParameters as IFeatureContributionMapper;
            Host.AssertValue(_predictor);

            Top = top;
            Bottom = bottom;
            Normalize = normalize;
        }

        private FeatureContributionCalculatingTransformer(IHostEnvironment env, ModelLoadContext ctx)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(FeatureContributionCalculatingTransformer)), ctx)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // base
            // IFeatureContributionMapper: predictor
            // int: top
            // int: bottom
            // bool: normalize

            ctx.LoadModel<IFeatureContributionMapper, SignatureLoadModel>(env, out _predictor, ModelFileUtils.DirPredictor);
            Top = ctx.Reader.ReadInt32();
            Contracts.CheckDecode(0 <= Top);
            Bottom = ctx.Reader.ReadInt32();
            Contracts.CheckDecode(0 <= Bottom);
            Normalize = ctx.Reader.ReadBoolByte();
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // base
            // IFeatureContributionMapper: predictor
            // int: top
            // int: bottom
            // bool: normalize

            SaveColumns(ctx);
            ctx.SaveModel(_predictor, ModelFileUtils.DirPredictor);
            Contracts.Assert(0 <= Top);
            ctx.Writer.Write(Top);
            Contracts.Assert(0 <= Bottom);
            ctx.Writer.Write(Bottom);
            ctx.Writer.WriteBoolByte(Normalize);
        }

        // Factory method for SignatureLoadModel.
        private static FeatureContributionCalculatingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            ctx.CheckAtModel(GetVersionInfo());
            return new FeatureContributionCalculatingTransformer(env, ctx);
        }

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, Schema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private protected override IRowMapper MakeRowMapper(Schema schema)
            => new Mapper(this, schema);

        private class Mapper : OneToOneMapperBase
        {
            private readonly FeatureContributionCalculatingTransformer _parent;
            private readonly VBuffer<ReadOnlyMemory<char>> _slotNames;
            private readonly int _featureColumnIndex;
            private readonly ColumnType _featureColumnType;

            public Mapper(FeatureContributionCalculatingTransformer parent, Schema schema)
                : base(parent.Host, parent, schema)
            {
                _parent = parent;

                // Check that the featureColumn is present and has the expected type.
                if (!schema.TryGetColumnIndex(_parent.ColumnPairs[0].input, out _featureColumnIndex))
                    throw Host.ExceptSchemaMismatch(nameof(schema), "input", _parent.ColumnPairs[0].input);
                _featureColumnType = schema[_featureColumnIndex].Type;
                if (_featureColumnType.ItemType != NumberType.R4 || !_featureColumnType.IsVector)
                    throw Host.ExceptSchemaMismatch(nameof(schema), "feature column", _parent.ColumnPairs[0].input, "Expected type is vector of float.", _featureColumnType.ItemType.ToString());

                if (InputSchema[_featureColumnIndex].HasSlotNames(_featureColumnType.VectorSize))
                    InputSchema[_featureColumnIndex].Metadata.GetValue(MetadataUtils.Kinds.SlotNames, ref _slotNames);
                else
                    _slotNames = VBufferUtils.CreateEmpty<ReadOnlyMemory<char>>(_featureColumnType.VectorSize);
            }

            // The FeatureContributionCalculatingTransformer produces two sets of columns: the columns obtained from scoring and the FeatureContribution column.
            // If the argument stringify is true, the type of the FeatureContribution column is string, otherwise it is a vector of float.
            protected override Schema.DetachedColumn[] GetOutputColumnsCore()
            {
                // Add FeatureContributions column.
                var builder = new MetadataBuilder();
                builder.Add(InputSchema[_featureColumnIndex].Metadata, x => x == MetadataUtils.Kinds.SlotNames);
                return new[] { new Schema.DetachedColumn(DefaultColumnNames.FeatureContributions, new VectorType(NumberType.R4, _featureColumnType.ValueCount), builder.GetMetadata()) };
            }

            protected override Delegate MakeGetter(Row input, int iinfo, Func<int, bool> active, out Action disposer)
            {
                disposer = null;
                Contracts.CheckValue(input, nameof(input));

                // REVIEW: Assuming Feature contributions will be VBuffer<float>.
                // For multiclass LR it needs to be VBuffer<float>[].
                return Utils.MarshalInvoke(GetValueGetter<int>, _featureColumnType.RawType, input, ColMapNewToOld[iinfo]);
            }

            private Delegate GetValueGetter<TSrc>(Row input, int colSrc)
            {
                Contracts.AssertValue(input);
                Contracts.AssertValue(_parent._predictor);

                var featureGetter = input.GetGetter<TSrc>(colSrc);

                var map = _parent._predictor.GetFeatureContributionMapper<TSrc, VBuffer<float>>(_parent.Top, _parent.Bottom, _parent.Normalize);
                var features = default(TSrc);

                return (ValueGetter<VBuffer<float>>)((ref VBuffer<float> dst) =>
                {
                    featureGetter(ref features);
                    map(in features, ref dst);
                });
            }
        }
    }

    /// <summary>
    /// Estimator producing a FeatureContributionCalculatingTransformer which scores the model on an input dataset and
    /// computes model-specific contribution scores for each feature.
    /// </summary>
    public sealed class FeatureContributionCalculatingEstimator : TrivialEstimator<FeatureContributionCalculatingTransformer>
    {
        private readonly string _featureColumn;
        private readonly ICalculateFeatureContribution _predictor;

        public static class Defaults
        {
            public const int Top = 10;
            public const int Bottom = 10;
            public const bool Normalize = true;
        }

        /// <summary>
        /// Feature Contribution Calculation computes model-specific contribution scores for each feature.
        /// Note that this functionality is not supported by all the models. See <see cref="FeatureContributionCalculatingTransformer"/> for a list of the suported models.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="modelParameters">Trained model parameters that support Feature Contribution Calculation and which will be used for scoring.</param>
        /// <param name="featureColumn">The name of the feature column that will be used as input.</param>
        /// <param name="top">The number of features with highest positive contributions for each data sample that will be retained in the FeatureContribution column.
        /// Note that if there are fewer features with positive contributions than <paramref name="top"/>, the rest will be returned as zeros.</param>
        /// <param name="bottom">The number of features with least negative contributions for each data sample that will be retained in the FeatureContribution column.
        /// Note that if there are fewer features with negative contributions than <paramref name="bottom"/>, the rest will be returned as zeros.</param>
        /// <param name="normalize">Whether the feature contributions should be normalized to the [-1, 1] interval.</param>
        public FeatureContributionCalculatingEstimator(IHostEnvironment env, ICalculateFeatureContribution modelParameters,
            string featureColumn = DefaultColumnNames.Features,
            int top = Defaults.Top,
            int bottom = Defaults.Bottom,
            bool normalize = Defaults.Normalize)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(FeatureContributionCalculatingTransformer)),
                  new FeatureContributionCalculatingTransformer(env, modelParameters, featureColumn, top, bottom, normalize))
        {
            _featureColumn = featureColumn;
            _predictor = modelParameters;
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            // Check that the featureColumn is present.
            Host.CheckValue(inputSchema, nameof(inputSchema));
            if (!inputSchema.TryFindColumn(_featureColumn, out var col))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", _featureColumn);
            // Check that the feature column is of the correct type: a vector of float.
            if (col.ItemType != NumberType.R4 || col.Kind != SchemaShape.Column.VectorKind.Vector)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "feature column", _featureColumn, "Expected type is vector of float.", col.GetTypeString());

            // Build output schemaShape.
            var result = inputSchema.ToDictionary(x => x.Name);

            // Add FeatureContributions column.
            var featContributionMetadata = new List<SchemaShape.Column>();
            if (col.Metadata.TryFindColumn(MetadataUtils.Kinds.SlotNames, out var slotMeta))
                featContributionMetadata.Add(slotMeta);
            result[DefaultColumnNames.FeatureContributions] = new SchemaShape.Column(
                DefaultColumnNames.FeatureContributions, col.Kind, col.ItemType, false, new SchemaShape(featContributionMetadata));

            return new SchemaShape(result.Values);
        }
    }

    internal static class FeatureContributionEntryPoint
    {
        [TlcModule.EntryPoint(Name = "Transforms.FeatureContributionCalculationTransformer",
            Desc = FeatureContributionCalculatingTransformer.Summary,
            UserName = FeatureContributionCalculatingTransformer.FriendlyName)]
        public static CommonOutputs.TransformOutput FeatureContributionCalculation(IHostEnvironment env, FeatureContributionCalculatingTransformer.Arguments args)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(nameof(FeatureContributionCalculatingTransformer));
            host.CheckValue(args, nameof(args));
            EntryPointUtils.CheckInputArgs(host, args);
            host.CheckValue(args.PredictorModel, nameof(args.PredictorModel));

            var predictor = args.PredictorModel.Predictor as ICalculateFeatureContribution;
            if (predictor == null)
                throw host.ExceptUserArg(nameof(predictor), "The provided model parameters do not support feature contribution calculation.");
            var outData = new FeatureContributionCalculatingTransformer(host, predictor, args.FeatureColumn, args.Top, args.Bottom, args.Normalize).Transform(args.Data);

            return new CommonOutputs.TransformOutput { Model = new TransformModelImpl(env, outData, args.Data), OutputData = outData};
        }
    }
}
