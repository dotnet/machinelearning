// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(FeatureContributionCalculatingTransformer.Summary, typeof(FeatureContributionCalculatingTransformer), null, typeof(SignatureLoadModel),
    FeatureContributionCalculatingTransformer.FriendlyName, FeatureContributionCalculatingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(FeatureContributionCalculatingTransformer), null, typeof(SignatureLoadRowMapper),
   FeatureContributionCalculatingTransformer.FriendlyName, FeatureContributionCalculatingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(void), typeof(FeatureContributionEntryPoint), null, typeof(SignatureEntryPointModule), FeatureContributionCalculatingTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// The FeatureContributionCalculationTransformer computes model-specific per-feature contributions to the score of each example.
    /// See the list of currently supported models below.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Scoring a dataset with a trained model produces a score, or prediction, for each example. To understand and explain these predictions
    /// it can be useful to inspect which features influenced them most significantly. FeatureContributionCalculationTransformer computes a model-specific
    /// list of per-feature contributions to the score for each example. These contributions can be positive (they make the score higher) or negative
    /// (they make the score lower).
    /// </para>
    /// <para>
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
    /// </para>
    /// <para>
    /// For linear models, the contribution of a given feature is equal to the product of feature value times the corresponding weight. Similarly,
    /// for Generalized Additive Models (GAM), the contribution of a feature is equal to the shape function for the given feature evaluated at
    /// the feature value.
    /// </para>
    /// <para>
    /// For tree-based models, the calculation of feature contribution essentially consists in determining which splits in the tree have the most impact
    /// on the final score and assigning the value of the impact to the features determining the split. More precisely, the contribution of a feature
    /// is equal to the change in score produced by exploring the opposite sub-tree every time a decision node for the given feature is encountered.
    /// Consider a simple case with a single decision tree that has a decision node for the binary feature F1. Given an example that has feature F1
    /// equal to true, we can calculate the score it would have obtained if we chose the subtree corresponding to the feature F1 being equal to false
    /// while keeping the other features constant. The contribution of feature F1 for the given example is the difference between the original score
    /// and the score obtained by taking the opposite decision at the node corresponding to feature F1. This algorithm extends naturally to models with
    /// many decision trees.
    /// </para>
    /// <para>
    /// See the sample below for an example of how to compute feature importance using the FeatureContributionCalculatingTransformer.
    /// </para>
    /// </remarks>
    public sealed class FeatureContributionCalculatingTransformer : OneToOneTransformerBase
    {
        internal sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "The predictor model to apply to data", SortOrder = 1)]
            public PredictorModel PredictorModel;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Name of feature column", SortOrder = 2)]
            public string FeatureColumn = DefaultColumnNames.Features;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of top contributions", SortOrder = 3)]
            public int Top = FeatureContributionCalculatingEstimator.Defaults.NumberOfPositiveContributions;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of bottom contributions", SortOrder = 4)]
            public int Bottom = FeatureContributionCalculatingEstimator.Defaults.NumberOfNegativeContributions;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether or not output of Features contribution should be normalized", ShortName = "norm", SortOrder = 5)]
            public bool Normalize = FeatureContributionCalculatingEstimator.Defaults.Normalize;
        }

        // Apparently, loader signature is limited in length to 24 characters.
        internal const string Summary = "For each data point, calculates the contribution of individual features to the model prediction.";
        internal const string FriendlyName = "Feature Contribution Calculation";
        internal const string LoaderSignature = "FeatureContribution";

        internal readonly int Top;
        internal readonly int Bottom;
        internal readonly bool Normalize;

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
        /// <param name="featureColumnName">The name of the feature column that will be used as input.</param>
        /// <param name="numberOfPositiveContributions">The number of positive contributions to report, sorted from highest magnitude to lowest magnitude.
        /// Note that if there are fewer features with positive contributions than <paramref name="numberOfPositiveContributions"/>, the rest will be returned as zeros.</param>
        /// <param name="numberOfNegativeContributions">The number of negative contributions to report, sorted from highest magnitude to lowest magnitude.
        /// Note that if there are fewer features with negative contributions than <paramref name="numberOfNegativeContributions"/>, the rest will be returned as zeros.</param>
        /// <param name="normalize">Whether the feature contributions should be normalized to the [-1, 1] interval.</param>
        internal FeatureContributionCalculatingTransformer(IHostEnvironment env, ICalculateFeatureContribution modelParameters,
            string featureColumnName = DefaultColumnNames.Features,
            int numberOfPositiveContributions = FeatureContributionCalculatingEstimator.Defaults.NumberOfPositiveContributions,
            int numberOfNegativeContributions = FeatureContributionCalculatingEstimator.Defaults.NumberOfNegativeContributions,
            bool normalize = FeatureContributionCalculatingEstimator.Defaults.Normalize)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(FeatureContributionCalculatingTransformer)), new[] { (name: DefaultColumnNames.FeatureContributions, source: featureColumnName) })
        {
            Host.CheckValue(modelParameters, nameof(modelParameters));
            Host.CheckNonEmpty(featureColumnName, nameof(featureColumnName));
            if (numberOfPositiveContributions < 0)
                throw Host.Except($"Number of top contribution must be non negative");
            if (numberOfNegativeContributions < 0)
                throw Host.Except($"Number of bottom contribution must be non negative");

            // If a predictor implements ICalculateFeatureContribution, it also implements the internal interface IFeatureContributionMapper.
            // This is how we keep the implementation of feature contribution calculation internal.
            _predictor = modelParameters as IFeatureContributionMapper;
            Host.AssertValue(_predictor);

            Top = numberOfPositiveContributions;
            Bottom = numberOfNegativeContributions;
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

        private protected override void SaveModel(ModelSaveContext ctx)
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
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema)
            => new Mapper(this, schema);

        private class Mapper : OneToOneMapperBase
        {
            private readonly FeatureContributionCalculatingTransformer _parent;
            private readonly VBuffer<ReadOnlyMemory<char>> _slotNames;
            private readonly int _featureColumnIndex;
            private readonly VectorType _featureColumnType;

            public Mapper(FeatureContributionCalculatingTransformer parent, DataViewSchema schema)
                : base(parent.Host, parent, schema)
            {
                _parent = parent;

                // Check that the featureColumn is present and has the expected type.
                if (!schema.TryGetColumnIndex(_parent.ColumnPairs[0].inputColumnName, out _featureColumnIndex))
                    throw Host.ExceptSchemaMismatch(nameof(schema), "input", _parent.ColumnPairs[0].inputColumnName);
                _featureColumnType = schema[_featureColumnIndex].Type as VectorType;
                if (_featureColumnType == null || _featureColumnType.ItemType != NumberDataViewType.Single)
                    throw Host.ExceptSchemaMismatch(nameof(schema), "feature", _parent.ColumnPairs[0].inputColumnName, "vector of float.", _featureColumnType.ItemType.ToString());

                if (InputSchema[_featureColumnIndex].HasSlotNames(_featureColumnType.Size))
                    InputSchema[_featureColumnIndex].Annotations.GetValue(AnnotationUtils.Kinds.SlotNames, ref _slotNames);
                else
                    _slotNames = VBufferUtils.CreateEmpty<ReadOnlyMemory<char>>(_featureColumnType.Size);
            }

            // The FeatureContributionCalculatingTransformer produces two sets of columns: the columns obtained from scoring and the FeatureContribution column.
            // If the argument stringify is true, the type of the FeatureContribution column is string, otherwise it is a vector of float.
            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                // Add FeatureContributions column.
                var builder = new DataViewSchema.Annotations.Builder();
                builder.Add(InputSchema[_featureColumnIndex].Annotations, x => x == AnnotationUtils.Kinds.SlotNames);
                return new[] { new DataViewSchema.DetachedColumn(DefaultColumnNames.FeatureContributions, new VectorType(NumberDataViewType.Single, _featureColumnType.Size), builder.ToAnnotations()) };
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> active, out Action disposer)
            {
                disposer = null;
                Contracts.CheckValue(input, nameof(input));

                // REVIEW: Assuming Feature contributions will be VBuffer<float>.
                // For multiclass LR it needs to be VBuffer<float>[].
                return Utils.MarshalInvoke(GetValueGetter<int>, _featureColumnType.RawType, input, ColMapNewToOld[iinfo]);
            }

            private Delegate GetValueGetter<TSrc>(DataViewRow input, int colSrc)
            {
                Contracts.AssertValue(input);
                Contracts.AssertValue(_parent._predictor);

                var featureGetter = input.GetGetter<TSrc>(input.Schema[colSrc]);

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

        internal static class Defaults
        {
            public const int NumberOfPositiveContributions = 10;
            public const int NumberOfNegativeContributions = 10;
            public const bool Normalize = true;
        }

        /// <summary>
        /// Feature Contribution Calculation computes model-specific contribution scores for each feature.
        /// Note that this functionality is not supported by all the models. See <see cref="FeatureContributionCalculatingTransformer"/> for a list of the suported models.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="model">A <see cref="ISingleFeaturePredictionTransformer{TModel}"/> that supports Feature Contribution Calculation,
        /// and which will also be used for scoring.</param>
        /// <param name="numberOfPositiveContributions">The number of positive contributions to report, sorted from highest magnitude to lowest magnitude.
        /// Note that if there are fewer features with positive contributions than <paramref name="numberOfPositiveContributions"/>, the rest will be returned as zeros.</param>
        /// <param name="numberOfNegativeContributions">The number of negative contributions to report, sorted from highest magnitude to lowest magnitude.
        /// Note that if there are fewer features with negative contributions than <paramref name="numberOfNegativeContributions"/>, the rest will be returned as zeros.</param>
        /// <param name="featureColumnName">TODO</param>
        /// <param name="normalize">Whether the feature contributions should be normalized to the [-1, 1] interval.</param>
        internal FeatureContributionCalculatingEstimator(IHostEnvironment env, ICalculateFeatureContribution model,
            int numberOfPositiveContributions = Defaults.NumberOfPositiveContributions,
            int numberOfNegativeContributions = Defaults.NumberOfNegativeContributions,
            string featureColumnName = DefaultColumnNames.Features,
            bool normalize = Defaults.Normalize)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(FeatureContributionCalculatingTransformer)),
                  new FeatureContributionCalculatingTransformer(env, model, featureColumnName, numberOfPositiveContributions, numberOfNegativeContributions, normalize))
        {
            _featureColumn = featureColumnName;
            _predictor = model;
        }

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            // Check that the featureColumn is present.
            Host.CheckValue(inputSchema, nameof(inputSchema));
            if (!inputSchema.TryFindColumn(_featureColumn, out var col))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", _featureColumn);
            // Check that the feature column is of the correct type: a vector of float.
            if (col.ItemType != NumberDataViewType.Single || col.Kind != SchemaShape.Column.VectorKind.Vector)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "column", _featureColumn, "vector of float.", col.GetTypeString());

            // Build output schemaShape.
            var result = inputSchema.ToDictionary(x => x.Name);

            // Add FeatureContributions column.
            var featContributionMetadata = new List<SchemaShape.Column>();
            if (col.Annotations.TryFindColumn(AnnotationUtils.Kinds.SlotNames, out var slotMeta))
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
        public static CommonOutputs.TransformOutput FeatureContributionCalculation(IHostEnvironment env, FeatureContributionCalculatingTransformer.Options options)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(nameof(FeatureContributionCalculatingTransformer));
            host.CheckValue(options, nameof(options));
            EntryPointUtils.CheckInputArgs(host, options);
            host.CheckValue(options.PredictorModel, nameof(options.PredictorModel));

            var predictor = options.PredictorModel.Predictor as ICalculateFeatureContribution;
            if (predictor == null)
                throw host.ExceptUserArg(nameof(predictor), "The provided model parameters do not support feature contribution calculation.");
            var outData = new FeatureContributionCalculatingTransformer(host, predictor, options.FeatureColumn, options.Top, options.Bottom, options.Normalize).Transform(options.Data);

            return new CommonOutputs.TransformOutput { Model = new TransformModelImpl(env, outData, options.Data), OutputData = outData};
        }
    }
}
