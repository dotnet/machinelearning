// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.FastTree;

[assembly: LoadableClass(typeof(TreeEnsembleFeaturizationTransformer), typeof(TreeEnsembleFeaturizationTransformer),
    null, typeof(SignatureLoadModel), "", TreeEnsembleFeaturizationTransformer.LoaderSignature)]

namespace Microsoft.ML.Trainers.FastTree
{
    /// <summary>
    /// <see cref="ITransformer"/> resulting from fitting any derived class of <see cref="TreeEnsembleFeaturizationEstimatorBase"/>.
    /// The derived classes include, for example, <see cref="FastTreeBinaryFeaturizationEstimator"/> and
    /// <see cref="FastForestRegressionFeaturizationEstimator"/>.
    /// </summary>
    public sealed class TreeEnsembleFeaturizationTransformer : PredictionTransformerBase<TreeEnsembleModelParameters>
    {
        internal const string LoaderSignature = "TreeEnseFeat";
        private readonly TreeEnsembleFeaturizerBindableMapper.Arguments _scorerArgs;
        private readonly DataViewSchema.DetachedColumn _featureDetachedColumn;
        /// <summary>
        /// See <see cref="TreeEnsembleFeaturizationEstimatorBase.OptionsBase.TreesColumnName"/>.
        /// </summary>
        private readonly string _treesColumnName;
        /// <summary>
        /// See <see cref="TreeEnsembleFeaturizationEstimatorBase.OptionsBase.LeavesColumnName"/>.
        /// </summary>
        private readonly string _leavesColumnName;
        /// <summary>
        /// See <see cref="TreeEnsembleFeaturizationEstimatorBase.OptionsBase.PathsColumnName"/>.
        /// </summary>
        private readonly string _pathsColumnName;
        /// <summary>
        /// Check if <see cref="_featureDetachedColumn"/> is compatible with <paramref name="inspectedFeatureColumn"/>.
        /// </summary>
        /// <param name="inspectedFeatureColumn">A column checked against <see cref="_featureDetachedColumn"/>.</param>
        private void CheckFeatureColumnCompatibility(DataViewSchema.Column inspectedFeatureColumn)
        {
            string nameErrorMessage = $"The column called {inspectedFeatureColumn.Name} does not match the expected " +
                $"feature column with name {_featureDetachedColumn.Name} and type {_featureDetachedColumn.Type}. " +
                $"Please rename your column by calling CopyColumns defined in TransformExtensionsCatalog";
            // Check if column names are the same.
            Host.Check(_featureDetachedColumn.Name == inspectedFeatureColumn.Name, nameErrorMessage);

            string typeErrorMessage = $"The column called {inspectedFeatureColumn.Name} has a type {inspectedFeatureColumn.Type}, " +
                $"which does not match the expected feature column with name {_featureDetachedColumn.Name} and type {_featureDetachedColumn.Type}. " +
                $"Please make sure your feature column type is {_featureDetachedColumn.Type}.";
            // Check if column types are identical.
            Host.Check(_featureDetachedColumn.Type.Equals(inspectedFeatureColumn.Type), typeErrorMessage);
        }

        /// <summary>
        /// Create <see cref="RoleMappedSchema"/> from <paramref name="schema"/> by using <see cref="_featureDetachedColumn"/> as the feature role.
        /// </summary>
        /// <param name="schema">The original schema to be mapped.</param>
        private RoleMappedSchema MakeFeatureRoleMappedSchema(DataViewSchema schema)
        {
            var roles = new List<KeyValuePair<RoleMappedSchema.ColumnRole, string>>();
            roles.Add(new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RoleMappedSchema.ColumnRole.Feature, _featureDetachedColumn.Name));
            return new RoleMappedSchema(schema, roles);
        }

        internal TreeEnsembleFeaturizationTransformer(IHostEnvironment env, DataViewSchema inputSchema,
            DataViewSchema.Column featureColumn, TreeEnsembleModelParameters modelParameters,
            string treesColumnName, string leavesColumnName, string pathsColumnName) :
            base(Contracts.CheckRef(env, nameof(env)).Register(nameof(TreeEnsembleFeaturizationTransformer)), modelParameters, inputSchema)
        {
            // Store featureColumn as a detached column because a fitted transformer can be applied to different IDataViews and different
            // IDataView may have different schemas.
            _featureDetachedColumn = new DataViewSchema.DetachedColumn(featureColumn);

            // Check if featureColumn matches a column in inputSchema. The answer is yes if they have the same name and type.
            // The indexed column, inputSchema[featureColumn.Index], should match the detached column, _featureDetachedColumn.
            CheckFeatureColumnCompatibility(inputSchema[featureColumn.Index]);

            // Store output column names so that this transformer can be saved into a file later.
            _treesColumnName = treesColumnName;
            _leavesColumnName = leavesColumnName;
            _pathsColumnName = pathsColumnName;

            // Create an argument, _scorerArgs, to pass the output column names to the underlying scorer.
            _scorerArgs = new TreeEnsembleFeaturizerBindableMapper.Arguments
            {
                TreesColumnName = _treesColumnName,
                LeavesColumnName = _leavesColumnName,
                PathsColumnName = _pathsColumnName
            };

            // Create a bindable mapper. It provides the core computation and can be attached to any IDataView and produce
            // a transformed IDataView.
            BindableMapper = new TreeEnsembleFeaturizerBindableMapper(env, _scorerArgs, modelParameters);

            // Create a scorer.
            var roleMappedSchema = MakeFeatureRoleMappedSchema(inputSchema);
            Scorer = new GenericScorer(Host, _scorerArgs, new EmptyDataView(Host, inputSchema), BindableMapper.Bind(Host, roleMappedSchema), roleMappedSchema);
        }

        private TreeEnsembleFeaturizationTransformer(IHostEnvironment host, ModelLoadContext ctx)
            : base(Contracts.CheckRef(host, nameof(host)).Register(nameof(TreeEnsembleFeaturizationTransformer)), ctx)
        {
            // *** Binary format ***
            // <base info>
            // string: feature column's name.
            // string: the name of the columns where tree prediction values are stored.
            // string: the name of the columns where trees' leave are stored.
            // string: the name of the columns where trees' paths are stored.

            // Load stored fields.
            string featureColumnName = ctx.LoadString();
            _featureDetachedColumn = new DataViewSchema.DetachedColumn(TrainSchema[featureColumnName]);
            _treesColumnName = ctx.LoadStringOrNull();
            _leavesColumnName = ctx.LoadStringOrNull();
            _pathsColumnName = ctx.LoadStringOrNull();

            // Create an argument to specify output columns' names of this transformer.
            _scorerArgs = new TreeEnsembleFeaturizerBindableMapper.Arguments
            {
                TreesColumnName = _treesColumnName,
                LeavesColumnName = _leavesColumnName,
                PathsColumnName = _pathsColumnName
            };

            // Create a bindable mapper. It provides the core computation and can be attached to any IDataView and produce
            // a transformed IDataView.
            BindableMapper = new TreeEnsembleFeaturizerBindableMapper(host, _scorerArgs, Model);

            // Create a scorer.
            var roleMappedSchema = MakeFeatureRoleMappedSchema(TrainSchema);
            Scorer = new GenericScorer(Host, _scorerArgs, new EmptyDataView(Host, TrainSchema), BindableMapper.Bind(Host, roleMappedSchema), roleMappedSchema);
        }

        /// <summary>
        /// <see cref="TreeEnsembleFeaturizationTransformer"/> appends three columns to the <paramref name="inputSchema"/>.
        /// The three columns are all <see cref="System.Single"/> vectors. The fist column stores the prediction values of all trees and
        /// its default name is "Trees". The second column (default name: "Leaves") contains leaf IDs where the given feature vector falls into.
        /// The third column (default name: "Paths") encodes the paths to those leaves via a 0-1 vector.
        /// </summary>
        /// <param name="inputSchema"><see cref="DataViewSchema"/> of the data to be transformed.</param>
        /// <returns><see cref="DataViewSchema"/> of the transformed data if the input schema is <paramref name="inputSchema"/>.</returns>
        public override DataViewSchema GetOutputSchema(DataViewSchema inputSchema) => Transform(new EmptyDataView(Host, inputSchema)).Schema;

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // model: prediction model.
            // stream: empty data view that contains train schema.
            // string: feature column name.
            // string: the name of the columns where tree prediction values are stored.
            // string: the name of the columns where trees' leave are stored.
            // string: the name of the columns where trees' paths are stored.

            ctx.SaveModel(Model, DirModel);
            ctx.SaveBinaryStream(DirTransSchema, writer =>
            {
                using (var ch = Host.Start("Saving train schema"))
                {
                    var saver = new BinarySaver(Host, new BinarySaver.Arguments { Silent = true });
                    DataSaverUtils.SaveDataView(ch, saver, new EmptyDataView(Host, TrainSchema), writer.BaseStream);
                }
            });

            ctx.SaveString(_featureDetachedColumn.Name);
            ctx.SaveStringOrNull(_treesColumnName);
            ctx.SaveStringOrNull(_leavesColumnName);
            ctx.SaveStringOrNull(_pathsColumnName);
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TREEFEAT", // "TREE" ensemble "FEAT"urizer.
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(TreeEnsembleFeaturizationTransformer).Assembly.FullName);
        }

        internal static TreeEnsembleFeaturizationTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
            => new TreeEnsembleFeaturizationTransformer(env, ctx);
    }
}
