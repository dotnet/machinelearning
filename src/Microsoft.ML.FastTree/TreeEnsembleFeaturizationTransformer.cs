// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Data.Conversion;
using Microsoft.ML.Data.IO;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms;
using Microsoft.ML.TreePredictor;

[assembly: LoadableClass(typeof(TreeEnsembleFeaturizationTransformer), typeof(TreeEnsembleFeaturizationTransformer),
    null, typeof(SignatureLoadModel), "", TreeEnsembleFeaturizationTransformer.LoaderSignature)]

namespace Microsoft.ML.Trainers.FastTree
{
    public sealed class TreeEnsembleFeaturizationTransformer : PredictionTransformerBase<TreeEnsembleModelParameters>
    {
        internal const string LoaderSignature = "TreeEnseFeat";
        private readonly TreeEnsembleFeaturizerBindableMapper.Arguments _scorerArgs;
        private readonly DataViewSchema.DetachedColumn _featureDetachedColumn;
        private readonly string _outputColumnSuffix;

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

        public TreeEnsembleFeaturizationTransformer(IHostEnvironment env, DataViewSchema inputSchema,
            DataViewSchema.Column featureColumn, TreeEnsembleModelParameters modelParameters, string outputColumnNameSuffix=null) :
            base(Contracts.CheckRef(env, nameof(env)).Register(nameof(TreeEnsembleFeaturizationTransformer)), modelParameters, inputSchema)
        {
            // Store featureColumn as a detached column because a fitted transformer can be applied to different IDataViews and different
            // IDataView may have different schemas.
            _featureDetachedColumn = new DataViewSchema.DetachedColumn(featureColumn);
            // Check if featureColumn matches a column in inputSchema. The answer is yes if they have the same name and type.
            // The indexed column, inputSchema[featureColumn.Index], should match the detached column, _featureDetachedColumn.
            CheckFeatureColumnCompatibility(inputSchema[featureColumn.Index]);
            // Store outputColumnNameSuffix so that this transformer can be saved into a file later.
            _outputColumnSuffix = outputColumnNameSuffix;
            // Create an argument, _scorerArgs, to pass the suffix of output column names to the underlying scorer.
            _scorerArgs = new TreeEnsembleFeaturizerBindableMapper.Arguments { Suffix = _outputColumnSuffix };
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
            // string: output columns' suffix.

            string featureColumnName = ctx.LoadString();
            _featureDetachedColumn = new DataViewSchema.DetachedColumn(TrainSchema[featureColumnName]);
            _outputColumnSuffix = ctx.LoadStringOrNull();

            BindableMapper = ScoreUtils.GetSchemaBindableMapper(Host, Model);

            var args = new GenericScorer.Arguments { Suffix = "" };
            var schema = MakeFeatureRoleMappedSchema(TrainSchema);
            Scorer = new GenericScorer(Host, args, new EmptyDataView(Host, TrainSchema), BindableMapper.Bind(Host, schema), schema);
        }

        public override DataViewSchema GetOutputSchema(DataViewSchema inputSchema) => Transform(new EmptyDataView(Host, inputSchema)).Schema;

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // model: prediction model.
            // stream: empty data view that contains train schema.
            // ids of strings: feature columns.
            // float: scorer threshold
            // id of string: scorer threshold column

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
            ctx.SaveStringOrNull(_outputColumnSuffix);
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

        private static TreeEnsembleFeaturizationTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
            => new TreeEnsembleFeaturizationTransformer(env, ctx);
    }
}