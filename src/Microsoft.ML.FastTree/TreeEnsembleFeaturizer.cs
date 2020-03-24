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
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms;
using Microsoft.ML.TreePredictor;

[assembly: LoadableClass(typeof(ISchemaBindableMapper), typeof(TreeEnsembleFeaturizerTransform), typeof(TreeEnsembleFeaturizerBindableMapper.Arguments),
    typeof(SignatureBindableMapper), "Tree Ensemble Featurizer Mapper", TreeEnsembleFeaturizerBindableMapper.LoadNameShort)]

[assembly: LoadableClass(typeof(IDataScorerTransform), typeof(TreeEnsembleFeaturizerTransform), typeof(TreeEnsembleFeaturizerBindableMapper.Arguments),
    typeof(SignatureDataScorer), "Tree Ensemble Featurizer Scorer", TreeEnsembleFeaturizerBindableMapper.LoadNameShort)]

[assembly: LoadableClass(typeof(ISchemaBindableMapper), typeof(TreeEnsembleFeaturizerTransform), null, typeof(SignatureLoadModel),
    "Tree Ensemble Featurizer Mapper", TreeEnsembleFeaturizerBindableMapper.LoaderSignature)]

[assembly: LoadableClass(TreeEnsembleFeaturizerTransform.TreeEnsembleSummary, typeof(IDataTransform), typeof(TreeEnsembleFeaturizerTransform),
    typeof(TreeEnsembleFeaturizerTransform.Arguments), typeof(SignatureDataTransform),
    TreeEnsembleFeaturizerTransform.UserName, TreeEnsembleFeaturizerBindableMapper.LoadNameShort, "TreeFeaturizationTransform")]

[assembly: LoadableClass(typeof(void), typeof(TreeFeaturize), null, typeof(SignatureEntryPointModule), "TreeFeaturize")]

namespace Microsoft.ML.Data
{
    /// <summary>
    /// A bindable mapper wrapper for tree ensembles, that creates a bound mapper with three outputs:
    /// 1. A vector containing the individual tree outputs of the tree ensemble.
    /// 2. An indicator vector for the leaves that the feature vector falls on in the tree ensemble.
    /// 3. An indicator vector for the internal nodes on the paths that the feature vector falls on in the tree ensemble.
    /// </summary>
    internal sealed class TreeEnsembleFeaturizerBindableMapper : ISchemaBindableMapper, ICanSaveModel
    {
        /// <summary>
        /// In addition to options inherited from <see cref="ScorerArgumentsBase"/>,
        /// <see cref="Arguments"/> adds output columns' names of tree-based featurizer.
        /// </summary>
        public sealed class Arguments : ScorerArgumentsBase
        {
            /// <summary>
            /// See <see cref="TreeEnsembleFeaturizationEstimatorBase.OptionsBase.TreesColumnName"/>.
            /// </summary>
            public string TreesColumnName;
            /// <summary>
            /// See <see cref="TreeEnsembleFeaturizationEstimatorBase.OptionsBase.LeavesColumnName"/>.
            /// </summary>
            public string LeavesColumnName;
            /// <summary>
            /// See <see cref="TreeEnsembleFeaturizationEstimatorBase.OptionsBase.PathsColumnName"/>.
            /// </summary>
            public string PathsColumnName;
        }

        private sealed class BoundMapper : ISchemaBoundRowMapper
        {
            public RoleMappedSchema InputRoleMappedSchema { get; }
            public DataViewSchema InputSchema => InputRoleMappedSchema.Schema;
            public DataViewSchema OutputSchema { get; }
            public ISchemaBindableMapper Bindable => _owner;

            private readonly TreeEnsembleFeaturizerBindableMapper _owner;
            private readonly IExceptionContext _ectx;

            /// <summary>
            /// Feature vector to be mapped to tree-based features.
            /// </summary>
            private DataViewSchema.Column FeatureColumn => InputRoleMappedSchema.Feature.Value;

            /// <summary>
            /// The name of the column that stores the prediction values of all trees. Its type is a vector of <see cref="System.Single"/>
            /// and the i-th vector element is the prediction value predicted by the i-th tree.
            /// If <see cref="_treesColumnName"/> is <see langword="null"/>, this output column may not be generated.
            /// </summary>
            private string _treesColumnName;

            /// <summary>
            /// The 0-1 encoding of all leaf nodes' IDs. Its type is a vector of <see cref="System.Single"/>. If the given feature
            /// vector falls into the first leaf of the first tree, the first element in the 0-1 encoding would be 1.
            /// If <see cref="_leavesColumnName"/> is <see langword="null"/>, this output column may not be generated.
            /// </summary>
            private string _leavesColumnName;

            /// <summary>
            /// The 0-1 encoding of the paths to the leaves. If the path to the first tree's leaf is node 1 (2nd node in the first tree),
            /// node 3 (4th node in the first tree), and node 5 (6th node in the first tree), the 2nd, 4th, and 6th element in that encoding
            /// would be 1.
            /// If <see cref="_pathsColumnName"/> is <see langword="null"/>, this output column may not be generated.
            /// </summary>
            private string _pathsColumnName;

            public BoundMapper(IExceptionContext ectx, TreeEnsembleFeaturizerBindableMapper owner, RoleMappedSchema schema,
                string treesColumnName, string leavesColumnName, string pathsColumnName)
            {
                Contracts.AssertValue(ectx);
                ectx.AssertValue(owner);
                ectx.AssertValue(schema);
                ectx.Assert(schema.Feature.HasValue);

                _ectx = ectx;

                _owner = owner;
                InputRoleMappedSchema = schema;

                // A vector containing the output of each tree on a given example.
                var treeValueType = new VectorDataViewType(NumberDataViewType.Single, owner._ensemble.TrainedEnsemble.NumTrees);
                // An indicator vector with length = the total number of leaves in the ensemble, indicating which leaf the example
                // ends up in all the trees in the ensemble.
                var leafIdType = new VectorDataViewType(NumberDataViewType.Single, owner._totalLeafCount);
                // An indicator vector with length = the total number of nodes in the ensemble, indicating the nodes on
                // the paths of the example in all the trees in the ensemble.
                // The total number of nodes in a binary tree is equal to the number of internal nodes + the number of leaf nodes,
                // and it is also equal to the number of children of internal nodes (which is 2 * the number of internal nodes)
                // plus one (since the root node is not a child of any node). So we have #internal + #leaf = 2*(#internal) + 1,
                // which means that #internal = #leaf - 1.
                // Therefore, the number of internal nodes in the ensemble is #leaf - #trees.
                var pathIdType = new VectorDataViewType(NumberDataViewType.Single, owner._totalLeafCount - owner._ensemble.TrainedEnsemble.NumTrees);

                // Start creating output schema with types derived above.
                var schemaBuilder = new DataViewSchema.Builder();

                _treesColumnName = treesColumnName;
                if (treesColumnName != null)
                {
                    // Metadata of tree values.
                    var treeIdMetadataBuilder = new DataViewSchema.Annotations.Builder();
                    treeIdMetadataBuilder.Add(AnnotationUtils.Kinds.SlotNames, AnnotationUtils.GetNamesType(treeValueType.Size),
                        (ValueGetter<VBuffer<ReadOnlyMemory<char>>>)owner.GetTreeSlotNames);

                    // Add the column of trees' output values
                    schemaBuilder.AddColumn(treesColumnName, treeValueType, treeIdMetadataBuilder.ToAnnotations());
                }

                _leavesColumnName = leavesColumnName;
                if (leavesColumnName != null)
                {
                    // Metadata of leaf IDs.
                    var leafIdMetadataBuilder = new DataViewSchema.Annotations.Builder();
                    leafIdMetadataBuilder.Add(AnnotationUtils.Kinds.SlotNames, AnnotationUtils.GetNamesType(leafIdType.Size),
                        (ValueGetter<VBuffer<ReadOnlyMemory<char>>>)owner.GetLeafSlotNames);
                    leafIdMetadataBuilder.Add(AnnotationUtils.Kinds.IsNormalized, BooleanDataViewType.Instance, (ref bool value) => value = true);

                    // Add the column of leaves' IDs where the input example reaches.
                    schemaBuilder.AddColumn(leavesColumnName, leafIdType, leafIdMetadataBuilder.ToAnnotations());
                }

                _pathsColumnName = pathsColumnName;
                if (pathsColumnName !=  null)
                {
                    // Metadata of path IDs.
                    var pathIdMetadataBuilder = new DataViewSchema.Annotations.Builder();
                    pathIdMetadataBuilder.Add(AnnotationUtils.Kinds.SlotNames, AnnotationUtils.GetNamesType(pathIdType.Size),
                        (ValueGetter<VBuffer<ReadOnlyMemory<char>>>)owner.GetPathSlotNames);
                    pathIdMetadataBuilder.Add(AnnotationUtils.Kinds.IsNormalized, BooleanDataViewType.Instance, (ref bool value) => value = true);

                    // Add the column of encoded paths which the input example passes.
                    schemaBuilder.AddColumn(pathsColumnName, pathIdType, pathIdMetadataBuilder.ToAnnotations());
                }

                OutputSchema = schemaBuilder.ToSchema();
            }

            DataViewRow ISchemaBoundRowMapper.GetRow(DataViewRow input, IEnumerable<DataViewSchema.Column> activeColumns)
            {
                _ectx.CheckValue(input, nameof(input));
                _ectx.CheckValue(activeColumns, nameof(activeColumns));
                return new SimpleRow(OutputSchema, input, CreateGetters(input, activeColumns));
            }

            private Delegate[] CreateGetters(DataViewRow input, IEnumerable<DataViewSchema.Column> activeColumns)
            {
                _ectx.AssertValue(input);
                _ectx.AssertValue(activeColumns);

                var delegates = new List<Delegate>();
                var activeIndices = activeColumns.Select(c => c.Index);
                var state = new State(_ectx, input, _owner._ensemble, _owner._totalLeafCount, FeatureColumn.Index);

                // Get the tree value getter.
                if (_treesColumnName != null)
                {
                    ValueGetter<VBuffer<float>> fn = state.GetTreeValues;
                    if(activeIndices.Contains(OutputSchema[_treesColumnName].Index))
                        delegates.Add(fn);
                    else
                        delegates.Add(null);
                }

                // Get the leaf indicator getter.
                if (_leavesColumnName != null )
                {
                    ValueGetter<VBuffer<float>> fn = state.GetLeafIds;
                    if (activeIndices.Contains(OutputSchema[_leavesColumnName].Index))
                        delegates.Add(fn);
                    else
                        delegates.Add(null);
                }

                // Get the path indicators getter.
                if (_pathsColumnName != null)
                {
                    ValueGetter<VBuffer<float>> fn = state.GetPathIds;
                    if (activeIndices.Contains(OutputSchema[_pathsColumnName].Index))
                        delegates.Add(fn);
                    else
                        delegates.Add(null);
                }

                return delegates.ToArray();
            }

            private sealed class State
            {
                private readonly IExceptionContext _ectx;
                private readonly DataViewRow _input;
                private readonly TreeEnsembleModelParameters _ensemble;
                private readonly int _numTrees;
                private readonly int _numLeaves;

                private VBuffer<float> _src;
                private ValueGetter<VBuffer<float>> _featureGetter;
                private long _cachedPosition;
                private readonly int[] _leafIds;
                private readonly List<int>[] _pathIds;

                private BufferBuilder<float> _leafIdBuilder;
                private BufferBuilder<float> _pathIdBuilder;
                private long _cachedLeafBuilderPosition;
                private long _cachedPathBuilderPosition;

                public State(IExceptionContext ectx, DataViewRow input, TreeEnsembleModelParameters ensemble, int numLeaves, int featureIndex)
                {
                    Contracts.AssertValue(ectx);
                    _ectx = ectx;
                    _ectx.AssertValue(input);
                    _ectx.AssertValue(ensemble);
                    _ectx.Assert(ensemble.TrainedEnsemble.NumTrees > 0);
                    _input = input;
                    _ensemble = ensemble;
                    _numTrees = _ensemble.TrainedEnsemble.NumTrees;
                    _numLeaves = numLeaves;

                    _src = default(VBuffer<float>);
                    _featureGetter = input.GetGetter<VBuffer<float>>(input.Schema[featureIndex]);

                    _cachedPosition = -1;
                    _leafIds = new int[_numTrees];
                    _pathIds = new List<int>[_numTrees];
                    for (int i = 0; i < _numTrees; i++)
                        _pathIds[i] = new List<int>();

                    _cachedLeafBuilderPosition = -1;
                    _cachedPathBuilderPosition = -1;
                }

                public void GetTreeValues(ref VBuffer<float> dst)
                {
                    EnsureCachedPosition();
                    var editor = VBufferEditor.Create(ref dst, _numTrees);
                    for (int i = 0; i < _numTrees; i++)
                        editor.Values[i] = _ensemble.GetLeafValue(i, _leafIds[i]);

                    dst = editor.Commit();
                }

                public void GetLeafIds(ref VBuffer<float> dst)
                {
                    EnsureCachedPosition();

                    _ectx.Assert(_input.Position >= 0);
                    _ectx.Assert(_cachedPosition == _input.Position);

                    if (_cachedLeafBuilderPosition != _input.Position)
                    {
                        if (_leafIdBuilder == null)
                            _leafIdBuilder = BufferBuilder<float>.CreateDefault();

                        _leafIdBuilder.Reset(_numLeaves, false);
                        var offset = 0;
                        var trees = ((ITreeEnsemble)_ensemble).GetTrees();
                        for (int i = 0; i < trees.Length; i++)
                        {
                            _leafIdBuilder.AddFeature(offset + _leafIds[i], 1);
                            offset += trees[i].NumLeaves;
                        }

                        _cachedLeafBuilderPosition = _input.Position;
                    }
                    _ectx.AssertValue(_leafIdBuilder);
                    _leafIdBuilder.GetResult(ref dst);
                }

                public void GetPathIds(ref VBuffer<float> dst)
                {
                    EnsureCachedPosition();
                    _ectx.Assert(_input.Position >= 0);
                    _ectx.Assert(_cachedPosition == _input.Position);

                    if (_cachedPathBuilderPosition != _input.Position)
                    {
                        if (_pathIdBuilder == null)
                            _pathIdBuilder = BufferBuilder<float>.CreateDefault();

                        var trees = ((ITreeEnsemble)_ensemble).GetTrees();
                        _pathIdBuilder.Reset(_numLeaves - _numTrees, dense: false);
                        var offset = 0;
                        for (int i = 0; i < _numTrees; i++)
                        {
                            var numNodes = trees[i].NumLeaves - 1;
                            var nodes = _pathIds[i];
                            _ectx.AssertValue(nodes);
                            for (int j = 0; j < nodes.Count; j++)
                            {
                                var node = nodes[j];
                                _ectx.Assert(0 <= node && node < numNodes);
                                _pathIdBuilder.AddFeature(offset + node, 1);
                            }
                            offset += numNodes;
                        }

                        _cachedPathBuilderPosition = _input.Position;
                    }
                    _ectx.AssertValue(_pathIdBuilder);
                    _pathIdBuilder.GetResult(ref dst);
                }

                private void EnsureCachedPosition()
                {
                    _ectx.Check(_input.Position >= 0, RowCursorUtils.FetchValueStateError);
                    if (_cachedPosition != _input.Position)
                    {
                        _featureGetter(ref _src);

                        _ectx.Assert(Utils.Size(_leafIds) == _numTrees);
                        _ectx.Assert(Utils.Size(_pathIds) == _numTrees);

                        for (int i = 0; i < _numTrees; i++)
                            _leafIds[i] = _ensemble.GetLeaf(i, in _src, ref _pathIds[i]);

                        _cachedPosition = _input.Position;
                    }
                }
            }

            public IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> GetInputColumnRoles()
            {
                yield return RoleMappedSchema.ColumnRole.Feature.Bind(FeatureColumn.Name);
            }

            /// <summary>
            /// Given a set of columns, return the input columns that are needed to generate those output columns.
            /// </summary>
            public IEnumerable<DataViewSchema.Column> GetDependenciesForNewColumns(IEnumerable<DataViewSchema.Column> dependingColumns)
            {
                if (dependingColumns.Count() == 0)
                    return Enumerable.Empty<DataViewSchema.Column>();

                return Enumerable.Repeat(FeatureColumn, 1);
            }
        }

        public const string LoadNameShort = "TreeFeat";
        public const string LoaderSignature = "TreeEnsembleMapper";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TREEMAPR",
                // verWrittenCur: 0x00010001, // Initial
                // verWrittenCur: 0x00010002, // Add _defaultValueForMissing
                verWrittenCur: 0x00010003, // Add output column names (_treesColumnName, _leavesColumnName, _pathsColumnName)
                verReadableCur: 0x00010003,
                verWeCanReadBack: 0x00010002,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(TreeEnsembleFeaturizerBindableMapper).Assembly.FullName);
        }

        private readonly IHost _host;
        private readonly TreeEnsembleModelParameters _ensemble;
        private readonly int _totalLeafCount;
        private readonly string _treesColumnName;
        private readonly string _leavesColumnName;
        private readonly string _pathsColumnName;

        public TreeEnsembleFeaturizerBindableMapper(IHostEnvironment env, Arguments args, IPredictor predictor)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(LoaderSignature);
            _host.CheckValue(args, nameof(args));
            _host.CheckValue(predictor, nameof(predictor));

            // Store output columns specified by the user.
            _treesColumnName = args.TreesColumnName;
            _leavesColumnName = args.LeavesColumnName;
            _pathsColumnName = args.PathsColumnName;

            // This function accepts models trained by FastTreeTrainer family. There are four types that "predictor" can be.
            //  1. CalibratedPredictorBase<FastTreeBinaryModelParameters, PlattCalibrator>
            //  2. FastTreeRankingModelParameters
            //  3. FastTreeRegressionModelParameters
            //  4. FastTreeTweedieModelParameters
            // Only (1) needs a special cast right below because all others are derived from TreeEnsembleModelParameters.
            if (predictor is CalibratedModelParametersBase<FastTreeBinaryModelParameters, PlattCalibrator> calibrated)
                predictor = calibrated.SubModel;
            _ensemble = predictor as TreeEnsembleModelParameters;
            _host.Check(_ensemble != null, "Predictor in model file does not have compatible type");

            _totalLeafCount = CountLeaves(_ensemble);
        }

        public TreeEnsembleFeaturizerBindableMapper(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(LoaderSignature);
            _host.AssertValue(ctx);
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // ensemble
            // string: treesColumnName
            // string: leavesColumnName
            // string: pathsColumnName

            ctx.LoadModel<TreeEnsembleModelParameters, SignatureLoadModel>(env, out _ensemble, "Ensemble");
            _totalLeafCount = CountLeaves(_ensemble);
            _treesColumnName = ctx.LoadStringOrNull();
            _leavesColumnName = ctx.LoadStringOrNull();
            _pathsColumnName = ctx.LoadStringOrNull();
        }

        void ICanSaveModel.Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // ensemble
            // string: treesColumnName
            // string: leavesColumnName
            // string: pathsColumnName

            _host.AssertValue(_ensemble);
            ctx.SaveModel(_ensemble, "Ensemble");
            ctx.SaveStringOrNull(_treesColumnName);
            ctx.SaveStringOrNull(_leavesColumnName);
            ctx.SaveStringOrNull(_pathsColumnName);
        }

        private static int CountLeaves(TreeEnsembleModelParameters ensemble)
        {
            Contracts.AssertValue(ensemble);

            var trees = ((ITreeEnsemble)ensemble).GetTrees();
            var numTrees = trees.Length;
            var totalLeafCount = 0;
            for (int i = 0; i < numTrees; i++)
                totalLeafCount += trees[i].NumLeaves;
            return totalLeafCount;
        }

        private void GetTreeSlotNames(ref VBuffer<ReadOnlyMemory<char>> dst)
        {
            var numTrees = _ensemble.TrainedEnsemble.NumTrees;

            var editor = VBufferEditor.Create(ref dst, numTrees);
            for (int t = 0; t < numTrees; t++)
                editor.Values[t] = string.Format("Tree{0:000}", t).AsMemory();

            dst = editor.Commit();
        }

        private void GetLeafSlotNames(ref VBuffer<ReadOnlyMemory<char>> dst)
        {
            var numTrees = _ensemble.TrainedEnsemble.NumTrees;

            var editor = VBufferEditor.Create(ref dst, _totalLeafCount);
            int i = 0;
            int t = 0;
            foreach (var tree in ((ITreeEnsemble)_ensemble).GetTrees())
            {
                for (int l = 0; l < tree.NumLeaves; l++)
                    editor.Values[i++] = string.Format("Tree{0:000}Leaf{1:000}", t, l).AsMemory();
                t++;
            }
            _host.Assert(i == _totalLeafCount);
            dst = editor.Commit();
        }

        private void GetPathSlotNames(ref VBuffer<ReadOnlyMemory<char>> dst)
        {
            var numTrees = _ensemble.TrainedEnsemble.NumTrees;

            var totalNodeCount = _totalLeafCount - numTrees;
            var editor = VBufferEditor.Create(ref dst, totalNodeCount);

            int i = 0;
            int t = 0;
            foreach (var tree in ((ITreeEnsemble)_ensemble).GetTrees())
            {
                var numLeaves = tree.NumLeaves;
                for (int l = 0; l < tree.NumLeaves - 1; l++)
                    editor.Values[i++] = string.Format("Tree{0:000}Node{1:000}", t, l).AsMemory();
                t++;
            }
            _host.Assert(i == totalNodeCount);
            dst = editor.Commit();
        }

        ISchemaBoundMapper ISchemaBindableMapper.Bind(IHostEnvironment env, RoleMappedSchema schema)
        {
            Contracts.AssertValue(env);
            env.AssertValue(schema);
            env.CheckParam(schema.Feature != null, nameof(schema), "Need a feature column");

            return new BoundMapper(env, this, schema, _treesColumnName, _leavesColumnName, _pathsColumnName);
        }
    }

    /// <include file='doc.xml' path='doc/members/member[@name="TreeEnsembleFeaturizerTransform"]'/>
    [BestFriend]
    internal static class TreeEnsembleFeaturizerTransform
    {
#pragma warning disable CS0649 // The fields will still be set via the reflection driven mechanisms.
        public sealed class Arguments : TrainAndScoreTransformer.ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "Trainer to use", ShortName = "tr", NullName = "<None>", SortOrder = 1, SignatureType = typeof(SignatureTreeEnsembleTrainer))]
            public IComponentFactory<ITrainer> Trainer;

            [Argument(ArgumentType.AtMostOnce, IsInputFileName = true, HelpText = "Predictor model file used in scoring",
                ShortName = "in", SortOrder = 2)]
            public string TrainedModelFile;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Output column: The suffix to append to the default column names",
                ShortName = "ex", SortOrder = 101)]
            public string Suffix;

            [Argument(ArgumentType.AtMostOnce, HelpText = "If specified, determines the permutation seed for applying this featurizer to a multiclass problem.",
                ShortName = "lps", SortOrder = 102)]
            public int LabelPermutationSeed;
        }

        /// <summary>
        /// REVIEW: Ideally we should have only one arguments class by using IComponentFactory for the model.
        /// For now it probably warrants a REVIEW comment here in case we'd like to merge these two arguments in the future.
        /// Also, it might be worthwhile to extract the common arguments to a base class.
        /// </summary>
        [TlcModule.EntryPointKind(typeof(CommonInputs.IFeaturizerInput))]
        public sealed class ArgumentsForEntryPoint : TransformInputBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Output column: The suffix to append to the default column names",
                ShortName = "ex", SortOrder = 101)]
            public string Suffix;

            [Argument(ArgumentType.AtMostOnce, HelpText = "If specified, determines the permutation seed for applying this featurizer to a multiclass problem.",
                ShortName = "lps", SortOrder = 102)]
            public int LabelPermutationSeed;

            [Argument(ArgumentType.Required, HelpText = "Trainer to use", SortOrder = 10, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            public PredictorModel PredictorModel;
        }
#pragma warning restore CS0649

        internal const string TreeEnsembleSummary =
            "Trains a tree ensemble, or loads it from a file, then maps a numeric feature vector " +
            "to three outputs: 1. A vector containing the individual tree outputs of the tree ensemble. " +
            "2. A vector indicating the leaves that the feature vector falls on in the tree ensemble. " +
            "3. A vector indicating the paths that the feature vector falls on in the tree ensemble. " +
            "If a both a model file and a trainer are specified - will use the model file. If neither are specified, " +
            "will train a default FastTree model. " +
            "This can handle key labels by training a regression model towards their optionally permuted indices.";

        internal const string UserName = "Tree Ensemble Featurization Transform";

        // Factory method for SignatureDataScorer.
        private static IDataScorerTransform Create(IHostEnvironment env,
            TreeEnsembleFeaturizerBindableMapper.Arguments args, IDataView data, ISchemaBoundMapper mapper, RoleMappedSchema trainSchema)
        {
            return new GenericScorer(env, args, data, mapper, trainSchema);
        }

        // Factory method for SignatureBindableMapper.
        private static ISchemaBindableMapper Create(IHostEnvironment env,
            TreeEnsembleFeaturizerBindableMapper.Arguments args, IPredictor predictor)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(predictor, nameof(predictor));

            return new TreeEnsembleFeaturizerBindableMapper(env, args, predictor);
        }

        // Factory method for SignatureLoadModel.
        private static ISchemaBindableMapper Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            return new TreeEnsembleFeaturizerBindableMapper(env, ctx);
        }

        // Factory method for SignatureDataTransform.
        private static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("Tree Featurizer Transform");

            host.CheckValue(args, nameof(args));
            host.CheckValue(input, nameof(input));
            host.CheckUserArg(!string.IsNullOrWhiteSpace(args.TrainedModelFile) || args.Trainer != null, nameof(args.TrainedModelFile),
                "Please specify either a trainer or an input model file.");
            host.CheckUserArg(!string.IsNullOrEmpty(args.FeatureColumn), nameof(args.FeatureColumn), "Transform needs an input features column");

            IDataTransform xf;
            using (var ch = host.Start("Create Tree Ensemble Scorer"))
            {
                var scorerArgs = new TreeEnsembleFeaturizerBindableMapper.Arguments() {
                    Suffix = args.Suffix, TreesColumnName = "Trees", LeavesColumnName = "Leaves", PathsColumnName = "Paths" };
                if (!string.IsNullOrWhiteSpace(args.TrainedModelFile))
                {
                    if (args.Trainer != null)
                        ch.Warning("Both an input model and a trainer were specified. Using the model file.");

                    ch.Trace("Loading model");
                    IPredictor predictor;
                    using (Stream strm = new FileStream(args.TrainedModelFile, FileMode.Open, FileAccess.Read, FileShare.Read))
                    using (var rep = RepositoryReader.Open(strm, ch))
                        ModelLoadContext.LoadModel<IPredictor, SignatureLoadModel>(host, out predictor, rep, ModelFileUtils.DirPredictor);

                    ch.Trace("Creating scorer");
                    var data = TrainAndScoreTransformer.CreateDataFromArgs(ch, input, args);
                    Contracts.Assert(data.Schema.Feature.HasValue);

                    // Make sure that the given predictor has the correct number of input features.
                    if (predictor is IWeaklyTypedCalibratedModelParameters calibrated)
                        predictor = calibrated.WeaklyTypedSubModel;
                    // Predictor should be a TreeEnsembleModelParameters, which implements IValueMapper, so this should
                    // be non-null.
                    var vm = predictor as IValueMapper;
                    ch.CheckUserArg(vm != null, nameof(args.TrainedModelFile), "Predictor in model file does not have compatible type");
                    if (vm.InputType.GetVectorSize() != data.Schema.Feature.Value.Type.GetVectorSize())
                    {
                        throw ch.ExceptUserArg(nameof(args.TrainedModelFile),
                            "Predictor in model file expects {0} features, but data has {1} features",
                            vm.InputType.GetVectorSize(), data.Schema.Feature.Value.Type.GetVectorSize());
                    }

                    ISchemaBindableMapper bindable = new TreeEnsembleFeaturizerBindableMapper(env, scorerArgs, predictor);
                    var bound = bindable.Bind(env, data.Schema);
                    xf = new GenericScorer(env, scorerArgs, input, bound, data.Schema);
                }
                else
                {
                    ch.AssertValue(args.Trainer);

                    ch.Trace("Creating TrainAndScoreTransform");

                    var trainScoreArgs = new TrainAndScoreTransformer.Arguments();
                    args.CopyTo(trainScoreArgs);
                    trainScoreArgs.Trainer = args.Trainer;

                    trainScoreArgs.Scorer = ComponentFactoryUtils.CreateFromFunction<IDataView, ISchemaBoundMapper, RoleMappedSchema, IDataScorerTransform>(
                        (e, data, mapper, trainSchema) => Create(e, scorerArgs, data, mapper, trainSchema));

                    var mapperFactory = ComponentFactoryUtils.CreateFromFunction<IPredictor, ISchemaBindableMapper>(
                            (e, predictor) => new TreeEnsembleFeaturizerBindableMapper(e, scorerArgs, predictor));

                    var labelInput = AppendLabelTransform(host, ch, input, trainScoreArgs.LabelColumn, args.LabelPermutationSeed);
                    var scoreXf = TrainAndScoreTransformer.Create(host, trainScoreArgs, labelInput, mapperFactory);

                    if (input == labelInput)
                        return scoreXf;
                    return (IDataTransform)ApplyTransformUtils.ApplyAllTransformsToData(host, scoreXf, input, labelInput);
                }
            }
            return xf;
        }

        public static IDataTransform CreateForEntryPoint(IHostEnvironment env, ArgumentsForEntryPoint args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("Tree Featurizer Transform");

            host.CheckValue(args, nameof(args));
            host.CheckValue(input, nameof(input));
            host.CheckUserArg(args.PredictorModel != null, nameof(args.PredictorModel), "Please specify a predictor model.");

            using (var ch = host.Start("Create Tree Ensemble Scorer"))
            {
                var scorerArgs = new TreeEnsembleFeaturizerBindableMapper.Arguments() {
                    Suffix = args.Suffix, TreesColumnName = "Trees", LeavesColumnName = "Leaves", PathsColumnName = "Paths" };
                var predictor = args.PredictorModel.Predictor;
                ch.Trace("Prepare data");
                RoleMappedData data = null;
                args.PredictorModel.PrepareData(env, input, out data, out var predictor2);
                ch.AssertValue(data);
                ch.Assert(data.Schema.Feature.HasValue);
                ch.Assert(predictor == predictor2);

                // Make sure that the given predictor has the correct number of input features.
                if (predictor is CalibratedModelParametersBase<IPredictorProducing<float>, Calibrators.ICalibrator>)
                    predictor = ((CalibratedModelParametersBase<IPredictorProducing<float>, Calibrators.ICalibrator>)predictor).SubModel;
                // Predictor should be a TreeEnsembleModelParameters, which implements IValueMapper, so this should
                // be non-null.
                var vm = predictor as IValueMapper;
                ch.CheckUserArg(vm != null, nameof(args.PredictorModel), "Predictor does not have compatible type");
                if (data != null && vm.InputType.GetVectorSize() != data.Schema.Feature.Value.Type.GetVectorSize())
                {
                    throw ch.ExceptUserArg(nameof(args.PredictorModel),
                        "Predictor expects {0} features, but data has {1} features",
                        vm.InputType.GetVectorSize(), data.Schema.Feature.Value.Type.GetVectorSize());
                }

                ISchemaBindableMapper bindable = new TreeEnsembleFeaturizerBindableMapper(env, scorerArgs, predictor);
                var bound = bindable.Bind(env, data.Schema);
                return new GenericScorer(env, scorerArgs, data.Data, bound, data.Schema);
            }
        }

        private static IDataView AppendFloatMapper<TInput>(IHostEnvironment env, IChannel ch, IDataView input,
            string col, KeyDataViewType type, int seed)
        {
            // Any key is convertible to ulong, so rather than add special case handling for all possible
            // key-types we just upfront convert it to the most general type (ulong) and work from there.
            KeyDataViewType dstType = new KeyDataViewType(typeof(ulong), type.Count);
            bool identity;
            var converter = Conversions.Instance.GetStandardConversion<TInput, ulong>(type, dstType, out identity);
            var isNa = Conversions.Instance.GetIsNAPredicate<TInput>(type);

            ValueMapper<TInput, Single> mapper;
            if (seed == 0)
            {
                mapper =
                    (in TInput src, ref Single dst) =>
                    {
                        //Attention: This method is called from multiple threads.
                        //Do not move the temp variable outside this method.
                        //If you do, the variable is shared between the threads and results in a race condition.
                        ulong temp = 0;
                        if (isNa(in src))
                        {
                            dst = Single.NaN;
                            return;
                        }
                        converter(in src, ref temp);
                        dst = (Single)temp - 1;
                    };
            }
            else
            {
                ch.Check(type.Count > 0, "Label must be of known cardinality.");
                int[] permutation = Utils.GetRandomPermutation(RandomUtils.Create(seed), type.GetCountAsInt32(env));
                mapper =
                    (in TInput src, ref Single dst) =>
                    {
                        //Attention: This method is called from multiple threads.
                        //Do not move the temp variable outside this method.
                        //If you do, the variable is shared between the threads and results in a race condition.
                        ulong temp = 0;
                        if (isNa(in src))
                        {
                            dst = Single.NaN;
                            return;
                        }
                        converter(in src, ref temp);
                        dst = (Single)permutation[(int)(temp - 1)];
                    };
            }

            return LambdaColumnMapper.Create(env, "Key to Float Mapper", input, col, col, type, NumberDataViewType.Single, mapper);
        }

        private static IDataView AppendLabelTransform(IHostEnvironment env, IChannel ch, IDataView input, string labelName, int labelPermutationSeed)
        {
            Contracts.AssertValue(env);
            env.AssertValue(ch);
            ch.AssertValue(input);
            ch.AssertNonWhiteSpace(labelName);

            var col = input.Schema.GetColumnOrNull(labelName);
            if (!col.HasValue)
                throw ch.ExceptSchemaMismatch(nameof(input), "label", labelName);

            DataViewType labelType = col.Value.Type;
            if (!(labelType is KeyDataViewType))
            {
                if (labelPermutationSeed != 0)
                    ch.Warning(
                        "labelPermutationSeed != 0 only applies on a multi-class learning problem when the label type is a key.");
                return input;
            }
            return Utils.MarshalInvoke(AppendFloatMapper<int>, labelType.RawType, env, ch, input, labelName, (KeyDataViewType)labelType,
                labelPermutationSeed);
        }
    }

    internal static partial class TreeFeaturize
    {
#pragma warning disable CS0649 // The fields will still be set via the reflection driven mechanisms.
        [TlcModule.EntryPoint(Name = "Transforms.TreeLeafFeaturizer",
            Desc = TreeEnsembleFeaturizerTransform.TreeEnsembleSummary,
            UserName = TreeEnsembleFeaturizerTransform.UserName,
            ShortName = TreeEnsembleFeaturizerBindableMapper.LoadNameShort)]
        public static CommonOutputs.TransformOutput Featurizer(IHostEnvironment env, TreeEnsembleFeaturizerTransform.ArgumentsForEntryPoint input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TreeFeaturizerTransform");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var xf = TreeEnsembleFeaturizerTransform.CreateForEntryPoint(env, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModelImpl(env, xf, input.Data), OutputData = xf };
        }
#pragma warning restore CS0649
    }
}
