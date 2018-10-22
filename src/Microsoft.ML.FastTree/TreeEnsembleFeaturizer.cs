// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.Conversion;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.FastTree;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.IO;

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

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// A bindable mapper wrapper for tree ensembles, that creates a bound mapper with three outputs:
    /// 1. A vector containing the individual tree outputs of the tree ensemble.
    /// 2. An indicator vector for the leaves that the feature vector falls on in the tree ensemble.
    /// 3. An indicator vector for the internal nodes on the paths that the feature vector falls on in the tree ensemble.
    /// </summary>
    public sealed class TreeEnsembleFeaturizerBindableMapper : ISchemaBindableMapper, ICanSaveModel
    {
        public static class OutputColumnNames
        {
            public const string Trees = "Trees";
            public const string Paths = "Paths";
            public const string Leaves = "Leaves";
        }

        public sealed class Arguments : ScorerArgumentsBase
        {
        }

        private sealed class BoundMapper : ISchemaBoundRowMapper
        {
            private sealed class SchemaImpl : ISchema
            {
                private readonly IExceptionContext _ectx;
                private readonly string[] _names;
                private readonly ColumnType[] _types;

                private readonly TreeEnsembleFeaturizerBindableMapper _parent;

                public int ColumnCount { get { return _types.Length; } }

                public SchemaImpl(IExceptionContext ectx, TreeEnsembleFeaturizerBindableMapper parent,
                    ColumnType treeValueColType, ColumnType leafIdColType, ColumnType pathIdColType)
                {
                    Contracts.CheckValueOrNull(ectx);
                    _ectx = ectx;
                    _ectx.AssertValue(parent);
                    _ectx.AssertValue(treeValueColType);
                    _ectx.AssertValue(leafIdColType);
                    _ectx.AssertValue(pathIdColType);

                    _parent = parent;

                    _names = new string[3];
                    _names[TreeIdx] = OutputColumnNames.Trees;
                    _names[LeafIdx] = OutputColumnNames.Leaves;
                    _names[PathIdx] = OutputColumnNames.Paths;

                    _types = new ColumnType[3];
                    _types[TreeIdx] = treeValueColType;
                    _types[LeafIdx] = leafIdColType;
                    _types[PathIdx] = pathIdColType;
                }

                public bool TryGetColumnIndex(string name, out int col)
                {
                    col = -1;
                    if (name == OutputColumnNames.Trees)
                        col = TreeIdx;
                    else if (name == OutputColumnNames.Leaves)
                        col = LeafIdx;
                    else if (name == OutputColumnNames.Paths)
                        col = PathIdx;
                    return col >= 0;
                }

                public string GetColumnName(int col)
                {
                    _ectx.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                    return _names[col];
                }

                public ColumnType GetColumnType(int col)
                {
                    _ectx.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                    return _types[col];
                }

                public IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypes(int col)
                {
                    _ectx.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                    yield return
                        MetadataUtils.GetNamesType(_types[col].VectorSize).GetPair(MetadataUtils.Kinds.SlotNames);
                    if (col == PathIdx || col == LeafIdx)
                        yield return BoolType.Instance.GetPair(MetadataUtils.Kinds.IsNormalized);
                }

                public ColumnType GetMetadataTypeOrNull(string kind, int col)
                {
                    _ectx.CheckParam(0 <= col && col < ColumnCount, nameof(col));

                    if ((col == PathIdx || col == LeafIdx) && kind == MetadataUtils.Kinds.IsNormalized)
                        return BoolType.Instance;
                    if (kind == MetadataUtils.Kinds.SlotNames)
                        return MetadataUtils.GetNamesType(_types[col].VectorSize);
                    return null;
                }

                public void GetMetadata<TValue>(string kind, int col, ref TValue value)
                {
                    _ectx.CheckParam(0 <= col && col < ColumnCount, nameof(col));

                    if ((col == PathIdx || col == LeafIdx) && kind == MetadataUtils.Kinds.IsNormalized)
                        MetadataUtils.Marshal<bool, TValue>(IsNormalized, col, ref value);
                    else if (kind == MetadataUtils.Kinds.SlotNames)
                    {
                        switch (col)
                        {
                            case TreeIdx:
                                MetadataUtils.Marshal<VBuffer<ReadOnlyMemory<char>>, TValue>(_parent.GetTreeSlotNames, col, ref value);
                                break;
                            case LeafIdx:
                                MetadataUtils.Marshal<VBuffer<ReadOnlyMemory<char>>, TValue>(_parent.GetLeafSlotNames, col, ref value);
                                break;
                            default:
                                Contracts.Assert(col == PathIdx);
                                MetadataUtils.Marshal<VBuffer<ReadOnlyMemory<char>>, TValue>(_parent.GetPathSlotNames, col, ref value);
                                break;
                        }
                    }
                    else
                        throw _ectx.ExceptGetMetadata();
                }

                private void IsNormalized(int iinfo, ref bool dst)
                {
                    dst = true;
                }
            }

            private const int TreeIdx = 0;
            private const int LeafIdx = 1;
            private const int PathIdx = 2;

            private readonly TreeEnsembleFeaturizerBindableMapper _owner;
            private readonly IExceptionContext _ectx;

            public RoleMappedSchema InputRoleMappedSchema { get; }

            public Schema Schema { get; }
            public Schema InputSchema => InputRoleMappedSchema.Schema;

            public ISchemaBindableMapper Bindable => _owner;

            public BoundMapper(IExceptionContext ectx, TreeEnsembleFeaturizerBindableMapper owner,
                RoleMappedSchema schema)
            {
                Contracts.AssertValue(ectx);
                ectx.AssertValue(owner);
                ectx.AssertValue(schema);
                ectx.AssertValue(schema.Feature);

                _ectx = ectx;

                _owner = owner;
                InputRoleMappedSchema = schema;

                // A vector containing the output of each tree on a given example.
                var treeValueType = new VectorType(NumberType.Float, _owner._ensemble.NumTrees);
                // An indicator vector with length = the total number of leaves in the ensemble, indicating which leaf the example
                // ends up in all the trees in the ensemble.
                var leafIdType = new VectorType(NumberType.Float, _owner._totalLeafCount);
                // An indicator vector with length = the total number of nodes in the ensemble, indicating the nodes on
                // the paths of the example in all the trees in the ensemble.
                // The total number of nodes in a binary tree is equal to the number of internal nodes + the number of leaf nodes,
                // and it is also equal to the number of children of internal nodes (which is 2 * the number of internal nodes)
                // plus one (since the root node is not a child of any node). So we have #internal + #leaf = 2*(#internal) + 1,
                // which means that #internal = #leaf - 1.
                // Therefore, the number of internal nodes in the ensemble is #leaf - #trees.
                var pathIdType = new VectorType(NumberType.Float, _owner._totalLeafCount - _owner._ensemble.NumTrees);
                Schema = Schema.Create(new SchemaImpl(ectx, owner, treeValueType, leafIdType, pathIdType));
            }

            public IRow GetRow(IRow input, Func<int, bool> predicate, out Action disposer)
            {
                _ectx.CheckValue(input, nameof(input));
                _ectx.CheckValue(predicate, nameof(predicate));
                disposer = null;
                return new SimpleRow(Schema, input, CreateGetters(input, predicate));
            }

            private Delegate[] CreateGetters(IRow input, Func<int, bool> predicate)
            {
                _ectx.AssertValue(input);
                _ectx.AssertValue(predicate);

                var delegates = new Delegate[3];

                var treeValueActive = predicate(TreeIdx);
                var leafIdActive = predicate(LeafIdx);
                var pathIdActive = predicate(PathIdx);

                if (!treeValueActive && !leafIdActive && !pathIdActive)
                    return delegates;

                var state = new State(_ectx, input, _owner._ensemble, _owner._totalLeafCount, InputRoleMappedSchema.Feature.Index);

                // Get the tree value getter.
                if (treeValueActive)
                {
                    ValueGetter<VBuffer<float>> fn = state.GetTreeValues;
                    delegates[TreeIdx] = fn;
                }

                // Get the leaf indicator getter.
                if (leafIdActive)
                {
                    ValueGetter<VBuffer<float>> fn = state.GetLeafIds;
                    delegates[LeafIdx] = fn;
                }

                // Get the path indicators getter.
                if (pathIdActive)
                {
                    ValueGetter<VBuffer<float>> fn = state.GetPathIds;
                    delegates[PathIdx] = fn;
                }

                return delegates;
            }

            private sealed class State
            {
                private readonly IExceptionContext _ectx;
                private readonly IRow _input;
                private readonly FastTreePredictionWrapper _ensemble;
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

                public State(IExceptionContext ectx, IRow input, FastTreePredictionWrapper ensemble, int numLeaves, int featureIndex)
                {
                    Contracts.AssertValue(ectx);
                    _ectx = ectx;
                    _ectx.AssertValue(input);
                    _ectx.AssertValue(ensemble);
                    _ectx.Assert(ensemble.NumTrees > 0);
                    _input = input;
                    _ensemble = ensemble;
                    _numTrees = _ensemble.NumTrees;
                    _numLeaves = numLeaves;

                    _src = default(VBuffer<float>);
                    _featureGetter = input.GetGetter<VBuffer<float>>(featureIndex);

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
                    var vals = dst.Values;
                    if (Utils.Size(vals) < _numTrees)
                        vals = new float[_numTrees];

                    for (int i = 0; i < _numTrees; i++)
                        vals[i] = _ensemble.GetLeafValue(i, _leafIds[i]);

                    dst = new VBuffer<float>(_numTrees, vals, dst.Indices);
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
                        var trees = _ensemble.GetTrees();
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

                        var trees = _ensemble.GetTrees();
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
                    _ectx.Check(_input.Position >= 0, "Getter was called on an inactive cursor.");
                    if (_cachedPosition != _input.Position)
                    {
                        _featureGetter(ref _src);

                        _ectx.Assert(Utils.Size(_leafIds) == _numTrees);
                        _ectx.Assert(Utils.Size(_pathIds) == _numTrees);

                        for (int i = 0; i < _numTrees; i++)
                            _leafIds[i] = _ensemble.GetLeaf(i, ref _src, ref _pathIds[i]);

                        _cachedPosition = _input.Position;
                    }
                }
            }

            public IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> GetInputColumnRoles()
            {
                yield return RoleMappedSchema.ColumnRole.Feature.Bind(InputRoleMappedSchema.Feature.Name);
            }

            public Func<int, bool> GetDependencies(Func<int, bool> predicate)
            {
                for (int i = 0; i < Schema.ColumnCount; i++)
                {
                    if (predicate(i))
                        return col => col == InputRoleMappedSchema.Feature.Index;
                }
                return col => false;
            }
        }

        public const string LoadNameShort = "TreeFeat";
        public const string LoaderSignature = "TreeEnsembleMapper";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TREEMAPR",
                // verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Add _defaultValueForMissing
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(TreeEnsembleFeaturizerBindableMapper).Assembly.FullName);
        }

        private readonly IHost _host;
        private readonly FastTreePredictionWrapper _ensemble;
        private readonly int _totalLeafCount;

        public TreeEnsembleFeaturizerBindableMapper(IHostEnvironment env, Arguments args, IPredictor predictor)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(LoaderSignature);
            _host.CheckValue(args, nameof(args));
            _host.CheckValue(predictor, nameof(predictor));

            if (predictor is CalibratedPredictorBase)
                predictor = ((CalibratedPredictorBase)predictor).SubPredictor;
            _ensemble = predictor as FastTreePredictionWrapper;
            _host.Check(_ensemble != null, "Predictor in model file does not have compatible type");

            _totalLeafCount = CountLeaves(_ensemble);
        }

        public TreeEnsembleFeaturizerBindableMapper(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(LoaderSignature);
            _host.AssertValue(ctx);

            // *** Binary format ***
            // ensemble

            ctx.LoadModel<FastTreePredictionWrapper, SignatureLoadModel>(env, out _ensemble, "Ensemble");
            _totalLeafCount = CountLeaves(_ensemble);
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // ensemble

            _host.AssertValue(_ensemble);
            ctx.SaveModel(_ensemble, "Ensemble");
        }

        private static int CountLeaves(FastTreePredictionWrapper ensemble)
        {
            Contracts.AssertValue(ensemble);

            var trees = ensemble.GetTrees();
            var numTrees = trees.Length;
            var totalLeafCount = 0;
            for (int i = 0; i < numTrees; i++)
                totalLeafCount += trees[i].NumLeaves;
            return totalLeafCount;
        }

        private void GetTreeSlotNames(int col, ref VBuffer<ReadOnlyMemory<char>> dst)
        {
            var numTrees = _ensemble.NumTrees;

            var names = dst.Values;
            if (Utils.Size(names) < numTrees)
                names = new ReadOnlyMemory<char>[numTrees];

            for (int t = 0; t < numTrees; t++)
                names[t] = string.Format("Tree{0:000}", t).AsMemory();

            dst = new VBuffer<ReadOnlyMemory<char>>(numTrees, names, dst.Indices);
        }

        private void GetLeafSlotNames(int col, ref VBuffer<ReadOnlyMemory<char>> dst)
        {
            var numTrees = _ensemble.NumTrees;

            var names = dst.Values;
            if (Utils.Size(names) < _totalLeafCount)
                names = new ReadOnlyMemory<char>[_totalLeafCount];

            int i = 0;
            int t = 0;
            foreach (var tree in _ensemble.GetTrees())
            {
                for (int l = 0; l < tree.NumLeaves; l++)
                    names[i++] = string.Format("Tree{0:000}Leaf{1:000}", t, l).AsMemory();
                t++;
            }
            _host.Assert(i == _totalLeafCount);
            dst = new VBuffer<ReadOnlyMemory<char>>(_totalLeafCount, names, dst.Indices);
        }

        private void GetPathSlotNames(int col, ref VBuffer<ReadOnlyMemory<char>> dst)
        {
            var numTrees = _ensemble.NumTrees;

            var totalNodeCount = _totalLeafCount - numTrees;
            var names = dst.Values;
            if (Utils.Size(names) < totalNodeCount)
                names = new ReadOnlyMemory<char>[totalNodeCount];

            int i = 0;
            int t = 0;
            foreach (var tree in _ensemble.GetTrees())
            {
                var numLeaves = tree.NumLeaves;
                for (int l = 0; l < tree.NumLeaves - 1; l++)
                    names[i++] = string.Format("Tree{0:000}Node{1:000}", t, l).AsMemory();
                t++;
            }
            _host.Assert(i == totalNodeCount);
            dst = new VBuffer<ReadOnlyMemory<char>>(totalNodeCount, names, dst.Indices);
        }

        public ISchemaBoundMapper Bind(IHostEnvironment env, RoleMappedSchema schema)
        {
            Contracts.AssertValue(env);
            env.AssertValue(schema);
            env.CheckParam(schema.Feature != null, nameof(schema), "Need a feature column");

            return new BoundMapper(env, this, schema);
        }
    }

    /// <include file='doc.xml' path='doc/members/member[@name="TreeEnsembleFeaturizerTransform"]'/>
    public static class TreeEnsembleFeaturizerTransform
    {
        public sealed class Arguments : TrainAndScoreTransform.ArgumentsBase
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
            public IPredictorModel PredictorModel;
        }

        internal const string TreeEnsembleSummary =
            "Trains a tree ensemble, or loads it from a file, then maps a numeric feature vector " +
            "to three outputs: 1. A vector containing the individual tree outputs of the tree ensemble. " +
            "2. A vector indicating the leaves that the feature vector falls on in the tree ensemble. " +
            "3. A vector indicating the paths that the feature vector falls on in the tree ensemble. " +
            "If a both a model file and a trainer are specified - will use the model file. If neither are specified, " +
            "will train a default FastTree model. " +
            "This can handle key labels by training a regression model towards their optionally permuted indices.";

        internal const string UserName = "Tree Ensemble Featurization Transform";

        public static IDataScorerTransform Create(IHostEnvironment env,
            TreeEnsembleFeaturizerBindableMapper.Arguments args, IDataView data, ISchemaBoundMapper mapper, RoleMappedSchema trainSchema)
        {
            return new GenericScorer(env, args, data, mapper, trainSchema);
        }

        public static ISchemaBindableMapper Create(IHostEnvironment env,
            TreeEnsembleFeaturizerBindableMapper.Arguments args, IPredictor predictor)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(predictor, nameof(predictor));

            return new TreeEnsembleFeaturizerBindableMapper(env, args, predictor);
        }

        public static ISchemaBindableMapper Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            return new TreeEnsembleFeaturizerBindableMapper(env, ctx);
        }

        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
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
                var scorerArgs = new TreeEnsembleFeaturizerBindableMapper.Arguments() { Suffix = args.Suffix };
                if (!string.IsNullOrWhiteSpace(args.TrainedModelFile))
                {
                    if (args.Trainer != null)
                        ch.Warning("Both an input model and a trainer were specified. Using the model file.");

                    ch.Trace("Loading model");
                    IPredictor predictor;
                    using (Stream strm = new FileStream(args.TrainedModelFile, FileMode.Open, FileAccess.Read))
                    using (var rep = RepositoryReader.Open(strm, ch))
                        ModelLoadContext.LoadModel<IPredictor, SignatureLoadModel>(host, out predictor, rep, ModelFileUtils.DirPredictor);

                    ch.Trace("Creating scorer");
                    var data = TrainAndScoreTransform.CreateDataFromArgs(ch, input, args);

                    // Make sure that the given predictor has the correct number of input features.
                    if (predictor is CalibratedPredictorBase)
                        predictor = ((CalibratedPredictorBase)predictor).SubPredictor;
                    // Predictor should be a FastTreePredictionWrapper, which implements IValueMapper, so this should
                    // be non-null.
                    var vm = predictor as IValueMapper;
                    ch.CheckUserArg(vm != null, nameof(args.TrainedModelFile), "Predictor in model file does not have compatible type");
                    if (vm.InputType.VectorSize != data.Schema.Feature.Type.VectorSize)
                    {
                        throw ch.ExceptUserArg(nameof(args.TrainedModelFile),
                            "Predictor in model file expects {0} features, but data has {1} features",
                            vm.InputType.VectorSize, data.Schema.Feature.Type.VectorSize);
                    }

                    var bindable = new TreeEnsembleFeaturizerBindableMapper(env, scorerArgs, predictor);
                    var bound = bindable.Bind(env, data.Schema);
                    xf = new GenericScorer(env, scorerArgs, input, bound, data.Schema);
                }
                else
                {
                    ch.AssertValue(args.Trainer);

                    ch.Trace("Creating TrainAndScoreTransform");

                    var trainScoreArgs = new TrainAndScoreTransform.Arguments();
                    args.CopyTo(trainScoreArgs);
                    trainScoreArgs.Trainer = args.Trainer;

                    trainScoreArgs.Scorer = ComponentFactoryUtils.CreateFromFunction<IDataView, ISchemaBoundMapper, RoleMappedSchema, IDataScorerTransform>(
                        (e, data, mapper, trainSchema) => Create(e, scorerArgs, data, mapper, trainSchema));

                    var mapperFactory = ComponentFactoryUtils.CreateFromFunction<IPredictor, ISchemaBindableMapper>(
                            (e, predictor) => new TreeEnsembleFeaturizerBindableMapper(e, scorerArgs, predictor));

                    var labelInput = AppendLabelTransform(host, ch, input, trainScoreArgs.LabelColumn, args.LabelPermutationSeed);
                    var scoreXf = TrainAndScoreTransform.Create(host, trainScoreArgs, labelInput, mapperFactory);

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
                var scorerArgs = new TreeEnsembleFeaturizerBindableMapper.Arguments() { Suffix = args.Suffix };
                var predictor = args.PredictorModel.Predictor;
                ch.Trace("Prepare data");
                RoleMappedData data = null;
                args.PredictorModel.PrepareData(env, input, out data, out var predictor2);
                ch.AssertValue(data);
                ch.Assert(predictor == predictor2);

                // Make sure that the given predictor has the correct number of input features.
                if (predictor is CalibratedPredictorBase)
                    predictor = ((CalibratedPredictorBase)predictor).SubPredictor;
                // Predictor should be a FastTreePredictionWrapper, which implements IValueMapper, so this should
                // be non-null.
                var vm = predictor as IValueMapper;
                ch.CheckUserArg(vm != null, nameof(args.PredictorModel), "Predictor does not have compatible type");
                if (data != null && vm.InputType.VectorSize != data.Schema.Feature.Type.VectorSize)
                {
                    throw ch.ExceptUserArg(nameof(args.PredictorModel),
                        "Predictor expects {0} features, but data has {1} features",
                        vm.InputType.VectorSize, data.Schema.Feature.Type.VectorSize);
                }

                var bindable = new TreeEnsembleFeaturizerBindableMapper(env, scorerArgs, predictor);
                var bound = bindable.Bind(env, data.Schema);
               return new GenericScorer(env, scorerArgs, data.Data, bound, data.Schema);
            }
        }

        private static IDataView AppendFloatMapper<TInput>(IHostEnvironment env, IChannel ch, IDataView input,
            string col, KeyType type, int seed)
        {
            // Any key is convertible to ulong, so rather than add special case handling for all possible
            // key-types we just upfront convert it to the most general type (ulong) and work from there.
            KeyType dstType = new KeyType(DataKind.U8, type.Min, type.Count, type.Contiguous);
            bool identity;
            var converter = Conversions.Instance.GetStandardConversion<TInput, ulong>(type, dstType, out identity);
            var isNa = Conversions.Instance.GetIsNAPredicate<TInput>(type);
            ulong temp = 0;

            ValueMapper<TInput, Single> mapper;
            if (seed == 0)
            {
                mapper =
                    (ref TInput src, ref Single dst) =>
                    {
                        if (isNa(ref src))
                        {
                            dst = Single.NaN;
                            return;
                        }
                        converter(ref src, ref temp);
                        dst = (Single)(temp - 1);
                    };
            }
            else
            {
                ch.Check(type.Count > 0, "Label must be of known cardinality.");
                int[] permutation = Utils.GetRandomPermutation(RandomUtils.Create(seed), type.Count);
                mapper =
                    (ref TInput src, ref Single dst) =>
                    {
                        if (isNa(ref src))
                        {
                            dst = Single.NaN;
                            return;
                        }
                        converter(ref src, ref temp);
                        dst = (Single)permutation[(int)(temp - 1)];
                    };
            }

            return LambdaColumnMapper.Create(env, "Key to Float Mapper", input, col, col, type, NumberType.Float, mapper);
        }

        private static IDataView AppendLabelTransform(IHostEnvironment env, IChannel ch, IDataView input, string labelName, int labelPermutationSeed)
        {
            Contracts.AssertValue(env);
            env.AssertValue(ch);
            ch.AssertValue(input);
            ch.AssertNonWhiteSpace(labelName);

            int col;
            if (!input.Schema.TryGetColumnIndex(labelName, out col))
                throw ch.Except("Label column '{0}' not found.", labelName);
            ColumnType labelType = input.Schema.GetColumnType(col);
            if (!labelType.IsKey)
            {
                if (labelPermutationSeed != 0)
                    ch.Warning(
                        "labelPermutationSeed != 0 only applies on a multi-class learning problem when the label type is a key.");
                return input;
            }
            return Utils.MarshalInvoke(AppendFloatMapper<int>, labelType.RawType, env, ch, input, labelName, labelType.AsKey,
                labelPermutationSeed);
        }
    }

    public static partial class TreeFeaturize
    {
        [TlcModule.EntryPoint(Name = "Transforms.TreeLeafFeaturizer",
            Desc = TreeEnsembleFeaturizerTransform.TreeEnsembleSummary,
            UserName = TreeEnsembleFeaturizerTransform.UserName,
            ShortName = TreeEnsembleFeaturizerBindableMapper.LoadNameShort,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.FastTree/doc.xml' path='doc/members/member[@name=""TreeEnsembleFeaturizerTransform""]/*'/>" })]
        public static CommonOutputs.TransformOutput Featurizer(IHostEnvironment env, TreeEnsembleFeaturizerTransform.ArgumentsForEntryPoint input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TreeFeaturizerTransform");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var xf = TreeEnsembleFeaturizerTransform.CreateForEntryPoint(env, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModel(env, xf, input.Data), OutputData = xf };
        }
    }
}
