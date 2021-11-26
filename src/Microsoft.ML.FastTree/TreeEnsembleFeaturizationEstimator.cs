// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Trainers.FastTree
{
    /// <summary>
    /// This class encapsulates the common behavior of all tree-based featurizers such as <see cref="FastTreeBinaryFeaturizationEstimator"/>,
    /// <see cref="FastForestBinaryFeaturizationEstimator"/>, <see cref="FastTreeRegressionFeaturizationEstimator"/>,
    /// <see cref="FastForestRegressionFeaturizationEstimator"/>, and <see cref="PretrainedTreeFeaturizationEstimator"/>.
    /// All tree-based featurizers share the same output schema computed by <see cref="GetOutputSchema(SchemaShape)"/>. All tree-based featurizers
    /// requires an input feature column name and a suffix for all output columns. The <see cref="ITransformer"/> returned by <see cref="Fit(IDataView)"/>
    /// produces three columns: (1) the prediction values of all trees, (2) the IDs of leaves the input feature vector falling into, and (3)
    /// the binary vector which encodes the paths to those destination leaves.
    /// </summary>
    public abstract class TreeEnsembleFeaturizationEstimatorBase : IEstimator<TreeEnsembleFeaturizationTransformer>
    {
        /// <summary>
        /// The common options of tree-based featurizations such as <see cref="FastTreeBinaryFeaturizationEstimator"/>, <see cref="FastForestBinaryFeaturizationEstimator"/>,
        /// <see cref="FastTreeRegressionFeaturizationEstimator"/>, <see cref="FastForestRegressionFeaturizationEstimator"/>, and <see cref="PretrainedTreeFeaturizationEstimator"/>.
        /// </summary>
        public abstract class OptionsBase
        {
            /// <summary>
            /// The name of feature column in the <see cref="IDataView"/> when calling <see cref="Fit(IDataView)"/>.
            /// The column type must be a vector of <see cref="System.Single"/>. The column called <see cref="InputColumnName"/> would be mapped
            /// to columns called <see cref="TreesColumnName"/>, <see cref="LeavesColumnName"/>, and <see cref="PathsColumnName"/> in the output
            /// of <see cref="TreeEnsembleFeaturizationEstimatorBase"/> and its derived classes. Note that <see cref="FeatureColumnName"/> is not
            /// necessary to be the same as the feature column used to train the underlying tree model.
            /// </summary>
            public string InputColumnName;

            /// <summary>
            /// The name of the column that stores the prediction values of all trees. Its type is a vector of <see cref="System.Single"/>
            /// and the i-th vector element is the prediction value predicted by the i-th tree.
            /// If <see cref="TreesColumnName"/> is <see langword="null"/>, this output column may not be generated.
            /// </summary>
            public string TreesColumnName;

            /// <summary>
            /// The 0-1 encoding of all leaf nodes' IDs. Its type is a vector of <see cref="System.Single"/>. If the given feature
            /// vector falls into the first leaf of the first tree, the first element in the 0-1 encoding would be 1.
            /// If <see cref="LeavesColumnName"/> is <see langword="null"/>, this output column may not be generated.
            /// </summary>
            public string LeavesColumnName;

            /// <summary>
            /// The 0-1 encoding of the paths to the leaves. If the path to the first tree's leaf is node 1 (2nd node in the first tree),
            /// node 3 (4th node in the first tree), and node 5 (6th node in the first tree), the 2nd, 4th, and 6th element in that encoding
            /// would be 1.
            /// If <see cref="PathsColumnName"/> is <see langword="null"/>, this output column may not be generated.
            /// </summary>
            public string PathsColumnName;
        };

        /// <summary>
        /// See <see cref="OptionsBase.InputColumnName"/>.
        /// </summary>
        private protected readonly string FeatureColumnName;

        /// <summary>
        /// See <see cref="OptionsBase.TreesColumnName"/>.
        /// </summary>
        private protected readonly string TreesColumnName;

        /// <summary>
        /// See <see cref="OptionsBase.LeavesColumnName"/>.
        /// </summary>
        private protected readonly string LeavesColumnName;

        /// <summary>
        /// See <see cref="OptionsBase.PathsColumnName"/>.
        /// </summary>
        private protected readonly string PathsColumnName;

        /// <summary>
        /// Environment of this instance. It controls error throwing and other environment settings.
        /// </summary>
        private protected readonly IHostEnvironment Env;

        private protected TreeEnsembleFeaturizationEstimatorBase(IHostEnvironment env, OptionsBase options)
        {
            Env = env;
            if (options.InputColumnName == null)
                throw Env.Except(nameof(options), "The " + nameof(options.InputColumnName) + " cannot be null.");
            if (options.TreesColumnName == null && options.LeavesColumnName == null && options.PathsColumnName == null)
                throw Env.Except($"{nameof(OptionsBase.TreesColumnName)}, {nameof(OptionsBase.LeavesColumnName)}, and {nameof(OptionsBase.PathsColumnName)} cannot be all null at the same time. " +
                    $"At least one output column name should be provided so at least one output column may be generated.");

            FeatureColumnName = options.InputColumnName;
            TreesColumnName = options.TreesColumnName;
            LeavesColumnName = options.LeavesColumnName;
            PathsColumnName = options.PathsColumnName;
        }

        /// <summary>
        /// All derived class should implement <see cref="PrepareModel(IDataView)"/> to tell how to get a <see cref="TreeEnsembleModelParameters"/>
        /// out from <paramref name="input"/> and parameters inside this or derived classes.
        /// </summary>
        /// <param name="input">Data used to train a tree model.</param>
        /// <returns>The trees used in <see cref="TreeEnsembleFeaturizationTransformer"/>.</returns>
        private protected abstract TreeEnsembleModelParameters PrepareModel(IDataView input);

        /// <summary>
        /// Produce a <see cref="TreeEnsembleModelParameters"/> which maps the column called <see cref="OptionsBase.InputColumnName"/> in <paramref name="input"/>
        /// to three output columns.
        /// </summary>
        public TreeEnsembleFeaturizationTransformer Fit(IDataView input)
        {
            var model = PrepareModel(input);
            return new TreeEnsembleFeaturizationTransformer(Env, input.Schema, input.Schema[FeatureColumnName], model,
                TreesColumnName, LeavesColumnName, PathsColumnName);
        }

        /// <summary>
        /// <see cref="PretrainedTreeFeaturizationEstimator"/> adds three float-vector columns into <paramref name="inputSchema"/>.
        /// Given a feature vector column, the added columns are the prediction values of all trees, the leaf IDs the feature
        /// vector falls into, and the paths to those leaves.
        /// </summary>
        /// <param name="inputSchema">A schema which contains a feature column. Note that feature column name can be specified
        /// by <see cref="OptionsBase.InputColumnName"/>.</param>
        /// <returns>Output <see cref="SchemaShape"/> produced by <see cref="PretrainedTreeFeaturizationEstimator"/>.</returns>
        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Env.CheckValue(inputSchema, nameof(inputSchema));

            if (!inputSchema.TryFindColumn(FeatureColumnName, out var col))
                throw Env.ExceptSchemaMismatch(nameof(inputSchema), "input", FeatureColumnName);

            var result = inputSchema.ToDictionary(x => x.Name);

            if (TreesColumnName != null)
                result[TreesColumnName] = new SchemaShape.Column(TreesColumnName,
                    SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Single, false);

            if (LeavesColumnName != null)
                result[LeavesColumnName] = new SchemaShape.Column(LeavesColumnName,
                    SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Single, false);

            if (PathsColumnName != null)
                result[PathsColumnName] = new SchemaShape.Column(PathsColumnName,
                    SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Single, false);

            return new SchemaShape(result.Values);
        }
    }
    /// <summary>
    /// A <see cref="IEstimator{TTransformer}"/> which contains a pre-trained <see cref="TreeEnsembleModelParameters"/> and calling its
    /// <see cref="IEstimator{TTransformer}.Fit(IDataView)"/> produces a featurizer based on the pre-trained model.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// ### Input and Output Columns
    ///
    /// The input label column data must be<xref:System.Single>.
    /// The input features column data must be a known-sized vector of<xref:System.Single>.
    ///
    /// This estimator outputs the following columns:
    ///
    /// | Output Column Name | Column Type | Description|
    /// | -- | -- | -- |
    /// | `Trees` | Vector of<xref:System.Single> | The output values of all trees. |
    /// | `Leaves` | Vector of <xref:System.Single> | The IDs of all leaves where the input feature vector falls into. |
    /// | `Paths` | Vector of <xref:System.Single> | The paths the input feature vector passed through to reach the leaves. |
    ///
    /// Those output columns are all optional and user can change their names.
    /// Please set the names of skipped columns to null so that they would not be produced.
    ///
    /// [!include[algorithm](~/../docs/samples/docs/api-reference/tree-featurization-prediction.md)]
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="TreeExtensions.FeaturizeByPretrainTreeEnsemble(TransformsCatalog, PretrainedTreeFeaturizationEstimator.Options)"/>
    public sealed class PretrainedTreeFeaturizationEstimator : TreeEnsembleFeaturizationEstimatorBase
    {
        /// <summary>
        /// <see cref="Options"/> of <see cref="PretrainedTreeFeaturizationEstimator"/> as
        /// used when calling <see cref="TreeExtensions.FeaturizeByPretrainTreeEnsemble(TransformsCatalog, Options)"/>.
        /// </summary>
        public sealed class Options : OptionsBase
        {
            /// <summary>
            /// The pretrained tree model used to do tree-based featurization. Note that <see cref="TreeEnsembleModelParameters"/> contains a collection of decision trees.
            /// </summary>
            public TreeEnsembleModelParameters ModelParameters;
        };

        private readonly TreeEnsembleModelParameters _modelParameters;

        internal PretrainedTreeFeaturizationEstimator(IHostEnvironment env, Options options) : base(env, options)
        {
            _modelParameters = options.ModelParameters;
        }

        /// <summary>
        /// Produce the <see cref="TreeEnsembleModelParameters"/> for tree-based feature engineering. This function does not
        /// invoke training procedure and just returns the pre-trained model passed in via <see cref="Options.ModelParameters"/>.
        /// </summary>
        private protected override TreeEnsembleModelParameters PrepareModel(IDataView input) => _modelParameters;
    }

    /// <summary>
    /// A <see cref="IEstimator{TTransformer}"/> to transform input feature vector to tree-based features.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// [!include[algorithm](~/../docs/samples/docs/api-reference/io-columns-tree-featurization-binary-classification.md)]
    ///
    /// [!include[algorithm](~/../docs/samples/docs/api-reference/tree-featurization-prediction.md)]
    ///
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="TreeExtensions.FeaturizeByFastTreeBinary(TransformsCatalog, FastTreeBinaryFeaturizationEstimator.Options)"/>
    public sealed class FastTreeBinaryFeaturizationEstimator : TreeEnsembleFeaturizationEstimatorBase
    {
        private readonly FastTreeBinaryTrainer.Options _trainerOptions;

        /// <summary>
        /// Options for the <see cref="FastTreeBinaryFeaturizationEstimator"/>.
        /// </summary>
        public sealed class Options : OptionsBase
        {
            /// <summary>
            /// The configuration of <see cref="FastTreeBinaryTrainer"/> used to train the underlying <see cref="TreeEnsembleModelParameters"/>.
            /// </summary>
            public FastTreeBinaryTrainer.Options TrainerOptions;
        }

        internal FastTreeBinaryFeaturizationEstimator(IHostEnvironment env, Options options)
            : base(env, options)
        {
            _trainerOptions = options.TrainerOptions;
        }

        private protected override TreeEnsembleModelParameters PrepareModel(IDataView input)
        {
            var trainer = new FastTreeBinaryTrainer(Env, _trainerOptions);
            var trained = trainer.Fit(input);
            return trained.Model.SubModel;
        }
    }

    /// <summary>
    /// A <see cref="IEstimator{TTransformer}"/> to transform input feature vector to tree-based features.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// [!include[algorithm](~/../docs/samples/docs/api-reference/io-columns-tree-featurization-regression.md)]
    ///
    /// [!include[algorithm](~/../docs/samples/docs/api-reference/tree-featurization-prediction.md)]
    ///
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="TreeExtensions.FeaturizeByFastTreeRegression(TransformsCatalog, FastTreeRegressionFeaturizationEstimator.Options)"/>
    public sealed class FastTreeRegressionFeaturizationEstimator : TreeEnsembleFeaturizationEstimatorBase
    {
        private readonly FastTreeRegressionTrainer.Options _trainerOptions;

        /// <summary>
        /// Options for the <see cref="FastTreeRegressionFeaturizationEstimator"/>.
        /// </summary>
        public sealed class Options : OptionsBase
        {
            /// <summary>
            /// The configuration of <see cref="FastTreeRegressionTrainer"/> used to train the underlying <see cref="TreeEnsembleModelParameters"/>.
            /// </summary>
            public FastTreeRegressionTrainer.Options TrainerOptions;
        }

        internal FastTreeRegressionFeaturizationEstimator(IHostEnvironment env, Options options)
            : base(env, options)
        {
            _trainerOptions = options.TrainerOptions;
        }

        private protected override TreeEnsembleModelParameters PrepareModel(IDataView input)
        {
            var trainer = new FastTreeRegressionTrainer(Env, _trainerOptions);
            var trained = trainer.Fit(input);
            return trained.Model;
        }
    }

    /// <summary>
    /// A <see cref="IEstimator{TTransformer}"/> to transform input feature vector to tree-based features.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// [!include[algorithm](~/../docs/samples/docs/api-reference/io-columns-tree-featurization-binary-classification.md)]
    ///
    /// [!include[algorithm](~/../docs/samples/docs/api-reference/tree-featurization-prediction.md)]
    ///
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="TreeExtensions.FeaturizeByFastForestBinary(TransformsCatalog, FastForestBinaryFeaturizationEstimator.Options)"/>
    public sealed class FastForestBinaryFeaturizationEstimator : TreeEnsembleFeaturizationEstimatorBase
    {
        private readonly FastForestBinaryTrainer.Options _trainerOptions;

        /// <summary>
        /// Options for the <see cref="FastForestBinaryFeaturizationEstimator"/>.
        /// </summary>
        public sealed class Options : OptionsBase
        {
            /// <summary>
            /// The configuration of <see cref="FastForestBinaryTrainer"/> used to train the underlying <see cref="TreeEnsembleModelParameters"/>.
            /// </summary>
            public FastForestBinaryTrainer.Options TrainerOptions;
        }

        internal FastForestBinaryFeaturizationEstimator(IHostEnvironment env, Options options)
            : base(env, options)
        {
            _trainerOptions = options.TrainerOptions;
        }

        private protected override TreeEnsembleModelParameters PrepareModel(IDataView input)
        {
            var trainer = new FastForestBinaryTrainer(Env, _trainerOptions);
            var trained = trainer.Fit(input);
            return trained.Model;
        }
    }

    /// <summary>
    /// A <see cref="IEstimator{TTransformer}"/> to transform input feature vector to tree-based features.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// [!include[algorithm](~/../docs/samples/docs/api-reference/io-columns-tree-featurization-regression.md)]
    ///
    /// [!include[algorithm](~/../docs/samples/docs/api-reference/tree-featurization-prediction.md)]
    ///
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="TreeExtensions.FeaturizeByFastForestRegression(TransformsCatalog, FastForestRegressionFeaturizationEstimator.Options)"/>
    public sealed class FastForestRegressionFeaturizationEstimator : TreeEnsembleFeaturizationEstimatorBase
    {
        private readonly FastForestRegressionTrainer.Options _trainerOptions;

        /// <summary>
        /// Options for the <see cref="FastForestRegressionFeaturizationEstimator"/>.
        /// </summary>
        public sealed class Options : OptionsBase
        {
            /// <summary>
            /// The configuration of <see cref="FastForestRegressionTrainer"/> used to train the underlying <see cref="TreeEnsembleModelParameters"/>.
            /// </summary>
            public FastForestRegressionTrainer.Options TrainerOptions;
        }

        internal FastForestRegressionFeaturizationEstimator(IHostEnvironment env, Options options)
            : base(env, options)
        {
            _trainerOptions = options.TrainerOptions;
        }

        private protected override TreeEnsembleModelParameters PrepareModel(IDataView input)
        {
            var trainer = new FastForestRegressionTrainer(Env, _trainerOptions);
            var trained = trainer.Fit(input);
            return trained.Model;
        }
    }

    /// <summary>
    /// A <see cref="IEstimator{TTransformer}"/> to transform input feature vector to tree-based features.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// [!include[algorithm](~/../docs/samples/docs/api-reference/io-columns-tree-featurization-ranking.md)]
    ///
    /// [!include[algorithm](~/../docs/samples/docs/api-reference/tree-featurization-prediction.md)]
    ///
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="TreeExtensions.FeaturizeByFastTreeRanking(TransformsCatalog, FastTreeRankingFeaturizationEstimator.Options)"/>
    public sealed class FastTreeRankingFeaturizationEstimator : TreeEnsembleFeaturizationEstimatorBase
    {
        private readonly FastTreeRankingTrainer.Options _trainerOptions;

        /// <summary>
        /// Options for the <see cref="FastTreeRankingFeaturizationEstimator"/>.
        /// </summary>
        public sealed class Options : OptionsBase
        {
            /// <summary>
            /// The configuration of <see cref="FastTreeRankingTrainer"/> used to train the underlying <see cref="TreeEnsembleModelParameters"/>.
            /// </summary>
            public FastTreeRankingTrainer.Options TrainerOptions;
        }

        internal FastTreeRankingFeaturizationEstimator(IHostEnvironment env, Options options)
            : base(env, options)
        {
            _trainerOptions = options.TrainerOptions;
        }

        private protected override TreeEnsembleModelParameters PrepareModel(IDataView input)
        {
            var trainer = new FastTreeRankingTrainer(Env, _trainerOptions);
            var trained = trainer.Fit(input);
            return trained.Model;
        }
    }

    /// <summary>
    /// A <see cref="IEstimator{TTransformer}"/> to transform input feature vector to tree-based features.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// [!include[algorithm](~/../docs/samples/docs/api-reference/io-columns-tree-featurization-regression.md)]
    ///
    /// [!include[algorithm](~/../docs/samples/docs/api-reference/tree-featurization-prediction.md)]
    ///
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="TreeExtensions.FeaturizeByFastTreeTweedie(TransformsCatalog, FastTreeTweedieFeaturizationEstimator.Options)"/>
    public sealed class FastTreeTweedieFeaturizationEstimator : TreeEnsembleFeaturizationEstimatorBase
    {
        private readonly FastTreeTweedieTrainer.Options _trainerOptions;

        /// <summary>
        /// Options for the <see cref="FastTreeTweedieFeaturizationEstimator"/>.
        /// </summary>
        public sealed class Options : OptionsBase
        {
            /// <summary>
            /// The configuration of <see cref="FastTreeTweedieTrainer"/> used to train the underlying <see cref="TreeEnsembleModelParameters"/>.
            /// </summary>
            public FastTreeTweedieTrainer.Options TrainerOptions;
        }

        internal FastTreeTweedieFeaturizationEstimator(IHostEnvironment env, Options options)
            : base(env, options)
        {
            _trainerOptions = options.TrainerOptions;
        }

        private protected override TreeEnsembleModelParameters PrepareModel(IDataView input)
        {
            var trainer = new FastTreeTweedieTrainer(Env, _trainerOptions);
            var trained = trainer.Fit(input);
            return trained.Model;
        }
    }
}
