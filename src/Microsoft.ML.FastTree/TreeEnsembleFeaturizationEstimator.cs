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
    public abstract class FeaturizationEstimatorBase : IEstimator<TreeEnsembleFeaturizationTransformer>
    {
        /// <summary>
        /// The common options of tree-based featurizations such as <see cref="FastTreeBinaryFeaturizationEstimator"/>, <see cref="FastForestBinaryFeaturizationEstimator"/>,
        /// <see cref="FastTreeRegressionFeaturizationEstimator"/>, <see cref="FastForestRegressionFeaturizationEstimator"/>, and <see cref="PretrainedTreeFeaturizationEstimator"/>.
        /// </summary>
        public class CommonOptions
        {
            /// <summary>
            /// The name of feature column in the <see cref="IDataView"/> when calling <see cref="Fit(IDataView)"/>.
            /// The column type must be a vector of <see cref="System.Single"/>.
            /// </summary>
            public string InputColumnName;

            /// <summary>
            /// The estimator has three output columns. Their names would be "Trees" + <see cref="OutputColumnsSuffix"/>,
            /// "Leaves" + <see cref="OutputColumnsSuffix"/>, and "Paths" + <see cref="OutputColumnsSuffix"/>. If <see cref="OutputColumnsSuffix"/>
            /// is <see langword="null"/>, the output names would be "Trees", "Leaves", and "Paths".
            /// </summary>
            public string OutputColumnsSuffix;
        };

        /// <summary>
        /// Feature column to apply tree-based featurization. Note that <see cref="FeatureColumnName"/> is not necessary to be the same as
        /// the feature column used to train the tree model.
        /// </summary>
        private protected readonly string FeatureColumnName;

        /// <summary>
        /// See <see cref="CommonOptions.OutputColumnsSuffix"/>.
        /// </summary>
        private protected readonly string OutputColumnSuffix;

        /// <summary>
        /// Environment of this instance. It controls error throwing and other enviroment settings.
        /// </summary>
        private protected readonly IHostEnvironment Env;

        private protected FeaturizationEstimatorBase(IHostEnvironment env, CommonOptions options)
        {
            Env = env;
            FeatureColumnName = options.InputColumnName;
            OutputColumnSuffix = options.OutputColumnsSuffix;
        }

        /// <summary>
        /// All derived class should implement <see cref="PrepareModel(IDataView)"/> to tell how to get a <see cref="TreeEnsembleModelParameters"/>
        /// out from <paramref name="input"/> and parameters inside this or derived classes.
        /// </summary>
        /// <param name="input">Data used to train a tree model.</param>
        /// <returns>The trees used in <see cref="TreeEnsembleFeaturizationTransformer"/>.</returns>
        private protected abstract TreeEnsembleModelParameters PrepareModel(IDataView input);

        /// <summary>
        /// Produce a <see cref="TreeEnsembleModelParameters"/> which maps the column called <see cref="CommonOptions.InputColumnName"/> in <paramref name="input"/>
        /// to three output columns.
        /// </summary>
        public TreeEnsembleFeaturizationTransformer Fit(IDataView input)
        {
            var model = PrepareModel(input);
            return new TreeEnsembleFeaturizationTransformer(Env, input.Schema,
                input.Schema[FeatureColumnName], model, OutputColumnSuffix);
        }

        /// <summary>
        /// <see cref="PretrainedTreeFeaturizationEstimator"/> adds three float-vector columns into <paramref name="inputSchema"/>.
        /// Given a feature vector column, the added columns are the prediction values of all trees, the leaf IDs the feature
        /// vector falls into, and the paths to those leaves.
        /// </summary>
        /// <param name="inputSchema">A schema which contains a feature column. Note that feature column name can be specified
        /// by <see cref="CommonOptions.InputColumnName"/>.</param>
        /// <returns>Output <see cref="SchemaShape"/> produced by <see cref="PretrainedTreeFeaturizationEstimator"/>.</returns>
        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Env.CheckValue(inputSchema, nameof(inputSchema));

            if (!inputSchema.TryFindColumn(FeatureColumnName, out var col))
                throw Env.ExceptSchemaMismatch(nameof(inputSchema), "input", FeatureColumnName);

            var result = inputSchema.ToDictionary(x => x.Name);

            var treeColumnName = OutputColumnSuffix != null ? OutputColumnSuffix + "Trees" : "Trees";
            result[treeColumnName] = new SchemaShape.Column(treeColumnName,
                SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Single, false);

            var leafColumnName = OutputColumnSuffix != null ? OutputColumnSuffix + "Leaves" : "Leaves";
            result[leafColumnName] = new SchemaShape.Column(leafColumnName,
                SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Single, false);

            var pathColumnName = OutputColumnSuffix != null ? OutputColumnSuffix + "Paths" : "Paths";
            result[pathColumnName] = new SchemaShape.Column(pathColumnName,
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
    /// The input column must be a known-sized vector of<xref:System.Single>.
    ///
    /// This estimator outputs the following columns:
    ///
    /// | Output Column Name | Column Type | Description|
    /// | -- | -- | -- |
    /// | `Trees` | Vector of<xref:System.Single> | The output values of all trees. |
    /// | `Leaves` | Vector of<xref:System.Single> | The IDs of all leaves where the input feature vector falls into. |
    /// | `Paths` | Vector of<xref:System.Single> | The paths the input feature vector passed through to reach the leaves. |
    ///
    /// Check the See Also section for links to usage examples.
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="TreeExtensions.PretrainTreeEnsembleFeaturizing(TransformsCatalog, PretrainedTreeFeaturizationEstimator.Options)"/>
    public sealed class PretrainedTreeFeaturizationEstimator : FeaturizationEstimatorBase
    {
        public sealed class Options : FeaturizationEstimatorBase.CommonOptions
        {
            public TreeEnsembleModelParameters ModelParameters;
        };

        private TreeEnsembleModelParameters _modelParameters;

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

    public sealed class FastTreeBinaryFeaturizationEstimator : FeaturizationEstimatorBase
    {
        private readonly FastTreeBinaryTrainer.Options _trainerOptions;

        public sealed class Options : CommonOptions
        {
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

    public sealed class FastTreeRegressionFeaturizationEstimator : FeaturizationEstimatorBase
    {
        private readonly FastTreeRegressionTrainer.Options _trainerOptions;

        public sealed class Options : CommonOptions
        {
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

    public sealed class FastForestBinaryFeaturizationEstimator : FeaturizationEstimatorBase
    {
        private readonly FastForestBinaryTrainer.Options _trainerOptions;

        public sealed class Options : CommonOptions
        {
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

    public sealed class FastForestRegressionFeaturizationEstimator : FeaturizationEstimatorBase
    {
        private readonly FastForestRegressionTrainer.Options _trainerOptions;

        public sealed class Options : CommonOptions
        {
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

    public sealed class FastTreeRankingFeaturizationEstimator : FeaturizationEstimatorBase
    {
        private readonly FastTreeRankingTrainer.Options _trainerOptions;

        public sealed class Options : CommonOptions
        {
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

    public sealed class FastTreeTweedieFeaturizationEstimator : FeaturizationEstimatorBase
    {
        private readonly FastTreeTweedieTrainer.Options _trainerOptions;

        public sealed class Options : CommonOptions
        {
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
