// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.FastTree;

namespace Microsoft.ML
{
    /// <summary>
    /// Tree <see cref="TrainCatalogBase"/> extension methods.
    /// </summary>
    public static class TreeExtensions
    {
        /// <summary>
        /// Predict a target using a decision tree regression model trained with the <see cref="FastTreeRegressionTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="RegressionCatalog"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numberOfTrees">Total number of decision trees to create in the ensemble.</param>
        /// <param name="numberOfLeaves">The maximum number of leaves per decision tree.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of datapoints allowed in a leaf of a regression tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        public static FastTreeRegressionTrainer FastTree(this RegressionCatalog.RegressionTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int numberOfLeaves = Defaults.NumberOfLeaves,
            int numberOfTrees = Defaults.NumberOfTrees,
            int minimumExampleCountPerLeaf = Defaults.MinimumExampleCountPerLeaf,
            double learningRate = Defaults.LearningRate)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new FastTreeRegressionTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, numberOfLeaves, numberOfTrees, minimumExampleCountPerLeaf, learningRate);
        }

        /// <summary>
        /// Predict a target using a decision tree regression model trained with the <see cref="FastTreeRegressionTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="RegressionCatalog"/>.</param>
        /// <param name="options">Algorithm advanced settings.</param>
        public static FastTreeRegressionTrainer FastTree(this RegressionCatalog.RegressionTrainers catalog,
            FastTreeRegressionTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new FastTreeRegressionTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using a decision tree binary classification model trained with the <see cref="FastTreeBinaryClassificationTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="BinaryClassificationCatalog"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numberOfTrees">Total number of decision trees to create in the ensemble.</param>
        /// <param name="numberOfLeaves">The maximum number of leaves per decision tree.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of datapoints allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        public static FastTreeBinaryClassificationTrainer FastTree(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int numberOfLeaves = Defaults.NumberOfLeaves,
            int numberOfTrees = Defaults.NumberOfTrees,
            int minimumExampleCountPerLeaf = Defaults.MinimumExampleCountPerLeaf,
            double learningRate = Defaults.LearningRate)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new FastTreeBinaryClassificationTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, numberOfLeaves, numberOfTrees, minimumExampleCountPerLeaf, learningRate);
        }

        /// <summary>
        /// Predict a target using a decision tree binary classification model trained with the <see cref="FastTreeBinaryClassificationTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="BinaryClassificationCatalog"/>.</param>
        /// <param name="options">Algorithm advanced settings.</param>
        public static FastTreeBinaryClassificationTrainer FastTree(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            FastTreeBinaryClassificationTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new FastTreeBinaryClassificationTrainer(env, options);
        }

        /// <summary>
        /// Ranks a series of inputs based on their relevance, training a decision tree ranking model through the <see cref="FastTreeRankingTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="RankingCatalog"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="rowGroupColumnName">The name of the group column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numberOfTrees">Total number of decision trees to create in the ensemble.</param>
        /// <param name="numberOfLeaves">The maximum number of leaves per decision tree.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of data points allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        public static FastTreeRankingTrainer FastTree(this RankingCatalog.RankingTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string rowGroupColumnName = DefaultColumnNames.GroupId,
            string exampleWeightColumnName = null,
            int numberOfLeaves = Defaults.NumberOfLeaves,
            int numberOfTrees = Defaults.NumberOfTrees,
            int minimumExampleCountPerLeaf = Defaults.MinimumExampleCountPerLeaf,
            double learningRate = Defaults.LearningRate)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new FastTreeRankingTrainer(env, labelColumnName, featureColumnName, rowGroupColumnName, exampleWeightColumnName, numberOfLeaves, numberOfTrees, minimumExampleCountPerLeaf, learningRate);
        }

        /// <summary>
        /// Ranks a series of inputs based on their relevance, training a decision tree ranking model through the <see cref="FastTreeRankingTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="RankingCatalog"/>.</param>
        /// <param name="options">Algorithm advanced settings.</param>
        public static FastTreeRankingTrainer FastTree(this RankingCatalog.RankingTrainers catalog,
            FastTreeRankingTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new FastTreeRankingTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using generalized additive models trained with the <see cref="GeneralizedAdditiveModelBinaryClassificationTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="BinaryClassificationCatalog"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numberOfIterations">The number of iterations to use in learning the features.</param>
        /// <param name="maximumBinCountPerFeature">The maximum number of bins to use to approximate features.</param>
        /// <param name="learningRate">The learning rate. GAMs work best with a small learning rate.</param>
        public static GeneralizedAdditiveModelBinaryClassificationTrainer GeneralizedAdditiveModel(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int numberOfIterations = GamDefaults.NumberOfIterations,
            int maximumBinCountPerFeature = GamDefaults.MaximumBinCountPerFeature,
            double learningRate = GamDefaults.LearningRate)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new GeneralizedAdditiveModelBinaryClassificationTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, numberOfIterations, learningRate, maximumBinCountPerFeature);
        }

        /// <summary>
        /// Predict a target using generalized additive models trained with the <see cref="GeneralizedAdditiveModelBinaryClassificationTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="BinaryClassificationCatalog"/>.</param>
        /// <param name="options">Algorithm advanced settings.</param>
        public static GeneralizedAdditiveModelBinaryClassificationTrainer GeneralizedAdditiveModel(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            GeneralizedAdditiveModelBinaryClassificationTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new GeneralizedAdditiveModelBinaryClassificationTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using generalized additive models trained with the <see cref="GeneralizedAdditiveModelRegressionTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="RegressionCatalog"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numberOfIterations">The number of iterations to use in learning the features.</param>
        /// <param name="maximumBinCountPerFeature">The maximum number of bins to use to approximate features.</param>
        /// <param name="learningRate">The learning rate. GAMs work best with a small learning rate.</param>
        public static GeneralizedAdditiveModelRegressionTrainer GeneralizedAdditiveModel(this RegressionCatalog.RegressionTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int numberOfIterations = GamDefaults.NumberOfIterations,
            int maximumBinCountPerFeature = GamDefaults.MaximumBinCountPerFeature,
            double learningRate = GamDefaults.LearningRate)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new GeneralizedAdditiveModelRegressionTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, numberOfIterations, learningRate, maximumBinCountPerFeature);
        }

        /// <summary>
        /// Predict a target using generalized additive models trained with the <see cref="GeneralizedAdditiveModelRegressionTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="RegressionCatalog"/>.</param>
        /// <param name="options">Algorithm advanced settings.</param>
        public static GeneralizedAdditiveModelRegressionTrainer GeneralizedAdditiveModel(this RegressionCatalog.RegressionTrainers catalog,
            GeneralizedAdditiveModelRegressionTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new GeneralizedAdditiveModelRegressionTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using a decision tree regression model trained with the <see cref="FastTreeTweedieRegressionTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="RegressionCatalog"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numberOfTrees">Total number of decision trees to create in the ensemble.</param>
        /// <param name="numberOfLeaves">The maximum number of leaves per decision tree.</param>
        /// <param name="minDatapointsInLeaves">The minimal number of datapoints allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        public static FastTreeTweedieRegressionTrainer FastTreeTweedie(this RegressionCatalog.RegressionTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int numberOfLeaves = Defaults.NumberOfLeaves,
            int numberOfTrees = Defaults.NumberOfTrees,
            int minDatapointsInLeaves = Defaults.MinimumExampleCountPerLeaf,
            double learningRate = Defaults.LearningRate)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new FastTreeTweedieRegressionTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, numberOfLeaves, numberOfTrees, minDatapointsInLeaves, learningRate);
        }

        /// <summary>
        /// Predict a target using a decision tree regression model trained with the <see cref="FastTreeTweedieRegressionTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="RegressionCatalog"/>.</param>
        /// <param name="options">Algorithm advanced settings.</param>
        public static FastTreeTweedieRegressionTrainer FastTreeTweedie(this RegressionCatalog.RegressionTrainers catalog,
            FastTreeTweedieRegressionTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new FastTreeTweedieRegressionTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using a decision tree regression model trained with the <see cref="FastForestRegressionTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="RegressionCatalog"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numberOfLeaves">The maximum number of leaves per decision tree.</param>
        /// <param name="numberOfTrees">Total number of decision trees to create in the ensemble.</param>
        /// <param name="minDatapointsInLeaves">The minimal number of datapoints allowed in a leaf of the tree, out of the subsampled data.</param>
        public static FastForestRegressionTrainer FastForest(this RegressionCatalog.RegressionTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int numberOfLeaves = Defaults.NumberOfLeaves,
            int numberOfTrees = Defaults.NumberOfTrees,
            int minDatapointsInLeaves = Defaults.MinimumExampleCountPerLeaf)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new FastForestRegressionTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, numberOfLeaves, numberOfTrees, minDatapointsInLeaves);
        }

        /// <summary>
        /// Predict a target using a decision tree regression model trained with the <see cref="FastForestRegressionTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="RegressionCatalog"/>.</param>
        /// <param name="options">Algorithm advanced settings.</param>
        public static FastForestRegressionTrainer FastForest(this RegressionCatalog.RegressionTrainers catalog,
            FastForestRegressionTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new FastForestRegressionTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using a decision tree regression model trained with the <see cref="FastForestClassification"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="BinaryClassificationCatalog"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numberOfTrees">Total number of decision trees to create in the ensemble.</param>
        /// <param name="numberOfLeaves">The maximum number of leaves per decision tree.</param>
        /// <param name="minDatapointsInLeaves">The minimal number of datapoints allowed in a leaf of the tree, out of the subsampled data.</param>
        public static FastForestClassification FastForest(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int numberOfLeaves = Defaults.NumberOfLeaves,
            int numberOfTrees = Defaults.NumberOfTrees,
            int minDatapointsInLeaves = Defaults.MinimumExampleCountPerLeaf)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new FastForestClassification(env, labelColumnName, featureColumnName, exampleWeightColumnName, numberOfLeaves, numberOfTrees, minDatapointsInLeaves);
        }

        /// <summary>
        /// Predict a target using a decision tree regression model trained with the <see cref="FastForestClassification"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="BinaryClassificationCatalog"/>.</param>
        /// <param name="options">Algorithm advanced settings.</param>
        public static FastForestClassification FastForest(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            FastForestClassification.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new FastForestClassification(env, options);
        }
    }
}
