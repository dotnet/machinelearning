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
        /// <param name="minimumExampleCountPerLeaf">The minimal number of data points required to form a new tree leaf.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[FastTreeRegression](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Regression/FastTree.cs)]
        /// ]]>
        /// </format>
        /// </example>
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
        /// <param name="options">Trainer options.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[FastTreeRegression](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Regression/FastTreeWithOptions.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static FastTreeRegressionTrainer FastTree(this RegressionCatalog.RegressionTrainers catalog,
            FastTreeRegressionTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new FastTreeRegressionTrainer(env, options);
        }

        /// <summary>
        /// Create <see cref="FastTreeBinaryTrainer"/>, which predicts a target using a decision tree binary classification model.
        /// </summary>
        /// <param name="catalog">The <see cref="BinaryClassificationCatalog"/>.</param>
        /// <param name="labelColumnName">The name of the label column. The column data must be <see cref="System.Boolean"/>.</param>
        /// <param name="featureColumnName">The name of the feature column. The column data must be a known-sized vector of <see cref="System.Single"/>.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numberOfTrees">Total number of decision trees to create in the ensemble.</param>
        /// <param name="numberOfLeaves">The maximum number of leaves per decision tree.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of data points required to form a new tree leaf.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[FastTreeBinaryClassification](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/FastTree.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static FastTreeBinaryTrainer FastTree(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
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
            return new FastTreeBinaryTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, numberOfLeaves, numberOfTrees, minimumExampleCountPerLeaf, learningRate);
        }

        /// <summary>
        /// Create <see cref="FastTreeBinaryTrainer"/> with advanced options, which predicts a target using a decision tree binary classification model.
        /// </summary>
        /// <param name="catalog">The <see cref="BinaryClassificationCatalog"/>.</param>
        /// <param name="options">Trainer options.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[FastTreeBinaryClassification](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/FastTreeWithOptions.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static FastTreeBinaryTrainer FastTree(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            FastTreeBinaryTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new FastTreeBinaryTrainer(env, options);
        }

        /// <summary>
        /// Ranks a series of inputs based on their relevance, training a decision tree ranking model with the <see cref="FastTreeRankingTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="RankingCatalog"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="rowGroupColumnName">The name of the group column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numberOfTrees">Total number of decision trees to create in the ensemble.</param>
        /// <param name="numberOfLeaves">The maximum number of leaves per decision tree.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of data points required to form a new tree leaf.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[FastTree](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Ranking/FastTree.cs)]
        /// ]]>
        /// </format>
        /// </example>
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
        /// Ranks a series of inputs based on their relevance, training a decision tree ranking model with the <see cref="FastTreeRankingTrainer"/> and advanced options.
        /// </summary>
        /// <param name="catalog">The <see cref="RankingCatalog"/>.</param>
        /// <param name="options">Trainer options.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[FastTree](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Ranking/FastTreeWithOptions.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static FastTreeRankingTrainer FastTree(this RankingCatalog.RankingTrainers catalog,
            FastTreeRankingTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new FastTreeRankingTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using generalized additive models (GAM) trained with the <see cref="GamBinaryTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="BinaryClassificationCatalog"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numberOfIterations">The number of iterations to use in learning the features.</param>
        /// <param name="maximumBinCountPerFeature">The maximum number of bins to use to approximate features.</param>
        /// <param name="learningRate">The learning rate. GAMs work best with a small learning rate.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[Gam](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/Gam.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static GamBinaryTrainer Gam(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int numberOfIterations = GamDefaults.NumberOfIterations,
            int maximumBinCountPerFeature = GamDefaults.MaximumBinCountPerFeature,
            double learningRate = GamDefaults.LearningRate)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new GamBinaryTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, numberOfIterations, learningRate, maximumBinCountPerFeature);
        }

        /// <summary>
        /// Predict a target using generalized additive models (GAM) trained with the <see cref="GamBinaryTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="BinaryClassificationCatalog"/>.</param>
        /// <param name="options">Trainer options.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[Gam](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/GamWithOptions.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static GamBinaryTrainer Gam(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            GamBinaryTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new GamBinaryTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using generalized additive models (GAM) trained with the <see cref="GamRegressionTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="RegressionCatalog"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numberOfIterations">The number of iterations to use in learning the features.</param>
        /// <param name="maximumBinCountPerFeature">The maximum number of bins to use to approximate features.</param>
        /// <param name="learningRate">The learning rate. GAMs work best with a small learning rate.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[Gam](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Regression/Gam.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static GamRegressionTrainer Gam(this RegressionCatalog.RegressionTrainers catalog,
        string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int numberOfIterations = GamDefaults.NumberOfIterations,
            int maximumBinCountPerFeature = GamDefaults.MaximumBinCountPerFeature,
            double learningRate = GamDefaults.LearningRate)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new GamRegressionTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, numberOfIterations, learningRate, maximumBinCountPerFeature);
        }

        /// <summary>
        /// Predict a target using generalized additive models (GAM) trained with the <see cref="GamRegressionTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="RegressionCatalog"/>.</param>
        /// <param name="options">Trainer options.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[Gam](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Regression/GamWithOptions.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static GamRegressionTrainer Gam(this RegressionCatalog.RegressionTrainers catalog,
            GamRegressionTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new GamRegressionTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using a decision tree regression model trained with the <see cref="FastTreeTweedieTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="RegressionCatalog"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numberOfTrees">Total number of decision trees to create in the ensemble.</param>
        /// <param name="numberOfLeaves">The maximum number of leaves per decision tree.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of data points required to form a new tree leaf.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[FastTreeTweedie](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Regression/FastTreeTweedie.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static FastTreeTweedieTrainer FastTreeTweedie(this RegressionCatalog.RegressionTrainers catalog,
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
            return new FastTreeTweedieTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, numberOfLeaves, numberOfTrees, minimumExampleCountPerLeaf, learningRate);
        }

        /// <summary>
        /// Predict a target using a decision tree regression model trained with the <see cref="FastTreeTweedieTrainer"/> and advanced options.
        /// </summary>
        /// <param name="catalog">The <see cref="RegressionCatalog"/>.</param>
        /// <param name="options">Trainer options.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[FastTreeTweedie](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Regression/FastTreeTweedieWithOptions.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static FastTreeTweedieTrainer FastTreeTweedie(this RegressionCatalog.RegressionTrainers catalog,
            FastTreeTweedieTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new FastTreeTweedieTrainer(env, options);
        }

        /// <summary>
        /// Create <see cref="FastForestRegressionTrainer"/>, which predicts a target using a decision tree regression model.
        /// </summary>
        /// <param name="catalog">The <see cref="RegressionCatalog"/>.</param>
        /// <param name="labelColumnName">The name of the label column. The column data must be <see cref="System.Single"/></param>
        /// <param name="featureColumnName">The name of the feature column. The column data must be a known-sized vector of <see cref="System.Single"/></param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numberOfLeaves">The maximum number of leaves per decision tree.</param>
        /// <param name="numberOfTrees">Total number of decision trees to create in the ensemble.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of data points required to form a new tree leaf.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[FastForestRegression](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Regression/FastForest.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static FastForestRegressionTrainer FastForest(this RegressionCatalog.RegressionTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int numberOfLeaves = Defaults.NumberOfLeaves,
            int numberOfTrees = Defaults.NumberOfTrees,
            int minimumExampleCountPerLeaf = Defaults.MinimumExampleCountPerLeaf)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new FastForestRegressionTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, numberOfLeaves, numberOfTrees, minimumExampleCountPerLeaf);
        }

        /// <summary>
        /// Create <see cref="FastForestRegressionTrainer"/> with advanced options, which predicts a target using a decision tree regression model.
        /// </summary>
        /// <param name="catalog">The <see cref="RegressionCatalog"/>.</param>
        /// <param name="options">Trainer options.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[FastForestRegression](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Regression/FastForestWithOptions.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static FastForestRegressionTrainer FastForest(this RegressionCatalog.RegressionTrainers catalog,
            FastForestRegressionTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new FastForestRegressionTrainer(env, options);
        }

        /// <summary>
        /// Create <see cref="FastForestBinaryTrainer"/>, which predicts a target using a decision tree regression model.
        /// </summary>
        /// <param name="catalog">The <see cref="BinaryClassificationCatalog"/>.</param>
        /// <param name="labelColumnName">The name of the label column. The column data must be <see cref="System.Boolean"/>.</param>
        /// <param name="featureColumnName">The name of the feature column. The column data must be a known-sized vector of <see cref="System.Single"/>.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numberOfTrees">Total number of decision trees to create in the ensemble.</param>
        /// <param name="numberOfLeaves">The maximum number of leaves per decision tree.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of data points required to form a new tree leaf.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[FastForestBinaryClassification](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/FastForest.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static FastForestBinaryTrainer FastForest(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int numberOfLeaves = Defaults.NumberOfLeaves,
            int numberOfTrees = Defaults.NumberOfTrees,
            int minimumExampleCountPerLeaf = Defaults.MinimumExampleCountPerLeaf)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new FastForestBinaryTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, numberOfLeaves, numberOfTrees, minimumExampleCountPerLeaf);
        }

        /// <summary>
        /// Create <see cref="FastForestBinaryTrainer"/> with advanced options, which predicts a target using a decision tree regression model.
        /// </summary>
        /// <param name="catalog">The <see cref="BinaryClassificationCatalog"/>.</param>
        /// <param name="options">Trainer options.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[FastForestBinaryClassification](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/FastForestWithOptions.cs)]
        /// ]]>
        /// </format>
        /// </example>

        public static FastForestBinaryTrainer FastForest(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            FastForestBinaryTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

            var env = CatalogUtils.GetEnvironment(catalog);
            return new FastForestBinaryTrainer(env, options);
        }
    }
}
