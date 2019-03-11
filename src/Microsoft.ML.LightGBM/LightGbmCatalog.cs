// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.LightGBM;
using Microsoft.ML.Runtime;

namespace Microsoft.ML
{
    /// <summary>
    /// LightGBM extension methods.
    /// </summary>
    public static class LightGbmExtensions
    {
        /// <summary>
        /// Predict a target using a decision tree regression model trained with the <see cref="LightGbmRegressionTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="RegressionCatalog"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numberOfLeaves">The number of leaves to use.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of data points allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="numberOfIterations">The number of iterations to use.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ScoreTensorFlowModel](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Regression/LightGBMRegression.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static LightGbmRegressionTrainer LightGbm(this RegressionCatalog.RegressionTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int? numberOfLeaves = null,
            int? minimumExampleCountPerLeaf = null,
            double? learningRate = null,
            int numberOfIterations = Options.Defaults.NumberOfIterations)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new LightGbmRegressionTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, numberOfLeaves, minimumExampleCountPerLeaf, learningRate, numberOfIterations);
        }

        /// <summary>
        /// Predict a target using a decision tree regression model trained with the <see cref="LightGbmRegressionTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="RegressionCatalog"/>.</param>
        /// <param name="options">Advanced options to the algorithm.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ScoreTensorFlowModel](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Regression/LightGBMRegressionWithOptions.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static LightGbmRegressionTrainer LightGbm(this RegressionCatalog.RegressionTrainers catalog,
            Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new LightGbmRegressionTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using a decision tree binary classification model trained with the <see cref="LightGbmBinaryClassificationTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="BinaryClassificationCatalog"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numberOfLeaves">The number of leaves to use.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of data points allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="numberOfIterations">The number of iterations to use.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ScoreTensorFlowModel](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/LightGbmBinaryClassification.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static LightGbmBinaryClassificationTrainer LightGbm(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int? numberOfLeaves = null,
            int? minimumExampleCountPerLeaf = null,
            double? learningRate = null,
            int numberOfIterations = Options.Defaults.NumberOfIterations)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new LightGbmBinaryClassificationTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, numberOfLeaves, minimumExampleCountPerLeaf, learningRate, numberOfIterations);
        }

        /// <summary>
        /// Predict a target using a decision tree binary classification model trained with the <see cref="LightGbmBinaryClassificationTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="BinaryClassificationCatalog"/>.</param>
        /// <param name="options">Advanced options to the algorithm.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ScoreTensorFlowModel](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/LightGBMBinaryClassificationWithOptions.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static LightGbmBinaryClassificationTrainer LightGbm(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new LightGbmBinaryClassificationTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using a decision tree ranking model trained with the <see cref="LightGbmRankingTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="RankingCatalog"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="rowGroupColumnName">The name of the group column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numberOfLeaves">The number of leaves to use.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of data points allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="numberOfIterations">The number of iterations to use.</param>
        public static LightGbmRankingTrainer LightGbm(this RankingCatalog.RankingTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string rowGroupColumnName = DefaultColumnNames.GroupId,
            string exampleWeightColumnName = null,
            int? numberOfLeaves = null,
            int? minimumExampleCountPerLeaf = null,
            double? learningRate = null,
            int numberOfIterations = Options.Defaults.NumberOfIterations)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new LightGbmRankingTrainer(env, labelColumnName, featureColumnName, rowGroupColumnName, exampleWeightColumnName, numberOfLeaves, minimumExampleCountPerLeaf, learningRate, numberOfIterations);
        }

        /// <summary>
        /// Predict a target using a decision tree ranking model trained with the <see cref="LightGbmRankingTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="RankingCatalog"/>.</param>
        /// <param name="options">Advanced options to the algorithm.</param>
        public static LightGbmRankingTrainer LightGbm(this RankingCatalog.RankingTrainers catalog,
            Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new LightGbmRankingTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using a decision tree multiclass classification model trained with the <see cref="LightGbmMulticlassClassificationTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="MulticlassClassificationCatalog"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numberOfLeaves">The number of leaves to use.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of data points allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="numberOfIterations">The number of iterations to use.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ScoreTensorFlowModel](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/MulticlassClassification/LightGBMMulticlassClassification.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static LightGbmMulticlassClassificationTrainer LightGbm(this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int? numberOfLeaves = null,
            int? minimumExampleCountPerLeaf = null,
            double? learningRate = null,
            int numberOfIterations = Options.Defaults.NumberOfIterations)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new LightGbmMulticlassClassificationTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, numberOfLeaves, minimumExampleCountPerLeaf, learningRate, numberOfIterations);
        }

        /// <summary>
        /// Predict a target using a decision tree multiclass classification model trained with the <see cref="LightGbmMulticlassClassificationTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="MulticlassClassificationCatalog"/>.</param>
        /// <param name="options">Advanced options to the algorithm.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ScoreTensorFlowModel](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/MulticlassClassification/LightGBMMulticlassClassificationWithOptions.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static LightGbmMulticlassClassificationTrainer LightGbm(this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new LightGbmMulticlassClassificationTrainer(env, options);
        }
    }
}
