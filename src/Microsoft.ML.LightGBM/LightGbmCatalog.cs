// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;
using Microsoft.ML.LightGBM;

namespace Microsoft.ML
{
    /// <summary>
    /// LightGBM extension methods.
    /// </summary>
    public static class LightGbmExtensions
    {
        /// <summary>
        /// Predict a target using a decision tree regression model trained with the <see cref="LightGbmRegressorTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="RegressionCatalog"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numLeaves">The number of leaves to use.</param>
        /// <param name="numBoostRound">Number of iterations.</param>
        /// <param name="minDataPerLeaf">The minimal number of documents allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ScoreTensorFlowModel](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Regression/LightGBMRegression.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static LightGbmRegressorTrainer LightGbm(this RegressionCatalog.RegressionTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int? numLeaves = null,
            int? minDataPerLeaf = null,
            double? learningRate = null,
            int numBoostRound = Options.Defaults.NumBoostRound)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new LightGbmRegressorTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, numLeaves, minDataPerLeaf, learningRate, numBoostRound);
        }

        /// <summary>
        /// Predict a target using a decision tree regression model trained with the <see cref="LightGbmRegressorTrainer"/>.
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
        public static LightGbmRegressorTrainer LightGbm(this RegressionCatalog.RegressionTrainers catalog,
            Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new LightGbmRegressorTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using a decision tree binary classification model trained with the <see cref="LightGbmBinaryTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="BinaryClassificationCatalog"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numLeaves">The number of leaves to use.</param>
        /// <param name="numBoostRound">Number of iterations.</param>
        /// <param name="minDataPerLeaf">The minimal number of documents allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ScoreTensorFlowModel](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/LightGbmBinaryClassification.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static LightGbmBinaryTrainer LightGbm(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int? numLeaves = null,
            int? minDataPerLeaf = null,
            double? learningRate = null,
            int numBoostRound = Options.Defaults.NumBoostRound)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new LightGbmBinaryTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, numLeaves, minDataPerLeaf, learningRate, numBoostRound);
        }

        /// <summary>
        /// Predict a target using a decision tree binary classification model trained with the <see cref="LightGbmBinaryTrainer"/>.
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
        public static LightGbmBinaryTrainer LightGbm(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new LightGbmBinaryTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using a decision tree ranking model trained with the <see cref="LightGbmRankingTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="RankingCatalog"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="rowGroupColumnName">The name of the group column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numLeaves">The number of leaves to use.</param>
        /// <param name="numBoostRound">Number of iterations.</param>
        /// <param name="minDataPerLeaf">The minimal number of documents allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        public static LightGbmRankingTrainer LightGbm(this RankingCatalog.RankingTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string rowGroupColumnName = DefaultColumnNames.GroupId,
            string exampleWeightColumnName = null,
            int? numLeaves = null,
            int? minDataPerLeaf = null,
            double? learningRate = null,
            int numBoostRound = Options.Defaults.NumBoostRound)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new LightGbmRankingTrainer(env, labelColumnName, featureColumnName, rowGroupColumnName, exampleWeightColumnName, numLeaves, minDataPerLeaf, learningRate, numBoostRound);
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
        /// Predict a target using a decision tree multiclass classification model trained with the <see cref="LightGbmMulticlassTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="MulticlassClassificationCatalog"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numLeaves">The number of leaves to use.</param>
        /// <param name="numBoostRound">Number of iterations.</param>
        /// <param name="minDataPerLeaf">The minimal number of documents allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ScoreTensorFlowModel](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/MulticlassClassification/LightGBMMulticlassClassification.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static LightGbmMulticlassTrainer LightGbm(this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int? numLeaves = null,
            int? minDataPerLeaf = null,
            double? learningRate = null,
            int numBoostRound = Options.Defaults.NumBoostRound)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new LightGbmMulticlassTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, numLeaves, minDataPerLeaf, learningRate, numBoostRound);
        }

        /// <summary>
        /// Predict a target using a decision tree multiclass classification model trained with the <see cref="LightGbmMulticlassTrainer"/>.
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
        public static LightGbmMulticlassTrainer LightGbm(this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new LightGbmMulticlassTrainer(env, options);
        }
    }
}
