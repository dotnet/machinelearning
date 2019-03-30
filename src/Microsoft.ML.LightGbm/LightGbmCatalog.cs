// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.LightGbm;

namespace Microsoft.ML
{
    /// <summary>
    /// LightGBM extension methods.
    /// </summary>
    public static class LightGbmExtensions
    {
        /// <summary>
        /// Predict a target using a gradient boosting decision tree regression model trained with the <see cref="LightGbmRegressionTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="RegressionCatalog"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numberOfLeaves">The maximum number of leaves in one tree.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of data points required to form a new tree leaf.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="numberOfIterations">The number of boosting iterations. A new tree is created in each iteration, so this is equivalent to the number of trees.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[LightGbmRegression](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Regression/LightGbm.cs)]
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
            int numberOfIterations = Defaults.NumberOfIterations)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new LightGbmRegressionTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, numberOfLeaves, minimumExampleCountPerLeaf, learningRate, numberOfIterations);
        }

        /// <summary>
        /// Predict a target using a gradient boosting decision tree regression model trained with the <see cref="LightGbmRegressionTrainer"/> and advanced options.
        /// </summary>
        /// <param name="catalog">The <see cref="RegressionCatalog"/>.</param>
        /// <param name="options">Trainer options.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[LightGbmRegression](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Regression/LightGbmWithOptions.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static LightGbmRegressionTrainer LightGbm(this RegressionCatalog.RegressionTrainers catalog,
            LightGbmRegressionTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new LightGbmRegressionTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using a gradient boosting decision tree binary classification model trained with the <see cref="LightGbmBinaryTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="BinaryClassificationCatalog"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numberOfLeaves">The maximum number of leaves in one tree.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of data points required to form a new tree leaf.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="numberOfIterations">The number of boosting iterations. A new tree is created in each iteration, so this is equivalent to the number of trees.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[LightGbmBinaryClassification](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/LightGbm.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static LightGbmBinaryTrainer LightGbm(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int? numberOfLeaves = null,
            int? minimumExampleCountPerLeaf = null,
            double? learningRate = null,
            int numberOfIterations = Defaults.NumberOfIterations)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new LightGbmBinaryTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, numberOfLeaves, minimumExampleCountPerLeaf, learningRate, numberOfIterations);
        }

        /// <summary>
        /// Predict a target using a gradient boosting decision tree binary classification model trained with the <see cref="LightGbmBinaryTrainer"/> and advanced options.
        /// </summary>
        /// <param name="catalog">The <see cref="BinaryClassificationCatalog"/>.</param>
        /// <param name="options">Trainer options.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[LightGbmBinaryClassification](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/LightGbmWithOptions.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static LightGbmBinaryTrainer LightGbm(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            LightGbmBinaryTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new LightGbmBinaryTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using a gradient boosting decision tree ranking model trained with the <see cref="LightGbmRankingTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="RankingCatalog"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="rowGroupColumnName">The name of the group column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numberOfLeaves">The maximum number of leaves in one tree.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of data points required to form a new tree leaf.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="numberOfIterations">The number of boosting iterations. A new tree is created in each iteration, so this is equivalent to the number of trees.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[LightGbmRanking](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Ranking/LightGbm.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static LightGbmRankingTrainer LightGbm(this RankingCatalog.RankingTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string rowGroupColumnName = DefaultColumnNames.GroupId,
            string exampleWeightColumnName = null,
            int? numberOfLeaves = null,
            int? minimumExampleCountPerLeaf = null,
            double? learningRate = null,
            int numberOfIterations = Defaults.NumberOfIterations)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new LightGbmRankingTrainer(env, labelColumnName, featureColumnName, rowGroupColumnName, exampleWeightColumnName,
                                                numberOfLeaves, minimumExampleCountPerLeaf, learningRate, numberOfIterations);
        }

        /// <summary>
        /// Predict a target using a gradient boosting decision tree ranking model trained with the <see cref="LightGbmRankingTrainer"/> and advanced options.
        /// </summary>
        /// <param name="catalog">The <see cref="RankingCatalog"/>.</param>
        /// <param name="options">Trainer options.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[LightGbmRanking](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Ranking/LightGbmWithOptions.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static LightGbmRankingTrainer LightGbm(this RankingCatalog.RankingTrainers catalog,
            LightGbmRankingTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new LightGbmRankingTrainer(env, options);
        }

        /// <summary>
        /// Predict a target using a gradient boosting decision tree multiclass classification model trained with the <see cref="LightGbmMulticlassTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="MulticlassClassificationCatalog"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numberOfLeaves">The maximum number of leaves in one tree.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of data points required to form a new tree leaf.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="numberOfIterations">The number of boosting iterations. A new tree is created in each iteration, so this is equivalent to the number of trees.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[LightGbmMulticlassClassification](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/MulticlassClassification/LightGbm.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static LightGbmMulticlassTrainer LightGbm(this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int? numberOfLeaves = null,
            int? minimumExampleCountPerLeaf = null,
            double? learningRate = null,
            int numberOfIterations = Defaults.NumberOfIterations)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new LightGbmMulticlassTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, numberOfLeaves, minimumExampleCountPerLeaf, learningRate, numberOfIterations);
        }

        /// <summary>
        /// Predict a target using a gradient boosting decision tree multiclass classification model trained with the <see cref="LightGbmMulticlassTrainer"/> and advanced options.
        /// </summary>
        /// <param name="catalog">The <see cref="MulticlassClassificationCatalog"/>.</param>
        /// <param name="options">Trainer options.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[LightGbmMulticlassClassification](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/MulticlassClassification/LightGbmWithOptions.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static LightGbmMulticlassTrainer LightGbm(this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            LightGbmMulticlassTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new LightGbmMulticlassTrainer(env, options);
        }
    }
}
