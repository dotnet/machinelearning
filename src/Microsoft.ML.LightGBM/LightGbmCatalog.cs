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
        /// <param name="labelColumn">The labelColumn column.</param>
        /// <param name="featureColumn">The features column.</param>
        /// <param name="weights">The weights column.</param>
        /// <param name="numLeaves">The number of leaves to use.</param>
        /// <param name="numBoostRound">Number of iterations.</param>
        /// <param name="minDataPerLeaf">The minimal number of documents allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        public static LightGbmRegressorTrainer LightGbm(this RegressionCatalog.RegressionTrainers catalog,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            int? numLeaves = null,
            int? minDataPerLeaf = null,
            double? learningRate = null,
            int numBoostRound = Options.Defaults.NumBoostRound)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new LightGbmRegressorTrainer(env, labelColumn, featureColumn, weights, numLeaves, minDataPerLeaf, learningRate, numBoostRound);
        }

        /// <summary>
        /// Predict a target using a decision tree regression model trained with the <see cref="LightGbmRegressorTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="RegressionCatalog"/>.</param>
        /// <param name="options">Advanced options to the algorithm.</param>
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
        /// <param name="labelColumn">The labelColumn column.</param>
        /// <param name="featureColumn">The features column.</param>
        /// <param name="weights">The weights column.</param>
        /// <param name="numLeaves">The number of leaves to use.</param>
        /// <param name="numBoostRound">Number of iterations.</param>
        /// <param name="minDataPerLeaf">The minimal number of documents allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        public static LightGbmBinaryTrainer LightGbm(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            int? numLeaves = null,
            int? minDataPerLeaf = null,
            double? learningRate = null,
            int numBoostRound = Options.Defaults.NumBoostRound)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new LightGbmBinaryTrainer(env, labelColumn, featureColumn, weights, numLeaves, minDataPerLeaf, learningRate, numBoostRound);
        }

        /// <summary>
        /// Predict a target using a decision tree binary classification model trained with the <see cref="LightGbmBinaryTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="BinaryClassificationCatalog"/>.</param>
        /// <param name="options">Advanced options to the algorithm.</param>
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
        /// <param name="labelColumn">The labelColumn column.</param>
        /// <param name="featureColumn">The features column.</param>
        /// <param name="weights">The weights column.</param>
        /// <param name="groupIdColumn">The groupId column.</param>
        /// <param name="numLeaves">The number of leaves to use.</param>
        /// <param name="numBoostRound">Number of iterations.</param>
        /// <param name="minDataPerLeaf">The minimal number of documents allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        public static LightGbmRankingTrainer LightGbm(this RankingCatalog.RankingTrainers catalog,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string groupIdColumn = DefaultColumnNames.GroupId,
            string weights = null,
            int? numLeaves = null,
            int? minDataPerLeaf = null,
            double? learningRate = null,
            int numBoostRound = Options.Defaults.NumBoostRound)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new LightGbmRankingTrainer(env, labelColumn, featureColumn, groupIdColumn, weights, numLeaves, minDataPerLeaf, learningRate, numBoostRound);
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
        /// <param name="labelColumn">The labelColumn column.</param>
        /// <param name="featureColumn">The features column.</param>
        /// <param name="weights">The weights column.</param>
        /// <param name="numLeaves">The number of leaves to use.</param>
        /// <param name="numBoostRound">Number of iterations.</param>
        /// <param name="minDataPerLeaf">The minimal number of documents allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        public static LightGbmMulticlassTrainer LightGbm(this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            int? numLeaves = null,
            int? minDataPerLeaf = null,
            double? learningRate = null,
            int numBoostRound = Options.Defaults.NumBoostRound)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new LightGbmMulticlassTrainer(env, labelColumn, featureColumn, weights, numLeaves, minDataPerLeaf, learningRate, numBoostRound);
        }

        /// <summary>
        /// Predict a target using a decision tree multiclass classification model trained with the <see cref="LightGbmMulticlassTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="MulticlassClassificationCatalog"/>.</param>
        /// <param name="options">Advanced options to the algorithm.</param>
        public static LightGbmMulticlassTrainer LightGbm(this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new LightGbmMulticlassTrainer(env, options);
        }
    }
}
