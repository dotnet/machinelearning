// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.LightGBM;
using Microsoft.ML.Trainers;
using System;

namespace Microsoft.ML
{
    /// <summary>
    /// Regression trainer estimators.
    /// </summary>
    public static class LightGbmRegressionExtensions
    {
        /// <summary>
        /// Predict a target using a decision tree regression model trained with the <see cref="LightGbmRegressorTrainer"/>.
        /// </summary>
        /// <param name="ctx">The <see cref="RegressionContext"/>.</param>
        /// <param name="label">The label column.</param>
        /// <param name="features">The features colum.</param>
        /// <param name="weights">The weights column.</param>
        /// <param name="numLeaves">The number of leaves to use.</param>
        /// <param name="numBoostRound">Number of iterations.</param>
        /// <param name="minDataPerLeaf">The minimal number of documents allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="advancedSettings">Algorithm advanced settings.</param>
        public static LightGbmRegressorTrainer LightGbm(this RegressionContext.RegressionTrainers ctx,
            string label = DefaultColumnNames.Label,
            string features = DefaultColumnNames.Features,
            string weights = null,
            int? numLeaves = null,
            int? minDataPerLeaf = null,
            double? learningRate = null,
            int numBoostRound = LightGbmArguments.Defaults.NumBoostRound,
            Action<LightGbmArguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new LightGbmRegressorTrainer(env, label, features, weights, numLeaves, minDataPerLeaf, learningRate, numBoostRound, advancedSettings);
        }
    }

    /// <summary>
    /// Binary Classification trainer estimators.
    /// </summary>
    public static class LightGbmClassificationExtensions
    {

        /// <summary>
        /// Predict a target using a decision tree binary classification model trained with the <see cref="LightGbmBinaryTrainer"/>.
        /// </summary>
        /// <param name="ctx">The <see cref="BinaryClassificationContext"/>.</param>
        /// <param name="label">The label column.</param>
        /// <param name="features">The features colum.</param>
        /// <param name="weights">The weights column.</param>
        /// <param name="numLeaves">The number of leaves to use.</param>
        /// <param name="numBoostRound">Number of iterations.</param>
        /// <param name="minDataPerLeaf">The minimal number of documents allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="advancedSettings">Algorithm advanced settings.</param>
        public static LightGbmBinaryTrainer LightGbm(this BinaryClassificationContext.BinaryClassificationTrainers ctx,
            string label = DefaultColumnNames.Label,
            string features = DefaultColumnNames.Features,
            string weights = null,
            int? numLeaves = null,
            int? minDataPerLeaf = null,
            double? learningRate = null,
            int numBoostRound = LightGbmArguments.Defaults.NumBoostRound,
            Action<LightGbmArguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new LightGbmBinaryTrainer(env, label, features, weights, numLeaves, minDataPerLeaf, learningRate, numBoostRound, advancedSettings);

        }
    }
}
