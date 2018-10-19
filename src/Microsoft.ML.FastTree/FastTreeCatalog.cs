// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.FastTree;
using System;

namespace Microsoft.ML
{
    /// <summary>
    /// FastTree <see cref="TrainContextBase"/> extension methods.
    /// </summary>
    public static class FastTreeRegressionExtensions
    {
        /// <summary>
        /// Predict a target using a decision tree regression model trained with the <see cref="FastTreeRegressionTrainer"/>.
        /// </summary>
        /// <param name="ctx">The <see cref="RegressionContext"/>.</param>
        /// <param name="label">The label column.</param>
        /// <param name="features">The features colum.</param>
        /// <param name="weights">The optional weights column.</param>
        /// <param name="numTrees">Total number of decision trees to create in the ensemble.</param>
        /// <param name="numLeaves">The maximum number of leaves per decision tree.</param>
        /// <param name="minDatapointsInLeafs">The minimal number of datapoints allowed in a leaf of a regression tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="advancedSettings">Algorithm advanced settings.</param>
        public static FastTreeRegressionTrainer FastTree(this RegressionContext.RegressionTrainers ctx,
            string label = DefaultColumnNames.Label,
            string features = DefaultColumnNames.Features,
            string weights = null,
            int numLeaves = Defaults.NumLeaves,
            int numTrees = Defaults.NumTrees,
            int minDatapointsInLeafs = Defaults.MinDocumentsInLeafs,
            double learningRate = Defaults.LearningRates,
            Action<FastTreeRegressionTrainer.Arguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new FastTreeRegressionTrainer(env, label, features, weights, numLeaves, numTrees, minDatapointsInLeafs, learningRate, advancedSettings);
        }
    }

    public static class FastTreeBinaryClassificationExtensions
    {

        /// <summary>
        /// Predict a target using a decision tree binary classification model trained with the <see cref="FastTreeBinaryClassificationTrainer"/>.
        /// </summary>
        /// <param name="ctx">The <see cref="BinaryClassificationContext"/>.</param>
        /// <param name="label">The label column.</param>
        /// <param name="features">The features colum.</param>
        /// <param name="weights">The optional weights column.</param>
        /// <param name="numTrees">Total number of decision trees to create in the ensemble.</param>
        /// <param name="numLeaves">The maximum number of leaves per decision tree.</param>
        /// <param name="minDatapointsInLeafs">The minimal number of datapoints allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="advancedSettings">Algorithm advanced settings.</param>
        public static FastTreeBinaryClassificationTrainer FastTree(this BinaryClassificationContext.BinaryClassificationTrainers ctx,
            string label = DefaultColumnNames.Label,
            string features = DefaultColumnNames.Features,
            string weights = null,
            int numLeaves = Defaults.NumLeaves,
            int numTrees = Defaults.NumTrees,
            int minDatapointsInLeafs = Defaults.MinDocumentsInLeafs,
            double learningRate = Defaults.LearningRates,
            Action<FastTreeBinaryClassificationTrainer.Arguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new FastTreeBinaryClassificationTrainer(env, label, features, weights, numLeaves, numTrees, minDatapointsInLeafs, learningRate, advancedSettings);
        }
    }

    public static class FastTreeRankingExtensions
    {

        /// <summary>
        /// Ranks a series of inputs based on their relevance, training a decision tree ranking model through the <see cref="FastTreeRankingTrainer"/>.
        /// </summary>
        /// <param name="ctx">The <see cref="RegressionContext"/>.</param>
        /// <param name="label">The label column.</param>
        /// <param name="features">The features colum.</param>
        /// <param name="groupId">The groupId column.</param>
        /// <param name="weights">The optional weights column.</param>
        /// <param name="advancedSettings">Algorithm advanced settings.</param>
        public static FastTreeRankingTrainer FastTree(this RankingContext.RankingTrainers ctx,
            string label = DefaultColumnNames.Label,
            string groupId = DefaultColumnNames.GroupId,
            string features = DefaultColumnNames.Features,
            string weights = null,
            Action<FastTreeRankingTrainer.Arguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new FastTreeRankingTrainer(env, label, features, groupId, weights, advancedSettings);
        }
    }
}
