// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Trainers.FastTree;
using System;

namespace Microsoft.ML
{
    /// <summary>
    /// Tree <see cref="TrainContextBase"/> extension methods.
    /// </summary>
    public static class TreeExtensions
    {
        /// <summary>
        /// Predict a target using a decision tree regression model trained with the <see cref="FastTreeRegressionTrainer"/>.
        /// </summary>
        /// <param name="ctx">The <see cref="RegressionContext"/>.</param>
        /// <param name="labelColumn">The label column.</param>
        /// <param name="featureColumn">The feature column.</param>
        /// <param name="weights">The optional weights column.</param>
        /// <param name="numTrees">Total number of decision trees to create in the ensemble.</param>
        /// <param name="numLeaves">The maximum number of leaves per decision tree.</param>
        /// <param name="minDatapointsInLeaves">The minimal number of datapoints allowed in a leaf of a regression tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="advancedSettings">Algorithm advanced settings.</param>
        public static FastTreeRegressionTrainer FastTree(this RegressionContext.RegressionTrainers ctx,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            int numLeaves = Defaults.NumLeaves,
            int numTrees = Defaults.NumTrees,
            int minDatapointsInLeaves = Defaults.MinDocumentsInLeaves,
            double learningRate = Defaults.LearningRates,
            Action<FastTreeRegressionTrainer.Arguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new FastTreeRegressionTrainer(env, labelColumn, featureColumn, weights, numLeaves, numTrees, minDatapointsInLeaves, learningRate, advancedSettings);
        }

        /// <summary>
        /// Predict a target using a decision tree binary classification model trained with the <see cref="FastTreeBinaryClassificationTrainer"/>.
        /// </summary>
        /// <param name="ctx">The <see cref="BinaryClassificationContext"/>.</param>
        /// <param name="labelColumn">The labelColumn column.</param>
        /// <param name="featureColumn">The featureColumn column.</param>
        /// <param name="weights">The optional weights column.</param>
        /// <param name="numTrees">Total number of decision trees to create in the ensemble.</param>
        /// <param name="numLeaves">The maximum number of leaves per decision tree.</param>
        /// <param name="minDatapointsInLeaves">The minimal number of datapoints allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="advancedSettings">Algorithm advanced settings.</param>
        public static FastTreeBinaryClassificationTrainer FastTree(this BinaryClassificationContext.BinaryClassificationTrainers ctx,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            int numLeaves = Defaults.NumLeaves,
            int numTrees = Defaults.NumTrees,
            int minDatapointsInLeaves = Defaults.MinDocumentsInLeaves,
            double learningRate = Defaults.LearningRates,
            Action<FastTreeBinaryClassificationTrainer.Arguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new FastTreeBinaryClassificationTrainer(env, labelColumn, featureColumn, weights, numLeaves, numTrees, minDatapointsInLeaves, learningRate, advancedSettings);
        }

        /// <summary>
        /// Ranks a series of inputs based on their relevance, training a decision tree ranking model through the <see cref="FastTreeRankingTrainer"/>.
        /// </summary>
        /// <param name="ctx">The <see cref="RankingContext"/>.</param>
        /// <param name="labelColumn">The labelColumn column.</param>
        /// <param name="featureColumn">The featureColumn column.</param>
        /// <param name="groupId">The groupId column.</param>
        /// <param name="weights">The optional weights column.</param>
        /// <param name="numTrees">Total number of decision trees to create in the ensemble.</param>
        /// <param name="numLeaves">The maximum number of leaves per decision tree.</param>
        /// <param name="minDatapointsInLeaves">The minimal number of datapoints allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="advancedSettings">Algorithm advanced settings.</param>
        public static FastTreeRankingTrainer FastTree(this RankingContext.RankingTrainers ctx,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string groupId = DefaultColumnNames.GroupId,
            string weights = null,
            int numLeaves = Defaults.NumLeaves,
            int numTrees = Defaults.NumTrees,
            int minDatapointsInLeaves = Defaults.MinDocumentsInLeaves,
            double learningRate = Defaults.LearningRates,
            Action<FastTreeRankingTrainer.Arguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new FastTreeRankingTrainer(env, labelColumn, featureColumn, groupId, weights, numLeaves, numTrees, minDatapointsInLeaves, learningRate, advancedSettings);
        }

        /// <summary>
        /// Predict a target using a decision tree regression model trained with the <see cref="FastTreeRegressionTrainer"/>.
        /// </summary>
        /// <param name="ctx">The <see cref="BinaryClassificationContext"/>.</param>
        /// <param name="labelColumn">The labelColumn column.</param>
        /// <param name="featureColumn">The featureColumn column.</param>
        /// <param name="weights">The optional weights column.</param>
        /// <param name="minDatapointsInLeaves">The minimal number of datapoints allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="advancedSettings">Algorithm advanced settings.</param>
        public static BinaryClassificationGamTrainer GeneralizedAdditiveMethods(this BinaryClassificationContext.BinaryClassificationTrainers ctx,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            int minDatapointsInLeaves = Defaults.MinDocumentsInLeaves,
            double learningRate = Defaults.LearningRates,
            Action<BinaryClassificationGamTrainer.Arguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new BinaryClassificationGamTrainer(env, labelColumn, featureColumn, weights, minDatapointsInLeaves, learningRate, advancedSettings);
        }

        /// <summary>
        /// Predict a target using a decision tree binary classification model trained with the <see cref="RegressionGamTrainer"/>.
        /// </summary>
        /// <param name="ctx">The <see cref="RegressionContext"/>.</param>
        /// <param name="labelColumn">The labelColumn column.</param>
        /// <param name="featureColumn">The featureColumn column.</param>
        /// <param name="weights">The optional weights column.</param>
        /// <param name="minDatapointsInLeaves">The minimal number of datapoints allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="advancedSettings">Algorithm advanced settings.</param>
        public static RegressionGamTrainer GeneralizedAdditiveMethods(this RegressionContext.RegressionTrainers ctx,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            int minDatapointsInLeaves = Defaults.MinDocumentsInLeaves,
            double learningRate = Defaults.LearningRates,
            Action<RegressionGamTrainer.Arguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new RegressionGamTrainer(env, labelColumn, featureColumn, weights, minDatapointsInLeaves, learningRate, advancedSettings);
        }

        /// <summary>
        /// Predict a target using a decision tree regression model trained with the <see cref="FastTreeTweedieTrainer"/>.
        /// </summary>
        /// <param name="ctx">The <see cref="RegressionContext"/>.</param>
        /// <param name="labelColumn">The labelColumn column.</param>
        /// <param name="featureColumn">The featureColumn column.</param>
        /// <param name="weights">The optional weights column.</param>
        /// <param name="numTrees">Total number of decision trees to create in the ensemble.</param>
        /// <param name="numLeaves">The maximum number of leaves per decision tree.</param>
        /// <param name="minDatapointsInLeaves">The minimal number of datapoints allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="advancedSettings">Algorithm advanced settings.</param>
        public static FastTreeTweedieTrainer FastTreeTweedie(this RegressionContext.RegressionTrainers ctx,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            int numLeaves = Defaults.NumLeaves,
            int numTrees = Defaults.NumTrees,
            int minDatapointsInLeaves = Defaults.MinDocumentsInLeaves,
            double learningRate = Defaults.LearningRates,
            Action<FastTreeTweedieTrainer.Arguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new FastTreeTweedieTrainer(env, labelColumn, featureColumn, weights, numLeaves, numTrees, minDatapointsInLeaves, learningRate, advancedSettings);
        }

        /// <summary>
        /// Predict a target using a decision tree regression model trained with the <see cref="FastForestRegression"/>.
        /// </summary>
        /// <param name="ctx">The <see cref="RegressionContext"/>.</param>
        /// <param name="labelColumn">The labelColumn column.</param>
        /// <param name="featureColumn">The featureColumn column.</param>
        /// <param name="weights">The optional weights column.</param>
        /// <param name="numTrees">Total number of decision trees to create in the ensemble.</param>
        /// <param name="numLeaves">The maximum number of leaves per decision tree.</param>
        /// <param name="minDatapointsInLeaves">The minimal number of datapoints allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="advancedSettings">Algorithm advanced settings.</param>
        public static FastForestRegression FastForest(this RegressionContext.RegressionTrainers ctx,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            int numLeaves = Defaults.NumLeaves,
            int numTrees = Defaults.NumTrees,
            int minDatapointsInLeaves = Defaults.MinDocumentsInLeaves,
            double learningRate = Defaults.LearningRates,
            Action<FastForestRegression.Arguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new FastForestRegression(env, labelColumn, featureColumn, weights, numLeaves, numTrees, minDatapointsInLeaves, learningRate, advancedSettings);
        }

        /// <summary>
        /// Predict a target using a decision tree regression model trained with the <see cref="FastForestClassification"/>.
        /// </summary>
        /// <param name="ctx">The <see cref="BinaryClassificationContext"/>.</param>
        /// <param name="labelColumn">The labelColumn column.</param>
        /// <param name="featureColumn">The featureColumn column.</param>
        /// <param name="weights">The optional weights column.</param>
        /// <param name="numTrees">Total number of decision trees to create in the ensemble.</param>
        /// <param name="numLeaves">The maximum number of leaves per decision tree.</param>
        /// <param name="minDatapointsInLeaves">The minimal number of datapoints allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="advancedSettings">Algorithm advanced settings.</param>
        public static FastForestClassification FastForest(this BinaryClassificationContext.BinaryClassificationTrainers ctx,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            int numLeaves = Defaults.NumLeaves,
            int numTrees = Defaults.NumTrees,
            int minDatapointsInLeaves = Defaults.MinDocumentsInLeaves,
            double learningRate = Defaults.LearningRates,
            Action<FastForestClassification.Arguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new FastForestClassification(env, labelColumn, featureColumn, weights,numLeaves, numTrees, minDatapointsInLeaves, learningRate, advancedSettings);
        }
    }
}
