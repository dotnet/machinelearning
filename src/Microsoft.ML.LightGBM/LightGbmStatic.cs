// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.LightGBM;
using Microsoft.ML.StaticPipe.Runtime;
using System;

namespace Microsoft.ML.StaticPipe
{
    /// <summary>
    /// Regression trainer estimators.
    /// </summary>
    public static class LightGbmTrainers
    {
        /// <summary>
        /// Predict a target using a tree regression model trained with the <see cref="LightGbmRegressorTrainer"/>.
        /// </summary>
        /// <param name="ctx">The <see cref="RegressionContext"/>.</param>
        /// <param name="label">The label column.</param>
        /// <param name="features">The features column.</param>
        /// <param name="weights">The weights column.</param>
        /// <param name="numLeaves">The number of leaves to use.</param>
        /// <param name="numBoostRound">Number of iterations.</param>
        /// <param name="minDataPerLeaf">The minimal number of documents allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="advancedSettings">Algorithm advanced settings.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained. Note that this action cannot change the result in any way;
        /// it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The Score output column indicating the predicted value.</returns>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[LightGBM](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Static/LightGBMRegression.cs?range=6-10,18-74 "LightGbm regression example.")]
        /// ]]></format>
        /// </example>
        public static Scalar<float> LightGbm(this RegressionContext.RegressionTrainers ctx,
            Scalar<float> label, Vector<float> features, Scalar<float> weights = null,
            int? numLeaves = null,
            int? minDataPerLeaf = null,
            double? learningRate = null,
            int numBoostRound = LightGbmArguments.Defaults.NumBoostRound,
            Action<LightGbmArguments> advancedSettings = null,
            Action<LightGbmRegressionPredictor> onFit = null)
        {
            CheckUserValues(label, features, weights, numLeaves, minDataPerLeaf, learningRate, numBoostRound, advancedSettings, onFit);

            var rec = new TrainerEstimatorReconciler.Regression(
               (env, labelName, featuresName, weightsName) =>
               {
                   var trainer = new LightGbmRegressorTrainer(env, labelName, featuresName, weightsName, numLeaves,
                       minDataPerLeaf, learningRate, numBoostRound, advancedSettings);
                   if (onFit != null)
                       return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                   return trainer;
               }, label, features, weights);

            return rec.Score;
        }

        /// <summary>
        /// Predict a target using a tree binary classification model trained with the <see cref="LightGbmBinaryTrainer"/>.
        /// </summary>
        /// <param name="ctx">The <see cref="BinaryClassificationContext"/>.</param>
        /// <param name="label">The label column.</param>
        /// <param name="features">The features column.</param>
        /// <param name="weights">The weights column.</param>
        /// <param name="numLeaves">The number of leaves to use.</param>
        /// <param name="numBoostRound">Number of iterations.</param>
        /// <param name="minDataPerLeaf">The minimal number of documents allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="advancedSettings">Algorithm advanced settings.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained. Note that this action cannot change the result in any way;
        /// it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The set of output columns including in order the predicted binary classification score (which will range
        /// from negative to positive infinity), the calibrated prediction (from 0 to 1), and the predicted label.</returns>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[LightGBM](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Static/LightGBMBinaryClassification.cs?range=6-11,19-114 "The LightGBM binary classification example.")]
        /// ]]></format>
        /// </example>
        public static (Scalar<float> score, Scalar<float> probability, Scalar<bool> predictedLabel) LightGbm(this BinaryClassificationContext.BinaryClassificationTrainers ctx,
            Scalar<bool> label, Vector<float> features, Scalar<float> weights = null,
            int? numLeaves = null,
            int? minDataPerLeaf = null,
            double? learningRate = null,
            int numBoostRound = LightGbmArguments.Defaults.NumBoostRound,
            Action<LightGbmArguments> advancedSettings = null,
            Action<IPredictorWithFeatureWeights<float>> onFit = null)
        {
            CheckUserValues(label, features, weights, numLeaves, minDataPerLeaf, learningRate, numBoostRound, advancedSettings, onFit);

            var rec = new TrainerEstimatorReconciler.BinaryClassifier(
               (env, labelName, featuresName, weightsName) =>
               {
                   var trainer = new LightGbmBinaryTrainer(env, labelName, featuresName, weightsName, numLeaves,
                       minDataPerLeaf, learningRate, numBoostRound, advancedSettings);

                   if (onFit != null)
                       return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                   else
                       return trainer;
               }, label, features, weights);

            return rec.Output;
        }

        /// <summary>
        /// Ranks a series of inputs based on their relevance, training a decision tree ranking model through the <see cref="LightGbmRankingTrainer"/>.
        /// </summary>
        /// <param name="ctx">The <see cref="RankingContext"/>.</param>
        /// <param name="label">The label column.</param>
        /// <param name="features">The features column.</param>
        /// <param name="groupId">The groupId column.</param>
        /// <param name="weights">The weights column.</param>
        /// <param name="numLeaves">The number of leaves to use.</param>
        /// <param name="numBoostRound">Number of iterations.</param>
        /// <param name="minDataPerLeaf">The minimal number of documents allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="advancedSettings">Algorithm advanced settings.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained. Note that this action cannot change the result in any way;
        /// it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The set of output columns including in order the predicted binary classification score (which will range
        /// from negative to positive infinity), the calibrated prediction (from 0 to 1), and the predicted label.</returns>
        public static Scalar<float> LightGbm<TVal>(this RankingContext.RankingTrainers ctx,
           Scalar<float> label, Vector<float> features, Key<uint, TVal> groupId, Scalar<float> weights = null,
            int? numLeaves = null,
            int? minDataPerLeaf = null,
            double? learningRate = null,
            int numBoostRound = LightGbmArguments.Defaults.NumBoostRound,
            Action<LightGbmArguments> advancedSettings = null,
            Action<LightGbmRankingPredictor> onFit = null)
        {
            CheckUserValues(label, features, weights, numLeaves, minDataPerLeaf, learningRate, numBoostRound, advancedSettings, onFit);
            Contracts.CheckValue(groupId, nameof(groupId));

            var rec = new TrainerEstimatorReconciler.Ranker<TVal>(
               (env, labelName, featuresName, groupIdName, weightsName) =>
               {
                   var trainer = new LightGbmRankingTrainer(env, labelName, featuresName, groupIdName, weightsName, numLeaves,
                       minDataPerLeaf, learningRate, numBoostRound, advancedSettings);

                   if (onFit != null)
                       return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                   return trainer;
               }, label, features, groupId, weights);

            return rec.Score;
        }

        /// <summary>
        /// Predict a target using a tree multiclass classification model trained with the <see cref="LightGbmMulticlassTrainer"/>.
        /// </summary>
        /// <param name="ctx">The multiclass classification context trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="weights">The weights column.</param>
        /// <param name="numLeaves">The number of leaves to use.</param>
        /// <param name="numBoostRound">Number of iterations.</param>
        /// <param name="minDataPerLeaf">The minimal number of documents allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="advancedSettings">A delegate to set more settings.
        /// The settings here will override the ones provided in the direct method signature,
        /// if both are present and have different values.
        /// The columns names, however need to be provided directly, not through the <paramref name="advancedSettings"/>.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained. Note that this action cannot change the
        /// result in any way; it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The set of output columns including in order the predicted per-class likelihoods (between 0 and 1, and summing up to 1), and the predicted label.</returns>
        public static (Vector<float> score, Key<uint, TVal> predictedLabel)
            LightGbm<TVal>(this MulticlassClassificationContext.MulticlassClassificationTrainers ctx,
            Key<uint, TVal> label,
            Vector<float> features,
            Scalar<float> weights = null,
            int? numLeaves = null,
            int? minDataPerLeaf = null,
            double? learningRate = null,
            int numBoostRound = LightGbmArguments.Defaults.NumBoostRound,
            Action<LightGbmArguments> advancedSettings = null,
            Action<OvaPredictor> onFit = null)
        {
            CheckUserValues(label, features, weights, numLeaves, minDataPerLeaf, learningRate, numBoostRound, advancedSettings, onFit);

            var rec = new TrainerEstimatorReconciler.MulticlassClassifier<TVal>(
                (env, labelName, featuresName, weightsName) =>
                {
                    var trainer = new LightGbmMulticlassTrainer(env, labelName, featuresName, weightsName, numLeaves,
                       minDataPerLeaf, learningRate, numBoostRound, advancedSettings);

                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                    return trainer;
                }, label, features, weights);

            return rec.Output;
        }

        private static void CheckUserValues(PipelineColumn label, Vector<float> features, Scalar<float> weights,
            int? numLeaves,
            int? minDataPerLeaf,
            double? learningRate,
            int numBoostRound,
            Delegate advancedSettings,
            Delegate onFit)
        {
            Contracts.CheckValue(label, nameof(label));
            Contracts.CheckValue(features, nameof(features));
            Contracts.CheckValueOrNull(weights);
            Contracts.CheckParam(!(numLeaves < 2), nameof(numLeaves), "Must be at least 2.");
            Contracts.CheckParam(!(minDataPerLeaf <= 0), nameof(minDataPerLeaf), "Must be positive");
            Contracts.CheckParam(!(learningRate <= 0), nameof(learningRate), "Must be positive");
            Contracts.CheckParam(numBoostRound > 0, nameof(numBoostRound), "Must be positive");
            Contracts.CheckValueOrNull(advancedSettings);
            Contracts.CheckValueOrNull(onFit);
        }
    }
}
