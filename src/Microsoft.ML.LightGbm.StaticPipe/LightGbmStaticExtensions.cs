// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Runtime;
using Microsoft.ML.StaticPipe;

namespace Microsoft.ML.Trainers.LightGbm.StaticPipe
{
    /// <summary>
    /// Regression trainer estimators.
    /// </summary>
    public static class LightGbmStaticExtensions
    {
        /// <summary>
        /// Predict a target using a tree regression model trained with the <see cref="LightGbmRegressionTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="RegressionCatalog"/>.</param>
        /// <param name="label">The label column.</param>
        /// <param name="features">The features column.</param>
        /// <param name="weights">The weights column.</param>
        /// <param name="numberOfLeaves">The number of leaves to use.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of data points allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="numberOfIterations">Number of iterations.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained. Note that this action cannot change the result in any way;
        /// it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The Score output column indicating the predicted value.</returns>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[LightGBM](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Static/LightGBMRegression.cs)]
        /// ]]></format>
        /// </example>
        public static Scalar<float> LightGbm(this RegressionCatalog.RegressionTrainers catalog,
            Scalar<float> label, Vector<float> features, Scalar<float> weights = null,
            int? numberOfLeaves = null,
            int? minimumExampleCountPerLeaf = null,
            double? learningRate = null,
            int numberOfIterations = Defaults.NumberOfIterations,
            Action<LightGbmRegressionModelParameters> onFit = null)
        {
            CheckUserValues(label, features, weights, numberOfLeaves, minimumExampleCountPerLeaf, learningRate, numberOfIterations, onFit);

            var rec = new TrainerEstimatorReconciler.Regression(
               (env, labelName, featuresName, weightsName) =>
               {
                   var trainer = new LightGbmRegressionTrainer(env, labelName, featuresName, weightsName, numberOfLeaves,
                       minimumExampleCountPerLeaf, learningRate, numberOfIterations);
                   if (onFit != null)
                       return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                   return trainer;
               }, label, features, weights);

            return rec.Score;
        }

        /// <summary>
        /// Predict a target using a tree regression model trained with the <see cref="LightGbmRegressionTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="RegressionCatalog"/>.</param>
        /// <param name="label">The label column.</param>
        /// <param name="features">The features column.</param>
        /// <param name="weights">The weights column.</param>
        /// <param name="options">Algorithm advanced settings.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained. Note that this action cannot change the result in any way;
        /// it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The Score output column indicating the predicted value.</returns>
        public static Scalar<float> LightGbm(this RegressionCatalog.RegressionTrainers catalog,
            Scalar<float> label, Vector<float> features, Scalar<float> weights,
            LightGbmRegressionTrainer.Options options,
            Action<LightGbmRegressionModelParameters> onFit = null)
        {
            Contracts.CheckValue(options, nameof(options));
            CheckUserValues(label, features, weights, onFit);

            var rec = new TrainerEstimatorReconciler.Regression(
               (env, labelName, featuresName, weightsName) =>
               {
                   options.LabelColumnName = labelName;
                   options.FeatureColumnName = featuresName;
                   options.ExampleWeightColumnName = weightsName;

                   var trainer = new LightGbmRegressionTrainer(env, options);
                   if (onFit != null)
                       return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                   return trainer;
               }, label, features, weights);

            return rec.Score;
        }

        /// <summary>
        /// Predict a target using a tree binary classification model trained with the <see cref="LightGbmBinaryTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="BinaryClassificationCatalog"/>.</param>
        /// <param name="label">The label column.</param>
        /// <param name="features">The features column.</param>
        /// <param name="weights">The weights column.</param>
        /// <param name="numberOfLeaves">The number of leaves to use.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of data points allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="numberOfIterations">Number of iterations.</param>
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
        ///  [!code-csharp[LightGBM](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Static/LightGBMBinaryClassification.cs)]
        /// ]]></format>
        /// </example>
        public static (Scalar<float> score, Scalar<float> probability, Scalar<bool> predictedLabel) LightGbm(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            Scalar<bool> label,
            Vector<float> features,
            Scalar<float> weights = null,
            int? numberOfLeaves = null,
            int? minimumExampleCountPerLeaf = null,
            double? learningRate = null,
            int numberOfIterations = Defaults.NumberOfIterations,
            Action<CalibratedModelParametersBase<LightGbmBinaryModelParameters, PlattCalibrator>> onFit = null)
        {
            CheckUserValues(label, features, weights, numberOfLeaves, minimumExampleCountPerLeaf, learningRate, numberOfIterations, onFit);

            var rec = new TrainerEstimatorReconciler.BinaryClassifier(
               (env, labelName, featuresName, weightsName) =>
               {
                   var trainer = new LightGbmBinaryTrainer(env, labelName, featuresName, weightsName, numberOfLeaves,
                       minimumExampleCountPerLeaf, learningRate, numberOfIterations);

                   if (onFit != null)
                       return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                   else
                       return trainer;
               }, label, features, weights);

            return rec.Output;
        }

        /// <summary>
        /// Predict a target using a tree binary classification model trained with the <see cref="LightGbmBinaryTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="BinaryClassificationCatalog"/>.</param>
        /// <param name="label">The label column.</param>
        /// <param name="features">The features column.</param>
        /// <param name="weights">The weights column.</param>
        /// <param name="options">Algorithm advanced settings.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained. Note that this action cannot change the result in any way;
        /// it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The set of output columns including in order the predicted binary classification score (which will range
        /// from negative to positive infinity), the calibrated prediction (from 0 to 1), and the predicted label.</returns>
        public static (Scalar<float> score, Scalar<float> probability, Scalar<bool> predictedLabel) LightGbm(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            Scalar<bool> label, Vector<float> features, Scalar<float> weights,
            LightGbmBinaryTrainer.Options options,
            Action<CalibratedModelParametersBase<LightGbmBinaryModelParameters, PlattCalibrator>> onFit = null)
        {
            Contracts.CheckValue(options, nameof(options));
            CheckUserValues(label, features, weights, onFit);

            var rec = new TrainerEstimatorReconciler.BinaryClassifier(
               (env, labelName, featuresName, weightsName) =>
               {
                   options.LabelColumnName = labelName;
                   options.FeatureColumnName = featuresName;
                   options.ExampleWeightColumnName = weightsName;

                   var trainer = new LightGbmBinaryTrainer(env, options);

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
        /// <param name="catalog">The <see cref="RankingCatalog"/>.</param>
        /// <param name="label">The label column.</param>
        /// <param name="features">The features column.</param>
        /// <param name="groupId">The groupId column.</param>
        /// <param name="weights">The weights column.</param>
        /// <param name="numberOfLeaves">The number of leaves to use.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of data points allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="numberOfIterations">Number of iterations.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained. Note that this action cannot change the result in any way;
        /// it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The set of output columns including in order the predicted binary classification score (which will range
        /// from negative to positive infinity), the calibrated prediction (from 0 to 1), and the predicted label.</returns>
        public static Scalar<float> LightGbm<TVal>(this RankingCatalog.RankingTrainers catalog,
            Scalar<float> label,
            Vector<float> features,
            Key<uint, TVal> groupId,
            Scalar<float> weights = null,
            int? numberOfLeaves = null,
            int? minimumExampleCountPerLeaf = null,
            double? learningRate = null,
            int numberOfIterations = Defaults.NumberOfIterations,
            Action<LightGbmRankingModelParameters> onFit = null)
        {
            CheckUserValues(label, features, weights, numberOfLeaves, minimumExampleCountPerLeaf, learningRate, numberOfIterations, onFit);
            Contracts.CheckValue(groupId, nameof(groupId));

            var rec = new TrainerEstimatorReconciler.Ranker<TVal>(
               (env, labelName, featuresName, groupIdName, weightsName) =>
               {
                   var trainer = new LightGbmRankingTrainer(env, labelName, featuresName, groupIdName, weightsName, numberOfLeaves,
                       minimumExampleCountPerLeaf, learningRate, numberOfIterations);

                   if (onFit != null)
                       return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                   return trainer;
               }, label, features, groupId, weights);

            return rec.Score;
        }

        /// <summary>
        /// Ranks a series of inputs based on their relevance, training a decision tree ranking model through the <see cref="LightGbmRankingTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="RankingCatalog"/>.</param>
        /// <param name="label">The label column.</param>
        /// <param name="features">The features column.</param>
        /// <param name="groupId">The groupId column.</param>
        /// <param name="weights">The weights column.</param>
        /// <param name="options">Algorithm advanced settings.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained. Note that this action cannot change the result in any way;
        /// it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The set of output columns including in order the predicted binary classification score (which will range
        /// from negative to positive infinity), the calibrated prediction (from 0 to 1), and the predicted label.</returns>
        public static Scalar<float> LightGbm<TVal>(this RankingCatalog.RankingTrainers catalog,
           Scalar<float> label, Vector<float> features, Key<uint, TVal> groupId, Scalar<float> weights,
            LightGbmRankingTrainer.Options options,
            Action<LightGbmRankingModelParameters> onFit = null)
        {
            Contracts.CheckValue(options, nameof(options));
            CheckUserValues(label, features, weights, onFit);
            Contracts.CheckValue(groupId, nameof(groupId));

            var rec = new TrainerEstimatorReconciler.Ranker<TVal>(
               (env, labelName, featuresName, groupIdName, weightsName) =>
               {
                   options.LabelColumnName = labelName;
                   options.FeatureColumnName = featuresName;
                   options.RowGroupColumnName = groupIdName;
                   options.ExampleWeightColumnName = weightsName;

                   var trainer = new LightGbmRankingTrainer(env, options);

                   if (onFit != null)
                       return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                   return trainer;
               }, label, features, groupId, weights);

            return rec.Score;
        }

        /// <summary>
        /// Predict a target using a tree multiclass classification model trained with the <see cref="LightGbmMulticlassTrainer"/>.
        /// </summary>
        /// <param name="catalog">The multiclass classification catalog trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="weights">The weights column.</param>
        /// <param name="numberOfLeaves">The number of leaves to use.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of data points allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="numberOfIterations">Number of iterations.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained. Note that this action cannot change the
        /// result in any way; it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The set of output columns including in order the predicted per-class likelihoods (between 0 and 1, and summing up to 1), and the predicted label.</returns>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[MF](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Static/LightGBMMulticlassWithInMemoryData.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static (Vector<float> score, Key<uint, TVal> predictedLabel)
            LightGbm<TVal>(this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            Key<uint, TVal> label,
            Vector<float> features,
            Scalar<float> weights = null,
            int? numberOfLeaves = null,
            int? minimumExampleCountPerLeaf = null,
            double? learningRate = null,
            int numberOfIterations = Defaults.NumberOfIterations,
            Action<OneVersusAllModelParameters> onFit = null)
        {
            CheckUserValues(label, features, weights, numberOfLeaves, minimumExampleCountPerLeaf, learningRate, numberOfIterations, onFit);

            var rec = new TrainerEstimatorReconciler.MulticlassClassificationReconciler<TVal>(
                (env, labelName, featuresName, weightsName) =>
                {
                    var trainer = new LightGbmMulticlassTrainer(env, labelName, featuresName, weightsName, numberOfLeaves,
                       minimumExampleCountPerLeaf, learningRate, numberOfIterations);

                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                    return trainer;
                }, label, features, weights);

            return rec.Output;
        }

        /// <summary>
        /// Predict a target using a tree multiclass classification model trained with the <see cref="LightGbmMulticlassTrainer"/>.
        /// </summary>
        /// <param name="catalog">The multiclass classification catalog trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="weights">The weights column.</param>
        /// <param name="options">Advanced options to the algorithm.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained. Note that this action cannot change the
        /// result in any way; it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The set of output columns including in order the predicted per-class likelihoods (between 0 and 1, and summing up to 1), and the predicted label.</returns>
        public static (Vector<float> score, Key<uint, TVal> predictedLabel)
            LightGbm<TVal>(this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            Key<uint, TVal> label,
            Vector<float> features,
            Scalar<float> weights,
            LightGbmMulticlassTrainer.Options options,
            Action<OneVersusAllModelParameters> onFit = null)
        {
            Contracts.CheckValue(options, nameof(options));
            CheckUserValues(label, features, weights, onFit);

            var rec = new TrainerEstimatorReconciler.MulticlassClassificationReconciler<TVal>(
                (env, labelName, featuresName, weightsName) =>
                {
                    options.LabelColumnName = labelName;
                    options.FeatureColumnName = featuresName;
                    options.ExampleWeightColumnName = weightsName;

                    var trainer = new LightGbmMulticlassTrainer(env, options);

                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                    return trainer;
                }, label, features, weights);

            return rec.Output;
        }

        private static void CheckUserValues(PipelineColumn label, Vector<float> features, Scalar<float> weights,
            int? numberOfLeaves,
            int? minimumExampleCountPerLeaf,
            double? learningRate,
            int numBoostRound,
            Delegate onFit)
        {
            Contracts.CheckValue(label, nameof(label));
            Contracts.CheckValue(features, nameof(features));
            Contracts.CheckValueOrNull(weights);
            Contracts.CheckParam(!(numberOfLeaves < 2), nameof(numberOfLeaves), "Must be at least 2.");
            Contracts.CheckParam(!(minimumExampleCountPerLeaf <= 0), nameof(minimumExampleCountPerLeaf), "Must be positive");
            Contracts.CheckParam(!(learningRate <= 0), nameof(learningRate), "Must be positive");
            Contracts.CheckParam(numBoostRound > 0, nameof(numBoostRound), "Must be positive");
            Contracts.CheckValueOrNull(onFit);
        }

        private static void CheckUserValues(PipelineColumn label, Vector<float> features, Scalar<float> weights, Delegate onFit)
        {
            Contracts.CheckValue(label, nameof(label));
            Contracts.CheckValue(features, nameof(features));
            Contracts.CheckValueOrNull(weights);
            Contracts.CheckValueOrNull(onFit);
        }
    }
}
