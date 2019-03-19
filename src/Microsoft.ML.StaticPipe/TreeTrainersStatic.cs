// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.FastTree;

namespace Microsoft.ML.StaticPipe
{
    /// <summary>
    /// FastTree <see cref="TrainCatalogBase"/> extension methods.
    /// </summary>
    public static class TreeRegressionExtensions
    {
        /// <summary>
        /// FastTree <see cref="RegressionCatalog"/> extension method.
        /// Predicts a target using a decision tree regression model trained with the <see cref="FastTreeRegressionTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="RegressionCatalog"/>.</param>
        /// <param name="label">The label column.</param>
        /// <param name="features">The features column.</param>
        /// <param name="weights">The optional weights column.</param>
        /// <param name="numberOfTrees">Total number of decision trees to create in the ensemble.</param>
        /// <param name="numberOfLeaves">The maximum number of leaves per decision tree.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of data points allowed in a leaf of a regression tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained. Note that this action cannot change the result in any way;
        /// it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The Score output column indicating the predicted value.</returns>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[FastTree](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Static/FastTreeRegression.cs)]
        /// ]]></format>
        /// </example>
        public static Scalar<float> FastTree(this RegressionCatalog.RegressionTrainers catalog,
            Scalar<float> label, Vector<float> features, Scalar<float> weights = null,
            int numberOfLeaves = Defaults.NumberOfLeaves,
            int numberOfTrees = Defaults.NumberOfTrees,
            int minimumExampleCountPerLeaf = Defaults.MinimumExampleCountPerLeaf,
            double learningRate = Defaults.LearningRate,
            Action<FastTreeRegressionModelParameters> onFit = null)
        {
            CheckUserValues(label, features, weights, numberOfLeaves, numberOfTrees, minimumExampleCountPerLeaf, learningRate, onFit);

            var rec = new TrainerEstimatorReconciler.Regression(
               (env, labelName, featuresName, weightsName) =>
               {
                   var trainer = new FastTreeRegressionTrainer(env, labelName, featuresName, weightsName, numberOfLeaves,
                       numberOfTrees, minimumExampleCountPerLeaf, learningRate);
                   if (onFit != null)
                       return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                   return trainer;
               }, label, features, weights);

            return rec.Score;
        }

        /// <summary>
        /// FastTree <see cref="RegressionCatalog"/> extension method.
        /// Predicts a target using a decision tree regression model trained with the <see cref="FastTreeRegressionTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="RegressionCatalog"/>.</param>
        /// <param name="label">The label column.</param>
        /// <param name="features">The features column.</param>
        /// <param name="weights">The optional weights column.</param>
        /// <param name="options">Algorithm advanced settings.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained. Note that this action cannot change the result in any way;
        /// it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The Score output column indicating the predicted value.</returns>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[FastTree](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Static/FastTreeRegression.cs)]
        /// ]]></format>
        /// </example>
        public static Scalar<float> FastTree(this RegressionCatalog.RegressionTrainers catalog,
            Scalar<float> label, Vector<float> features, Scalar<float> weights,
            FastTreeRegressionTrainer.Options options,
            Action<FastTreeRegressionModelParameters> onFit = null)
        {
            Contracts.CheckValueOrNull(options);
            CheckUserValues(label, features, weights, onFit);

            var rec = new TrainerEstimatorReconciler.Regression(
               (env, labelName, featuresName, weightsName) =>
               {
                   options.LabelColumnName = labelName;
                   options.FeatureColumnName = featuresName;
                   options.ExampleWeightColumnName = weightsName;

                   var trainer = new FastTreeRegressionTrainer(env, options);
                   if (onFit != null)
                       return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                   return trainer;
               }, label, features, weights);

            return rec.Score;
        }

        /// <summary>
        /// FastTree <see cref="BinaryClassificationCatalog"/> extension method.
        /// Predict a target using a decision tree binary classification model trained with the <see cref="FastTreeBinaryTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="BinaryClassificationCatalog"/>.</param>
        /// <param name="label">The label column.</param>
        /// <param name="features">The features column.</param>
        /// <param name="weights">The optional weights column.</param>
        /// <param name="numberOfTrees">Total number of decision trees to create in the ensemble.</param>
        /// <param name="numberOfLeaves">The maximum number of leaves per decision tree.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of data points allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
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
        ///  [!code-csharp[FastTree](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Static/FastTreeBinaryClassification.cs)]
        /// ]]></format>
        /// </example>
        public static (Scalar<float> score, Scalar<float> probability, Scalar<bool> predictedLabel) FastTree(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            Scalar<bool> label, Vector<float> features, Scalar<float> weights = null,
            int numberOfLeaves = Defaults.NumberOfLeaves,
            int numberOfTrees = Defaults.NumberOfTrees,
            int minimumExampleCountPerLeaf = Defaults.MinimumExampleCountPerLeaf,
            double learningRate = Defaults.LearningRate,
            Action<CalibratedModelParametersBase<FastTreeBinaryModelParameters, PlattCalibrator>> onFit = null)
        {
            CheckUserValues(label, features, weights, numberOfLeaves, numberOfTrees, minimumExampleCountPerLeaf, learningRate, onFit);

            var rec = new TrainerEstimatorReconciler.BinaryClassifier(
               (env, labelName, featuresName, weightsName) =>
               {
                   var trainer = new FastTreeBinaryTrainer(env, labelName, featuresName, weightsName, numberOfLeaves,
                       numberOfTrees, minimumExampleCountPerLeaf, learningRate);

                   if (onFit != null)
                       return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                   else
                       return trainer;
               }, label, features, weights);

            return rec.Output;
        }

        /// <summary>
        /// FastTree <see cref="BinaryClassificationCatalog"/> extension method.
        /// Predict a target using a decision tree binary classification model trained with the <see cref="FastTreeBinaryTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="BinaryClassificationCatalog"/>.</param>
        /// <param name="label">The label column.</param>
        /// <param name="features">The features column.</param>
        /// <param name="weights">The optional weights column.</param>
        /// <param name="options">Algorithm advanced settings.</param>
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
        ///  [!code-csharp[FastTree](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Static/FastTreeBinaryClassification.cs)]
        /// ]]></format>
        /// </example>
        public static (Scalar<float> score, Scalar<float> probability, Scalar<bool> predictedLabel) FastTree(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            Scalar<bool> label, Vector<float> features, Scalar<float> weights,
            FastTreeBinaryTrainer.Options options,
            Action<CalibratedModelParametersBase<FastTreeBinaryModelParameters, PlattCalibrator>> onFit = null)
        {
            Contracts.CheckValueOrNull(options);
            CheckUserValues(label, features, weights, onFit);

            var rec = new TrainerEstimatorReconciler.BinaryClassifier(
               (env, labelName, featuresName, weightsName) =>
               {
                   options.LabelColumnName = labelName;
                   options.FeatureColumnName = featuresName;
                   options.ExampleWeightColumnName = weightsName;

                   var trainer = new FastTreeBinaryTrainer(env, options);

                   if (onFit != null)
                       return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                   else
                       return trainer;
               }, label, features, weights);

            return rec.Output;
        }

        /// <summary>
        /// FastTree <see cref="RankingCatalog"/>.
        /// Ranks a series of inputs based on their relevance, training a decision tree ranking model through the <see cref="FastTreeRankingTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="RegressionCatalog"/>.</param>
        /// <param name="label">The label column.</param>
        /// <param name="features">The features column.</param>
        /// <param name="groupId">The groupId column.</param>
        /// <param name="weights">The optional weights column.</param>
        /// <param name="numberOfTrees">Total number of decision trees to create in the ensemble.</param>
        /// <param name="numberOfLeaves">The maximum number of leaves per decision tree.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of data points allowed in a leaf of a regression tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained. Note that this action cannot change the result in any way;
        /// it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The Score output column indicating the predicted value.</returns>
        public static Scalar<float> FastTree<TVal>(this RankingCatalog.RankingTrainers catalog,
            Scalar<float> label, Vector<float> features, Key<uint, TVal> groupId, Scalar<float> weights = null,
            int numberOfLeaves = Defaults.NumberOfLeaves,
            int numberOfTrees = Defaults.NumberOfTrees,
            int minimumExampleCountPerLeaf = Defaults.MinimumExampleCountPerLeaf,
            double learningRate = Defaults.LearningRate,
            Action<FastTreeRankingModelParameters> onFit = null)
        {
            CheckUserValues(label, features, weights, numberOfLeaves, numberOfTrees, minimumExampleCountPerLeaf, learningRate, onFit);

            var rec = new TrainerEstimatorReconciler.Ranker<TVal>(
               (env, labelName, featuresName, groupIdName, weightsName) =>
               {
                   var trainer = new FastTreeRankingTrainer(env, labelName, featuresName, groupIdName, weightsName, numberOfLeaves,
                       numberOfTrees, minimumExampleCountPerLeaf, learningRate);
                   if (onFit != null)
                       return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                   return trainer;
               }, label, features, groupId, weights);

            return rec.Score;
        }

        /// <summary>
        /// FastTree <see cref="RankingCatalog"/>.
        /// Ranks a series of inputs based on their relevance, training a decision tree ranking model through the <see cref="FastTreeRankingTrainer"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="RegressionCatalog"/>.</param>
        /// <param name="label">The label column.</param>
        /// <param name="features">The features column.</param>
        /// <param name="groupId">The groupId column.</param>
        /// <param name="weights">The optional weights column.</param>
        /// <param name="options">Algorithm advanced settings.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained. Note that this action cannot change the result in any way;
        /// it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The Score output column indicating the predicted value.</returns>
        public static Scalar<float> FastTree<TVal>(this RankingCatalog.RankingTrainers catalog,
            Scalar<float> label, Vector<float> features, Key<uint, TVal> groupId, Scalar<float> weights,
            FastTreeRankingTrainer.Options options,
            Action<FastTreeRankingModelParameters> onFit = null)
        {
            Contracts.CheckValueOrNull(options);
            CheckUserValues(label, features, weights, onFit);

            var rec = new TrainerEstimatorReconciler.Ranker<TVal>(
               (env, labelName, featuresName, groupIdName, weightsName) =>
               {
                   options.LabelColumnName = labelName;
                   options.FeatureColumnName = featuresName;
                   options.RowGroupColumnName = groupIdName;
                   options.ExampleWeightColumnName = weightsName;

                   var trainer = new FastTreeRankingTrainer(env, options);
                   if (onFit != null)
                       return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                   return trainer;
               }, label, features, groupId, weights);

            return rec.Score;
        }

        internal static void CheckUserValues(PipelineColumn label, Vector<float> features, Scalar<float> weights,
            int numberOfLeaves,
            int numberOfTrees,
            int minimumExampleCountPerLeaf,
            double learningRate,
            Delegate onFit)
        {
            Contracts.CheckValue(label, nameof(label));
            Contracts.CheckValue(features, nameof(features));
            Contracts.CheckValueOrNull(weights);
            Contracts.CheckParam(numberOfLeaves >= 2, nameof(numberOfLeaves), "Must be at least 2.");
            Contracts.CheckParam(numberOfTrees > 0, nameof(numberOfTrees), "Must be positive");
            Contracts.CheckParam(minimumExampleCountPerLeaf > 0, nameof(minimumExampleCountPerLeaf), "Must be positive");
            Contracts.CheckParam(learningRate > 0, nameof(learningRate), "Must be positive");
            Contracts.CheckValueOrNull(onFit);
        }

        internal static void CheckUserValues(PipelineColumn label, Vector<float> features, Scalar<float> weights,
            Delegate onFit)
        {
            Contracts.CheckValue(label, nameof(label));
            Contracts.CheckValue(features, nameof(features));
            Contracts.CheckValueOrNull(weights);
            Contracts.CheckValueOrNull(onFit);
        }
    }
}
