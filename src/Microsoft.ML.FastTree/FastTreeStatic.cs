// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.FastTree;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.StaticPipe.Runtime;
using System;

namespace Microsoft.ML.Trainers
{
    /// <summary>
    /// FastTree <see cref="TrainContextBase"/> extension methods.
    /// </summary>
    public static class FastTreeStatic
    {
        /// <summary>
        /// FastTree <see cref="RegressionContext"/> extension method.
        /// </summary>
        /// <param name="ctx">The <see cref="RegressionContext"/>.</param>
        /// <param name="label">The label column.</param>
        /// <param name="features">The features colum.</param>
        /// <param name="weights">The weights column.</param>
        /// <param name="numLeaves">The number of leaves to use.</param>
        /// <param name="numTrees">Total number of decision trees to create in the ensemble.</param>
        /// <param name="minDocumentsInLeafs">The minimal number of documents allowed in a leaf of a regression tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="advancedSettings">Algorithm advanced settings.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TTupleInShape, TTupleOutShape, TTransformer}.Fit(DataView{TTupleInShape})"/> method is called on the
        /// <see cref="Estimator{TTupleInShape, TTupleOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained. Note that this action cannot change the result in any way;
        /// it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The Score output column indicating the predicted value.</returns>
        public static Scalar<float> FastTree(this RegressionContext.RegressionTrainers ctx,
            Scalar<float> label, Vector<float> features, Scalar<float> weights = null,
            int numLeaves = Defaults.NumLeaves,
            int numTrees = Defaults.NumTrees,
            int minDocumentsInLeafs = Defaults.MinDocumentsInLeafs,
            double learningRate= Defaults.LearningRates,
            Action<FastTreeRegressionTrainer.Arguments> advancedSettings = null,
            Action<FastTreeRegressionPredictor> onFit = null)
        {
            CheckUserValues(label, features, weights, numLeaves, numTrees, minDocumentsInLeafs, learningRate, advancedSettings, onFit);

            var rec = new TrainerEstimatorReconciler.Regression(
               (env, labelName, featuresName, weightsName) =>
               {
                   var trainer = new FastTreeRegressionTrainer(env, labelName, featuresName, weightsName, numLeaves,
                       numTrees, minDocumentsInLeafs, learningRate, advancedSettings);
                   if (onFit != null)
                       return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                   return trainer;
               }, label, features, weights);

            return rec.Score;
        }

        /// <summary>
        /// FastTree <see cref="BinaryClassificationContext"/> extension method.
        /// </summary>
        /// <param name="ctx">The <see cref="BinaryClassificationContext"/>.</param>
        /// <param name="label">The label column.</param>
        /// <param name="features">The features colum.</param>
        /// <param name="weights">The weights column.</param>
        /// <param name="numLeaves">The number of leaves to use.</param>
        /// <param name="numTrees">Total number of decision trees to create in the ensemble.</param>
        /// <param name="minDocumentsInLeafs">The minimal number of documents allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="advancedSettings">Algorithm advanced settings.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TTupleInShape, TTupleOutShape, TTransformer}.Fit(DataView{TTupleInShape})"/> method is called on the
        /// <see cref="Estimator{TTupleInShape, TTupleOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained. Note that this action cannot change the result in any way;
        /// it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The set of output columns including in order the predicted binary classification score (which will range
        /// from negative to positive infinity), the calibrated prediction (from 0 to 1), and the predicted label.</returns>
        public static (Scalar<float> score, Scalar<float> probability, Scalar<bool> predictedLabel) FastTree(this BinaryClassificationContext.BinaryClassificationTrainers ctx,
            Scalar<bool> label, Vector<float> features, Scalar<float> weights = null,
            int numLeaves = Defaults.NumLeaves,
            int numTrees = Defaults.NumTrees,
            int minDocumentsInLeafs = Defaults.MinDocumentsInLeafs,
            double learningRate = Defaults.LearningRates,
            Action<FastTreeBinaryClassificationTrainer.Arguments> advancedSettings = null,
            Action<IPredictorWithFeatureWeights<float>> onFit = null)
        {
            CheckUserValues(label, features, weights, numLeaves, numTrees, minDocumentsInLeafs, learningRate, advancedSettings, onFit);

            var rec = new TrainerEstimatorReconciler.BinaryClassifier(
               (env, labelName, featuresName, weightsName) =>
               {
                   var trainer = new FastTreeBinaryClassificationTrainer(env, labelName, featuresName, weightsName, numLeaves,
                       numTrees, minDocumentsInLeafs, learningRate,  advancedSettings);

                if (onFit != null)
                    return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                else
                    return trainer;
               }, label, features, weights);

            return rec.Output;
        }

        private static void CheckUserValues(PipelineColumn label, Vector<float> features, Scalar<float> weights,
            int numLeaves,
            int numTrees,
            int minDocumentsInLeafs,
            double learningRate,
            Delegate advancedSettings,
            Delegate onFit)
        {
            Contracts.CheckValue(label, nameof(label));
            Contracts.CheckValue(features, nameof(features));
            Contracts.CheckValueOrNull(weights);
            Contracts.CheckParam(numLeaves >= 2, nameof(numLeaves), "Must be at least 2.");
            Contracts.CheckParam(numTrees > 0, nameof(numTrees), "Must be positive");
            Contracts.CheckParam(minDocumentsInLeafs > 0, nameof(minDocumentsInLeafs), "Must be positive");
            Contracts.CheckParam(learningRate > 0, nameof(learningRate), "Must be positive");
            Contracts.CheckValueOrNull(advancedSettings);
            Contracts.CheckValueOrNull(onFit);
        }
    }
}
