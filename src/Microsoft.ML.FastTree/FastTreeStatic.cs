// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data.StaticPipe.Runtime;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.FastTree;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Training;
using System;

namespace Microsoft.ML.Trainers
{
    public static class FastTreeStatic
    {
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

        private static void CheckUserValues<TVal, TArgs, TPred>(Scalar<TVal> label, Vector<float> features, Scalar<float> weights = null,
            int numLeaves = Defaults.NumLeaves,
            int numTrees = Defaults.NumTrees,
            int minDocumentsInLeafs = Defaults.MinDocumentsInLeafs,
            double learningRate = Defaults.LearningRates,
            Action<TArgs> advancedSettings = null,
            Action<TPred> onFit = null)
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
