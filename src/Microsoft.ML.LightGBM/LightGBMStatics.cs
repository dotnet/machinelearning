// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data.StaticPipe.Runtime;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.LightGBM;
using Microsoft.ML.Runtime.Training;
using System;

namespace Microsoft.ML.Trainers
{
    public static class LightGbmStatics
    {
        public static Scalar<float> LightGbm(this RegressionContext.RegressionTrainers ctx,
            Scalar<float> label, Vector<float> features, Scalar<float> weights = null,
            int? numLeaves = null,
            int? minDataPerLeaf = null,
            double? learningRate = null,
            int numBoostRound = 100,
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

        public static (Scalar<float> score, Scalar<float> probability, Scalar<bool> predictedLabel) FastTree(this BinaryClassificationContext.BinaryClassificationTrainers ctx,
            Scalar<bool> label, Vector<float> features, Scalar<float> weights = null,
            int? numLeaves = null,
            int? minDataPerLeaf = null,
            double? learningRate = null,
            int numBoostRound = 100,
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

        private static void CheckUserValues<TVal, TArgs, TPred>(Scalar<TVal> label, Vector<float> features, Scalar<float> weights,
            int? numLeaves,
            int? minDataPerLeaf,
            double? learningRate,
            int numBoostRound,
            Action<TArgs> advancedSettings,
            Action<TPred> onFit)
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
