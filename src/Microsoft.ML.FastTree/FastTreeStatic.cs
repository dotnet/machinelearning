// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data.StaticPipe.Runtime;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.FastTree;
using Microsoft.ML.Runtime.Training;
using System;

namespace Microsoft.ML.Trainers
{
    public static class FastTreeStatic
    {
        public static Scalar<float> FastTree(this RegressionContext.RegressionTrainers ctx,
            Scalar<float> label, Vector<float> features, Scalar<float> weights = null,
            int? numLeaves = 20,
            int? numTrees = 100,
            int? minDocumentsInLeafs = 10,
            double? learningRates= 0.2,
            Action<FastTreeRegressionTrainer.Arguments> advancedSettings = null,
            Action<FastTreeRegressionPredictor> onFit = null)
        {
            Contracts.CheckValue(label, nameof(label));
            Contracts.CheckValue(features, nameof(features));
            Contracts.CheckValueOrNull(weights);
            Contracts.CheckParam(numLeaves > 0, nameof(numLeaves), "Must be positive");
            Contracts.CheckParam(numTrees > 0, nameof(numTrees), "Must be positive");
            Contracts.CheckParam(minDocumentsInLeafs > 0, nameof(minDocumentsInLeafs), "Must be positive");
            Contracts.CheckParam(learningRates > 0, nameof(learningRates), "Must be positive");
            Contracts.CheckValueOrNull(advancedSettings);
            Contracts.CheckValueOrNull(onFit);

            var args = new FastTreeRegressionTrainer.Arguments();

            //caching the defaults.
            var defaults = (args.NumLeaves, args.NumTrees, args.MinDocumentsInLeafs, args.LearningRates);

            advancedSettings.Invoke(args);

            // Check that the user didn't supply different parameters in the args, from what it specified directly.
            TrainerUtils.CheckArgsAndAdvancedSettingMismatch(numLeaves, defaults.NumLeaves, args.NumLeaves, nameof(numLeaves));
            TrainerUtils.CheckArgsAndAdvancedSettingMismatch(numTrees, defaults.NumTrees, args.NumTrees, nameof(numTrees));
            TrainerUtils.CheckArgsAndAdvancedSettingMismatch(minDocumentsInLeafs, defaults.MinDocumentsInLeafs, args.MinDocumentsInLeafs, nameof(minDocumentsInLeafs));
            TrainerUtils.CheckArgsAndAdvancedSettingMismatch(numLeaves, defaults.NumLeaves, args.NumLeaves, nameof(numLeaves));

            var rec = new TrainerEstimatorReconciler.Regression(
               (env, labelName, featuresName, weightsName) =>
               {
                   var trainer = new FastTreeRegressionTrainer(env, labelName, featuresName, weightsName, advancedSettings);
                   if (onFit != null)
                       return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                   return trainer;
               }, label, features, weights);

            return rec.Score;
        }

        public static (Scalar<float> score, Scalar<float> probability, Scalar<bool> predictedLabel) FastTree(this BinaryClassificationContext.BinaryClassificationTrainers ctx,
            Scalar<bool> label, Vector<float> features, Scalar<float> weights = null,
            int numLeaves = 20,
            int numTrees = 100,
            int minDocumentsInLeafs = 10,
            double learningRates = 0.2,
            Action<FastTreeBinaryClassificationTrainer.Arguments> advancedSettings = null,
            Action<FastTreeBinaryPredictor> onFit = null)
        {
            Contracts.CheckValue(label, nameof(label));
            Contracts.CheckValue(features, nameof(features));
            Contracts.CheckValueOrNull(weights);
            Contracts.CheckParam(numLeaves > 0, nameof(numLeaves), "Must be positive");
            Contracts.CheckParam(numTrees > 0, nameof(numTrees), "Must be positive");
            Contracts.CheckParam(minDocumentsInLeafs > 0, nameof(minDocumentsInLeafs), "Must be positive");
            Contracts.CheckParam(learningRates > 0, nameof(learningRates), "Must be positive");
            Contracts.CheckValueOrNull(advancedSettings);
            Contracts.CheckValueOrNull(onFit);

            var args = new FastTreeBinaryClassificationTrainer.Arguments();

            //caching the defaults.
            var defaults = (args.NumLeaves, args.NumTrees, args.MinDocumentsInLeafs, args.LearningRates);

            advancedSettings.Invoke(args);

            // Check that the user didn't supply different parameters in the args, from what it specified directly.
            // Warn if that is the case
            TrainerUtils.CheckArgsAndAdvancedSettingMismatch(numLeaves, defaults.NumLeaves, args.NumLeaves, nameof(numLeaves));
            TrainerUtils.CheckArgsAndAdvancedSettingMismatch(numTrees, defaults.NumTrees, args.NumTrees, nameof(numTrees));
            TrainerUtils.CheckArgsAndAdvancedSettingMismatch(minDocumentsInLeafs, defaults.MinDocumentsInLeafs, args.MinDocumentsInLeafs, nameof(minDocumentsInLeafs));
            TrainerUtils.CheckArgsAndAdvancedSettingMismatch(numLeaves, defaults.NumLeaves, args.NumLeaves, nameof(numLeaves));

            var rec = new TrainerEstimatorReconciler.BinaryClassifier(
               (env, labelName, featuresName, weightsName) =>
               {
                   var trainer = new FastTreeBinaryClassificationTrainer(env, args);

                if (onFit != null)
                    return trainer.WithOnFitDelegate(trans => onFit(trans.Model as FastTreeBinaryPredictor));
                else
                    return trainer;
               }, label, features, weights);

            return rec.Output;
        }
    }
}
