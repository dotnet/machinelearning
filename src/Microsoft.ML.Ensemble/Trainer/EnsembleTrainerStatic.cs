// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Ensemble;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.StaticPipe.Runtime;

namespace Microsoft.ML.StaticPipe
{
    using Arguments = LogisticRegression.Arguments;

    /// <summary>
    /// Binary Classification trainer estimators.
    /// </summary>
    public static partial class BinaryClassificationTrainers
    {
        /// <summary>
        ///  Predict a target using a linear binary classification model trained with the <see cref="Microsoft.ML.Runtime.Ensemble.EnsembleTrainer"/> trainer.
        /// </summary>
        /// <param name="ctx">The multiclass classification context trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TTupleInShape, TTupleOutShape, TTransformer}.Fit(DataView{TTupleInShape})"/> method is called on the
        /// <see cref="Estimator{TTupleInShape, TTupleOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained. Note that this action cannot change the
        /// result in any way; it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The set of output columns including in order the predicted per-class likelihoods (between 0 and 1, and summing up to 1), and the predicted label.</returns>
        public static (Scalar<float> score, Scalar<float> probability, Scalar<bool> predictedLabel) EnsembleTrainer(this BinaryClassificationContext.BinaryClassificationTrainers ctx,
            Scalar<bool> label,
            Vector<float> features,
            Action<EnsembleTrainer.Arguments> advancedSettings = null,
            Action<IPredictorProducing<float>> onFit = null)
        {
            //LbfgsStaticUtils.ValidateParams(label, features, weights, l1Weight, l2Weight, optimizationTolerance, memorySize, enoforceNoNegativity, onFit);

            var rec = new TrainerEstimatorReconciler.BinaryClassifier(
                (env, labelName, featuresName, weightsName) =>
                {
                    var trainer = new EnsembleTrainer(env, featuresName, labelName, advancedSettings);

                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                    return trainer;

                }, label, features, null);

            return rec.Output;
        }
    }

    /// <summary>
    /// Regression trainer estimators.
    /// </summary>
    public static partial class RegressionTrainers
    {

        /// <summary>
        /// Predict a target using a linear regression model trained with the <see cref="Microsoft.ML.Runtime.Ensemble.RegressionEnsembleTrainer"/> trainer.
        /// </summary>
        /// <param name="ctx">The multiclass classification context trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TTupleInShape, TTupleOutShape, TTransformer}.Fit(DataView{TTupleInShape})"/> method is called on the
        /// <see cref="Estimator{TTupleInShape, TTupleOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained. Note that this action cannot change the
        /// result in any way; it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The set of output columns including in order the predicted per-class likelihoods (between 0 and 1, and summing up to 1), and the predicted label.</returns>
        public static Scalar<float> RegressionEnsembleTrainer(this RegressionContext.RegressionTrainers ctx,
            Scalar<float> label,
            Vector<float> features,
            Action<RegressionEnsembleTrainer.Arguments> advancedSettings = null,
            Action<IPredictorProducing<float>> onFit = null)
        {
            //LbfgsStaticUtils.ValidateParams(label, features, weights, l1Weight, l2Weight, optimizationTolerance, memorySize, enoforceNoNegativity, onFit);

            var rec = new TrainerEstimatorReconciler.Regression(
                (env, labelName, featuresName, weightsName) =>
                {
                    var trainer = new RegressionEnsembleTrainer(env, featuresName, labelName, advancedSettings);

                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));

                    return trainer;
                }, label, features, null);

            return rec.Score;
        }
    }

    /// <summary>
    /// MultiClass Classification trainer estimators.
    /// </summary>
    public static partial class MultiClassClassificationTrainers
    {

        /// <summary>
        /// Predict a target using a linear multiclass classification model trained with the <see cref="Microsoft.ML.Runtime.Ensemble.MulticlassDataPartitionEnsembleTrainer"/> trainer.
        /// </summary>
        /// <param name="ctx">The multiclass classification context trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TTupleInShape, TTupleOutShape, TTransformer}.Fit(DataView{TTupleInShape})"/> method is called on the
        /// <see cref="Estimator{TTupleInShape, TTupleOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained. Note that this action cannot change the
        /// result in any way; it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The set of output columns including in order the predicted per-class likelihoods (between 0 and 1, and summing up to 1), and the predicted label.</returns>
        public static (Vector<float> score, Key<uint, TVal> predictedLabel)
            MulticlassDataPartitionEnsembleTrainer<TVal>(this MulticlassClassificationContext.MulticlassClassificationTrainers ctx,
            Key<uint, TVal> label,
            Vector<float> features,
            Action<MulticlassDataPartitionEnsembleTrainer.Arguments> advancedSettings = null,
            Action<EnsembleMultiClassPredictor> onFit = null)
        {
            //LbfgsStaticUtils.ValidateParams(label, features, weights, l1Weight, l2Weight, optimizationTolerance, memorySize, enoforceNoNegativity, onFit);

            var rec = new TrainerEstimatorReconciler.MulticlassClassifier<TVal>(
                (env, labelName, featuresName, weightsName) =>
                {
                    var trainer = new MulticlassDataPartitionEnsembleTrainer(env, featuresName, labelName, advancedSettings);

                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                    return trainer;
                }, label, features, null);

            return rec.Output;
        }

    }

    //internal static class LbfgsStaticUtils
    //{

    //    internal static void ValidateParams(PipelineColumn label,
    //        Vector<float> features,
    //        Scalar<float> weights = null,
    //        float l1Weight = Arguments.Defaults.L1Weight,
    //        float l2Weight = Arguments.Defaults.L2Weight,
    //        float optimizationTolerance = Arguments.Defaults.OptTol,
    //        int memorySize = Arguments.Defaults.MemorySize,
    //        bool enoforceNoNegativity = Arguments.Defaults.EnforceNonNegativity,
    //        Delegate onFit = null)
    //    {
    //        Contracts.CheckValue(label, nameof(label));
    //        Contracts.CheckValue(features, nameof(features));
    //        Contracts.CheckParam(l2Weight >= 0, nameof(l2Weight), "Must be non-negative");
    //        Contracts.CheckParam(l1Weight >= 0, nameof(l1Weight), "Must be non-negative");
    //        Contracts.CheckParam(optimizationTolerance > 0, nameof(optimizationTolerance), "Must be positive");
    //        Contracts.CheckParam(memorySize > 0, nameof(memorySize), "Must be positive");
    //        Contracts.CheckValueOrNull(onFit);
    //    }
    //}
}
