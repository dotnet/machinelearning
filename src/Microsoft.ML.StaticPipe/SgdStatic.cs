// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Calibration;
using Microsoft.ML.StaticPipe.Runtime;
using Microsoft.ML.Trainers;

namespace Microsoft.ML.StaticPipe
{
    /// <summary>
    /// Binary Classification trainer estimators.
    /// </summary>
    public static class SgdStaticExtensions
    {
        /// <summary>
        ///  Predict a target using logistic regression trained with the <see cref="SgdBinaryTrainer"/> trainer.
        /// </summary>
        /// <param name="catalog">The binary classificaiton catalog trainer object.</param>
        /// <param name="label">The name of the label column.</param>
        /// <param name="features">The name of the feature column.</param>
        /// <param name="weights">The name for the example weight column.</param>
        /// <param name="maxIterations">The maximum number of iterations; set to 1 to simulate online learning.</param>
        /// <param name="initLearningRate">The initial learning rate used by SGD.</param>
        /// <param name="l2Weight">The L2 regularization constant.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TTupleInShape, TTupleOutShape, TTransformer}.Fit(DataView{TTupleInShape})"/> method is called on the
        /// <see cref="Estimator{TTupleInShape, TTupleOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained.  Note that this action cannot change the result in any way; it is only a way for the caller to
        /// be informed about what was learnt.</param>
        /// <returns>The predicted output.</returns>
        public static (Scalar<float> score, Scalar<float> probability, Scalar<bool> predictedLabel) StochasticGradientDescentClassificationTrainer(
            this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            Scalar<bool> label,
            Vector<float> features,
            Scalar<float> weights = null,
            int maxIterations = SgdBinaryTrainer.Options.Defaults.MaxIterations,
            double initLearningRate = SgdBinaryTrainer.Options.Defaults.InitLearningRate,
            float l2Weight = SgdBinaryTrainer.Options.Defaults.L2Weight,
            Action<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>> onFit = null)
        {
            var rec = new TrainerEstimatorReconciler.BinaryClassifier(
                (env, labelName, featuresName, weightsName) =>
                {
                    var trainer = new SgdBinaryTrainer(env, labelName, featuresName, weightsName, maxIterations, initLearningRate, l2Weight);

                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                    return trainer;

                }, label, features, weights);

            return rec.Output;
        }

        /// <summary>
        ///  Predict a target using logistic regression trained with the <see cref="SgdBinaryTrainer"/> trainer.
        /// </summary>
        /// <param name="catalog">The binary classificaiton catalog trainer object.</param>
        /// <param name="label">The name of the label column.</param>
        /// <param name="features">The name of the feature column.</param>
        /// <param name="weights">The name for the example weight column.</param>
        /// <param name="options">Advanced arguments to the algorithm.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TTupleInShape, TTupleOutShape, TTransformer}.Fit(DataView{TTupleInShape})"/> method is called on the
        /// <see cref="Estimator{TTupleInShape, TTupleOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained.  Note that this action cannot change the result in any way; it is only a way for the caller to
        /// be informed about what was learnt.</param>
        /// <returns>The predicted output.</returns>
        public static (Scalar<float> score, Scalar<float> probability, Scalar<bool> predictedLabel) StochasticGradientDescentClassificationTrainer(
            this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            Scalar<bool> label,
            Vector<float> features,
            Scalar<float> weights,
            SgdBinaryTrainer.Options options,
            Action<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>> onFit = null)
        {
            var rec = new TrainerEstimatorReconciler.BinaryClassifier(
                (env, labelName, featuresName, weightsName) =>
                {
                    options.FeatureColumn = featuresName;
                    options.LabelColumn = labelName;
                    options.WeightColumn = weightsName != null ? Optional<string>.Explicit(weightsName) : Optional<string>.Implicit(DefaultColumnNames.Weight);

                    var trainer = new SgdBinaryTrainer(env, options);

                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                    return trainer;

                }, label, features, weights);

            return rec.Output;
        }

        /// <summary>
        ///  Predict a target using a linear classification model trained with the <see cref="SgdNonCalibratedBinaryTrainer"/> trainer.
        /// </summary>
        /// <param name="catalog">The binary classificaiton catalog trainer object.</param>
        /// <param name="label">The name of the label column.</param>
        /// <param name="features">The name of the feature column.</param>
        /// <param name="weights">The name for the example weight column.</param>
        /// <param name="maxIterations">The maximum number of iterations; set to 1 to simulate online learning.</param>
        /// <param name="initLearningRate">The initial learning rate used by SGD.</param>
        /// <param name="l2Weight">The L2 regularization constant.</param>
        /// <param name="loss">The loss function to use.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TTupleInShape, TTupleOutShape, TTransformer}.Fit(DataView{TTupleInShape})"/> method is called on the
        /// <see cref="Estimator{TTupleInShape, TTupleOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained.  Note that this action cannot change the result in any way; it is only a way for the caller to
        /// be informed about what was learnt.</param>
        /// <returns>The predicted output.</returns>
        public static (Scalar<float> score, Scalar<bool> predictedLabel) StochasticGradientDescentNonCalibratedClassificationTrainer(
            this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            Scalar<bool> label,
            Vector<float> features,
            Scalar<float> weights = null,
            int maxIterations = SgdNonCalibratedBinaryTrainer.Options.Defaults.MaxIterations,
            double initLearningRate = SgdNonCalibratedBinaryTrainer.Options.Defaults.InitLearningRate,
            float l2Weight = SgdNonCalibratedBinaryTrainer.Options.Defaults.L2Weight,
            IClassificationLoss loss = null,
            Action<LinearBinaryModelParameters> onFit = null)
        {
            var rec = new TrainerEstimatorReconciler.BinaryClassifierNoCalibration(
                (env, labelName, featuresName, weightsName) =>
                {
                    var trainer = new SgdNonCalibratedBinaryTrainer(env, labelName, featuresName, weightsName,
                        maxIterations, initLearningRate, l2Weight, loss);

                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                    return trainer;

                }, label, features, weights);

            return rec.Output;
        }

        /// <summary>
        ///  Predict a target using a linear classification model trained with the <see cref="SgdNonCalibratedBinaryTrainer"/> trainer.
        /// </summary>
        /// <param name="catalog">The binary classificaiton catalog trainer object.</param>
        /// <param name="label">The name of the label column.</param>
        /// <param name="features">The name of the feature column.</param>
        /// <param name="weights">The name for the example weight column.</param>
        /// <param name="options">Advanced arguments to the algorithm.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TTupleInShape, TTupleOutShape, TTransformer}.Fit(DataView{TTupleInShape})"/> method is called on the
        /// <see cref="Estimator{TTupleInShape, TTupleOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained.  Note that this action cannot change the result in any way; it is only a way for the caller to
        /// be informed about what was learnt.</param>
        /// <returns>The predicted output.</returns>
        public static (Scalar<float> score, Scalar<bool> predictedLabel) StochasticGradientDescentNonCalibratedClassificationTrainer(
            this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            Scalar<bool> label,
            Vector<float> features,
            Scalar<float> weights,
            SgdNonCalibratedBinaryTrainer.Options options,
            Action<LinearBinaryModelParameters> onFit = null)
        {
            var rec = new TrainerEstimatorReconciler.BinaryClassifierNoCalibration(
                (env, labelName, featuresName, weightsName) =>
                {
                    options.FeatureColumn = featuresName;
                    options.LabelColumn = labelName;
                    options.WeightColumn = weightsName != null ? Optional<string>.Explicit(weightsName) : Optional<string>.Implicit(DefaultColumnNames.Weight);

                    var trainer = new SgdNonCalibratedBinaryTrainer(env, options);

                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                    return trainer;

                }, label, features, weights);

            return rec.Output;
        }
    }
}
