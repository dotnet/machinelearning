// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers;

namespace Microsoft.ML.StaticPipe
{
    /// <summary>
    /// Binary Classification trainer estimators.
    /// </summary>
    public static class SgdStaticExtensions
    {
        /// <summary>
        ///  Predict a target using logistic regression trained with the <see cref="SgdCalibratedTrainer"/> trainer.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="label">The name of the label column.</param>
        /// <param name="features">The name of the feature column.</param>
        /// <param name="weights">The name for the example weight column.</param>
        /// <param name="numberOfIterations">The maximum number of iterations; set to 1 to simulate online learning.</param>
        /// <param name="initialLearningRate">The initial learning rate used by SGD.</param>
        /// <param name="l2Regularization">The L2 regularization constant.</param>
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
            int numberOfIterations = SgdCalibratedTrainer.Options.Defaults.NumberOfIterations,
            double initialLearningRate = SgdCalibratedTrainer.Options.Defaults.InitialLearningRate,
            float l2Regularization = SgdCalibratedTrainer.Options.Defaults.L2Regularization,
            Action<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>> onFit = null)
        {
            var rec = new TrainerEstimatorReconciler.BinaryClassifier(
                (env, labelName, featuresName, weightsName) =>
                {
                    var trainer = new SgdCalibratedTrainer(env, labelName, featuresName, weightsName, numberOfIterations, initialLearningRate, l2Regularization);

                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                    return trainer;

                }, label, features, weights);

            return rec.Output;
        }

        /// <summary>
        ///  Predict a target using logistic regression trained with the <see cref="SgdCalibratedTrainer"/> trainer.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
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
            SgdCalibratedTrainer.Options options,
            Action<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>> onFit = null)
        {
            var rec = new TrainerEstimatorReconciler.BinaryClassifier(
                (env, labelName, featuresName, weightsName) =>
                {
                    options.FeatureColumnName = featuresName;
                    options.LabelColumnName = labelName;
                    options.ExampleWeightColumnName = weightsName;

                    var trainer = new SgdCalibratedTrainer(env, options);

                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                    return trainer;

                }, label, features, weights);

            return rec.Output;
        }

        /// <summary>
        ///  Predict a target using a linear classification model trained with the <see cref="SgdNonCalibratedTrainer"/> trainer.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="label">The name of the label column.</param>
        /// <param name="features">The name of the feature column.</param>
        /// <param name="weights">The name for the example weight column.</param>
        /// <param name="numberOfIterations">The maximum number of iterations; set to 1 to simulate online learning.</param>
        /// <param name="initialLearningRate">The initial learning rate used by SGD.</param>
        /// <param name="l2Regularization">The L2 regularization constant.</param>
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
            int numberOfIterations = SgdNonCalibratedTrainer.Options.Defaults.NumberOfIterations,
            double initialLearningRate = SgdNonCalibratedTrainer.Options.Defaults.InitialLearningRate,
            float l2Regularization = SgdNonCalibratedTrainer.Options.Defaults.L2Regularization,
            IClassificationLoss loss = null,
            Action<LinearBinaryModelParameters> onFit = null)
        {
            var rec = new TrainerEstimatorReconciler.BinaryClassifierNoCalibration(
                (env, labelName, featuresName, weightsName) =>
                {
                    var trainer = new SgdNonCalibratedTrainer(env, labelName, featuresName, weightsName,
                        numberOfIterations, initialLearningRate, l2Regularization, loss);

                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                    return trainer;

                }, label, features, weights);

            return rec.Output;
        }

        /// <summary>
        ///  Predict a target using a linear classification model trained with the <see cref="SgdNonCalibratedTrainer"/> trainer.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
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
            SgdNonCalibratedTrainer.Options options,
            Action<LinearBinaryModelParameters> onFit = null)
        {
            var rec = new TrainerEstimatorReconciler.BinaryClassifierNoCalibration(
                (env, labelName, featuresName, weightsName) =>
                {
                    options.FeatureColumnName = featuresName;
                    options.LabelColumnName = labelName;
                    options.ExampleWeightColumnName = weightsName;

                    var trainer = new SgdNonCalibratedTrainer(env, options);

                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                    return trainer;

                }, label, features, weights);

            return rec.Output;
        }
    }
}
