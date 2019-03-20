// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;

namespace Microsoft.ML.StaticPipe
{
    /// <summary>
    /// Extension methods and utilities for instantiating SDCA trainer estimators inside statically typed pipelines.
    /// </summary>
    public static class SdcaStaticExtensions
    {
        /// <summary>
        /// Predict a target using a linear regression model trained with the SDCA trainer.
        /// </summary>
        /// <param name="catalog">The regression catalog trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="l2Regularization">The L2 regularization hyperparameter.</param>
        /// <param name="l1Threshold">The L1 regularization hyperparameter. Higher values will tend to lead to more sparse model.</param>
        /// <param name="numberOfIterations">The maximum number of passes to perform over the data.</param>
        /// <param name="loss">The custom loss, if unspecified will be <see cref="SquaredLoss"/>.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained.  Note that this action cannot change the result in any way; it is only a way for the caller to
        /// be informed about what was learnt.</param>
        /// <returns>The predicted output.</returns>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[SDCA](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Static/SDCARegression.cs)]
        /// ]]></format>
        /// </example>
        public static Scalar<float> Sdca(this RegressionCatalog.RegressionTrainers catalog,
            Scalar<float> label, Vector<float> features, Scalar<float> weights = null,
            float? l2Regularization = null,
            float? l1Threshold = null,
            int? numberOfIterations = null,
            ISupportSdcaRegressionLoss loss = null,
            Action<LinearRegressionModelParameters> onFit = null)
        {
            Contracts.CheckValue(label, nameof(label));
            Contracts.CheckValue(features, nameof(features));
            Contracts.CheckValueOrNull(weights);
            Contracts.CheckParam(!(l2Regularization < 0), nameof(l2Regularization), "Must not be negative, if specified.");
            Contracts.CheckParam(!(l1Threshold < 0), nameof(l1Threshold), "Must not be negative, if specified.");
            Contracts.CheckParam(!(numberOfIterations < 1), nameof(numberOfIterations), "Must be positive if specified");
            Contracts.CheckValueOrNull(loss);
            Contracts.CheckValueOrNull(onFit);

            var rec = new TrainerEstimatorReconciler.Regression(
                (env, labelName, featuresName, weightsName) =>
                {
                    var trainer = new SdcaRegressionTrainer(env, labelName, featuresName, weightsName, loss, l2Regularization, l1Threshold, numberOfIterations);
                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                    return trainer;
                }, label, features, weights);

            return rec.Score;
        }

        /// <summary>
        /// Predict a target using a linear regression model trained with the SDCA trainer.
        /// </summary>
        /// <param name="catalog">The regression catalog trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="options">Advanced arguments to the algorithm.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained.  Note that this action cannot change the result in any way; it is only a way for the caller to
        /// be informed about what was learnt.</param>
        /// <returns>The predicted output.</returns>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[SDCA](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Static/SDCARegression.cs)]
        /// ]]></format>
        /// </example>
        public static Scalar<float> Sdca(this RegressionCatalog.RegressionTrainers catalog,
            Scalar<float> label, Vector<float> features, Scalar<float> weights,
            SdcaRegressionTrainer.Options options,
            Action<LinearRegressionModelParameters> onFit = null)
        {
            Contracts.CheckValue(label, nameof(label));
            Contracts.CheckValue(features, nameof(features));
            Contracts.CheckValueOrNull(weights);
            Contracts.CheckValueOrNull(options);
            Contracts.CheckValueOrNull(onFit);

            var rec = new TrainerEstimatorReconciler.Regression(
                (env, labelName, featuresName, weightsName) =>
                {
                    options.LabelColumnName = labelName;
                    options.FeatureColumnName = featuresName;

                    var trainer = new SdcaRegressionTrainer(env, options);
                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                    return trainer;
                }, label, features, weights);

            return rec.Score;
        }

        /// <summary>
        /// Predict a target using a linear binary classification model trained with the SDCA trainer, and log-loss.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="l2Regularization">The L2 regularization hyperparameter.</param>
        /// <param name="l1Threshold">The L1 regularization hyperparameter. Higher values will tend to lead to more sparse model.</param>
        /// <param name="numberOfIterations">The maximum number of passes to perform over the data.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained, as well as the calibrator on top of that model. Note that this action cannot change the
        /// result in any way; it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The set of output columns including in order the predicted binary classification score (which will range
        /// from negative to positive infinity), the calibrated prediction (from 0 to 1), and the predicted label.</returns>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[SDCA](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Static/SDCABinaryClassification.cs)]
        /// ]]></format>
        /// </example>
        public static (Scalar<float> score, Scalar<float> probability, Scalar<bool> predictedLabel) Sdca(
            this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            Scalar<bool> label, Vector<float> features, Scalar<float> weights = null,
            float? l2Regularization = null,
            float? l1Threshold = null,
            int? numberOfIterations = null,
            Action<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>> onFit = null)
        {
            Contracts.CheckValue(label, nameof(label));
            Contracts.CheckValue(features, nameof(features));
            Contracts.CheckValueOrNull(weights);
            Contracts.CheckParam(!(l2Regularization < 0), nameof(l2Regularization), "Must not be negative, if specified.");
            Contracts.CheckParam(!(l1Threshold < 0), nameof(l1Threshold), "Must not be negative, if specified.");
            Contracts.CheckParam(!(numberOfIterations < 1), nameof(numberOfIterations), "Must be positive if specified");
            Contracts.CheckValueOrNull(onFit);

            var rec = new TrainerEstimatorReconciler.BinaryClassifier(
                (env, labelName, featuresName, weightsName) =>
                {
                    var trainer = new SdcaCalibratedBinaryTrainer(env, labelName, featuresName, weightsName, l2Regularization, l1Threshold, numberOfIterations);
                    if (onFit != null)
                    {
                        return trainer.WithOnFitDelegate(trans =>
                        {
                            onFit(trans.Model);
                        });
                    }
                    return trainer;
                }, label, features, weights);

            return rec.Output;
        }

        /// <summary>
        /// Predict a target using a linear binary classification model trained with the SDCA trainer, and log-loss.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="options">Advanced arguments to the algorithm.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained, as well as the calibrator on top of that model. Note that this action cannot change the
        /// result in any way; it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The set of output columns including in order the predicted binary classification score (which will range
        /// from negative to positive infinity), the calibrated prediction (from 0 to 1), and the predicted label.</returns>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[SDCA](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Static/SDCABinaryClassification.cs)]
        /// ]]></format>
        /// </example>
        public static (Scalar<float> score, Scalar<float> probability, Scalar<bool> predictedLabel) Sdca(
            this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            Scalar<bool> label, Vector<float> features, Scalar<float> weights,
            SdcaCalibratedBinaryTrainer.Options options,
            Action<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>> onFit = null)
        {
            Contracts.CheckValue(label, nameof(label));
            Contracts.CheckValue(features, nameof(features));
            Contracts.CheckValueOrNull(weights);
            Contracts.CheckValueOrNull(options);
            Contracts.CheckValueOrNull(onFit);

            var rec = new TrainerEstimatorReconciler.BinaryClassifier(
                (env, labelName, featuresName, weightsName) =>
                {
                    options.LabelColumnName = labelName;
                    options.FeatureColumnName = featuresName;

                    var trainer = new SdcaCalibratedBinaryTrainer(env, options);
                    if (onFit != null)
                    {
                        return trainer.WithOnFitDelegate(trans =>
                        {
                            onFit(trans.Model);
                        });
                    }
                    return trainer;
                }, label, features, weights);

            return rec.Output;
        }

        /// <summary>
        /// Predict a target using a linear binary classification model trained with the SDCA trainer, and a custom loss.
        /// Note that because we cannot be sure that all loss functions will produce naturally calibrated outputs, setting
        /// a custom loss function will not produce a calibrated probability column.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="loss">The custom loss.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="l2Regularization">The L2 regularization hyperparameter.</param>
        /// <param name="l1Threshold">The L1 regularization hyperparameter. Higher values will tend to lead to more sparse model.</param>
        /// <param name="numberOfIterations">The maximum number of passes to perform over the data.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained, as well as the calibrator on top of that model. Note that this action cannot change the
        /// result in any way; it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The set of output columns including in order the predicted binary classification score (which will range
        /// from negative to positive infinity), and the predicted label.</returns>
        public static (Scalar<float> score, Scalar<bool> predictedLabel) SdcaNonCalibrated(
            this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            Scalar<bool> label, Vector<float> features,
            ISupportSdcaClassificationLoss loss,
            Scalar<float> weights = null,
            float? l2Regularization = null,
            float? l1Threshold = null,
            int? numberOfIterations = null,
            Action<LinearBinaryModelParameters> onFit = null)
        {
            Contracts.CheckValue(label, nameof(label));
            Contracts.CheckValue(features, nameof(features));
            Contracts.CheckValue(loss, nameof(loss));
            Contracts.CheckValueOrNull(weights);
            Contracts.CheckParam(!(l2Regularization < 0), nameof(l2Regularization), "Must not be negative, if specified.");
            Contracts.CheckParam(!(l1Threshold < 0), nameof(l1Threshold), "Must not be negative, if specified.");
            Contracts.CheckParam(!(numberOfIterations < 1), nameof(numberOfIterations), "Must be positive if specified");
            Contracts.CheckValueOrNull(onFit);

            var rec = new TrainerEstimatorReconciler.BinaryClassifierNoCalibration(
                (env, labelName, featuresName, weightsName) =>
                {
                    var trainer = new SdcaNonCalibratedBinaryTrainer(env, labelName, featuresName, weightsName, loss, l2Regularization, l1Threshold, numberOfIterations);
                    if (onFit != null)
                    {
                        return trainer.WithOnFitDelegate(trans =>
                        {
                            onFit(trans.Model);
                        });
                    }
                    return trainer;
                }, label, features, weights);

            return rec.Output;
        }

        /// <summary>
        /// Predict a target using a linear binary classification model trained with the SDCA trainer, and a custom loss.
        /// Note that because we cannot be sure that all loss functions will produce naturally calibrated outputs, setting
        /// a custom loss function will not produce a calibrated probability column.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="loss">The custom loss.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="options">Advanced arguments to the algorithm.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained, as well as the calibrator on top of that model. Note that this action cannot change the
        /// result in any way; it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The set of output columns including in order the predicted binary classification score (which will range
        /// from negative to positive infinity), and the predicted label.</returns>
        public static (Scalar<float> score, Scalar<bool> predictedLabel) SdcaNonCalibrated(
            this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            Scalar<bool> label, Vector<float> features, Scalar<float> weights,
            ISupportSdcaClassificationLoss loss,
            SdcaNonCalibratedBinaryTrainer.Options options,
            Action<LinearBinaryModelParameters> onFit = null)
        {
            Contracts.CheckValue(label, nameof(label));
            Contracts.CheckValue(features, nameof(features));
            Contracts.CheckValueOrNull(weights);
            Contracts.CheckValueOrNull(options);
            Contracts.CheckValueOrNull(onFit);

            var rec = new TrainerEstimatorReconciler.BinaryClassifierNoCalibration(
                (env, labelName, featuresName, weightsName) =>
                {
                    options.FeatureColumnName = featuresName;
                    options.LabelColumnName = labelName;

                    var trainer = new SdcaNonCalibratedBinaryTrainer(env, options);
                    if (onFit != null)
                    {
                        return trainer.WithOnFitDelegate(trans =>
                        {
                            onFit(trans.Model);
                        });
                    }
                    return trainer;
                }, label, features, weights);

            return rec.Output;
        }

        /// <summary>
        /// Predict a target using a maximum entropy classification model trained with the SDCA trainer.
        /// </summary>
        /// <param name="catalog">The multiclass classification catalog trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="l2Regularization">The L2 regularization hyperparameter.</param>
        /// <param name="l1Threshold">The L1 regularization hyperparameter. Higher values will tend to lead to more sparse model.</param>
        /// <param name="numberOfIterations">The maximum number of passes to perform over the data.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained. Note that this action cannot change the
        /// result in any way; it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The set of output columns including in order the predicted per-class likelihoods (between 0 and 1, and summing up to 1), and the predicted label.</returns>
        public static (Vector<float> score, Key<uint, TVal> predictedLabel) Sdca<TVal>(
            this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            Key<uint, TVal> label,
            Vector<float> features,
            Scalar<float> weights = null,
            float? l2Regularization = null,
            float? l1Threshold = null,
            int? numberOfIterations = null,
            Action<MaximumEntropyModelParameters> onFit = null)
        {
            Contracts.CheckValue(label, nameof(label));
            Contracts.CheckValue(features, nameof(features));
            Contracts.CheckValueOrNull(weights);
            Contracts.CheckParam(!(l2Regularization < 0), nameof(l2Regularization), "Must not be negative, if specified.");
            Contracts.CheckParam(!(l1Threshold < 0), nameof(l1Threshold), "Must not be negative, if specified.");
            Contracts.CheckParam(!(numberOfIterations < 1), nameof(numberOfIterations), "Must be positive if specified");
            Contracts.CheckValueOrNull(onFit);

            var rec = new TrainerEstimatorReconciler.MulticlassClassificationReconciler<TVal>(
                (env, labelName, featuresName, weightsName) =>
                {
                    var trainer = new SdcaCalibratedMulticlassTrainer(env, labelName, featuresName, weightsName, l2Regularization, l1Threshold, numberOfIterations);
                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                    return trainer;
                }, label, features, weights);

            return rec.Output;
        }

        /// <summary>
        /// Predict a target using a maximum entropy classification model trained with the SDCA trainer.
        /// </summary>
        /// <param name="catalog">The multiclass classification catalog trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="options">Advanced arguments to the algorithm.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained. Note that this action cannot change the
        /// result in any way; it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The set of output columns including in order the predicted per-class likelihoods (between 0 and 1, and summing up to 1), and the predicted label.</returns>
        public static (Vector<float> score, Key<uint, TVal> predictedLabel) Sdca<TVal>(
            this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            Key<uint, TVal> label,
            Vector<float> features,
            Scalar<float> weights,
            SdcaCalibratedMulticlassTrainer.Options options,
            Action<MaximumEntropyModelParameters> onFit = null)
        {
            Contracts.CheckValue(label, nameof(label));
            Contracts.CheckValue(features, nameof(features));
            Contracts.CheckValueOrNull(weights);
            Contracts.CheckValueOrNull(options);
            Contracts.CheckValueOrNull(onFit);

            var rec = new TrainerEstimatorReconciler.MulticlassClassificationReconciler<TVal>(
                (env, labelName, featuresName, weightsName) =>
                {
                    options.LabelColumnName = labelName;
                    options.FeatureColumnName = featuresName;

                    var trainer = new SdcaCalibratedMulticlassTrainer(env, options);
                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                    return trainer;
                }, label, features, weights);

            return rec.Output;
        }

        /// <summary>
        /// Predict a target using a linear multiclass classification model trained with the SDCA trainer.
        /// </summary>
        /// <param name="catalog">The multiclass classification catalog trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="loss">The custom loss, for example, <see cref="HingeLoss"/>.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="l2Regularization">The L2 regularization hyperparameter.</param>
        /// <param name="l1Threshold">The L1 regularization hyperparameter. Higher values will tend to lead to more sparse model.</param>
        /// <param name="numberOfIterations">The maximum number of passes to perform over the data.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained. Note that this action cannot change the
        /// result in any way; it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The set of output columns including in order the predicted per-class likelihoods (between 0 and 1, and summing up to 1), and the predicted label.</returns>
        public static (Vector<float> score, Key<uint, TVal> predictedLabel) SdcaNonCalibrated<TVal>(
            this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            Key<uint, TVal> label,
            Vector<float> features,
            ISupportSdcaClassificationLoss loss,
            Scalar<float> weights = null,
            float? l2Regularization = null,
            float? l1Threshold = null,
            int? numberOfIterations = null,
            Action<LinearMulticlassModelParameters> onFit = null)
        {
            Contracts.CheckValue(label, nameof(label));
            Contracts.CheckValue(features, nameof(features));
            Contracts.CheckValueOrNull(loss);
            Contracts.CheckValueOrNull(weights);
            Contracts.CheckParam(!(l2Regularization < 0), nameof(l2Regularization), "Must not be negative, if specified.");
            Contracts.CheckParam(!(l1Threshold < 0), nameof(l1Threshold), "Must not be negative, if specified.");
            Contracts.CheckParam(!(numberOfIterations < 1), nameof(numberOfIterations), "Must be positive if specified");
            Contracts.CheckValueOrNull(onFit);

            var rec = new TrainerEstimatorReconciler.MulticlassClassificationReconciler<TVal>(
                (env, labelName, featuresName, weightsName) =>
                {
                    var trainer = new SdcaNonCalibratedMulticlassTrainer(env, labelName, featuresName, weightsName, loss, l2Regularization, l1Threshold, numberOfIterations);
                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                    return trainer;
                }, label, features, weights);

            return rec.Output;
        }

        /// <summary>
        /// Predict a target using a linear multiclass classification model trained with the SDCA trainer.
        /// </summary>
        /// <param name="catalog">The multiclass classification catalog trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="options">Advanced arguments to the algorithm.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained. Note that this action cannot change the
        /// result in any way; it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The set of output columns including in order the predicted per-class likelihoods (between 0 and 1, and summing up to 1), and the predicted label.</returns>
        public static (Vector<float> score, Key<uint, TVal> predictedLabel) SdcaNonCalibrated<TVal>(
            this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            Key<uint, TVal> label,
            Vector<float> features,
            Scalar<float> weights,
            SdcaNonCalibratedMulticlassTrainer.Options options,
            Action<LinearMulticlassModelParameters> onFit = null)
        {
            Contracts.CheckValue(label, nameof(label));
            Contracts.CheckValue(features, nameof(features));
            Contracts.CheckValueOrNull(weights);
            Contracts.CheckValueOrNull(options);
            Contracts.CheckValueOrNull(onFit);

            var rec = new TrainerEstimatorReconciler.MulticlassClassificationReconciler<TVal>(
                (env, labelName, featuresName, weightsName) =>
                {
                    options.LabelColumnName = labelName;
                    options.FeatureColumnName = featuresName;

                    var trainer = new SdcaNonCalibratedMulticlassTrainer(env, options);
                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                    return trainer;
                }, label, features, weights);

            return rec.Output;
        }
    }
}
