// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.StaticPipe.Runtime;

namespace Microsoft.ML.StaticPipe
{
    /// <summary>
    /// Extension methods and utilities for instantiating SDCA trainer estimators inside statically typed pipelines.
    /// </summary>
    public static partial class RegressionTrainers
    {
        /// <summary>
        /// Predict a target using a linear regression model trained with the SDCA trainer.
        /// </summary>
        /// <param name="ctx">The regression context trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="l2Const">The L2 regularization hyperparameter.</param>
        /// <param name="l1Threshold">The L1 regularization hyperparameter. Higher values will tend to lead to more sparse model.</param>
        /// <param name="maxIterations">The maximum number of passes to perform over the data.</param>
        /// <param name="loss">The custom loss, if unspecified will be <see cref="SquaredLossSDCARegressionLossFunction"/>.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained.  Note that this action cannot change the result in any way; it is only a way for the caller to
        /// be informed about what was learnt.</param>
        /// <returns>The predicted output.</returns>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[SDCA](../../../docs/samples/Microsoft.ML.Samples/Trainers.cs?range=5-8,12-70 "The SDCA regression example.")]
        /// ]]></format>
        /// </example>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[SDCA](/Microsoft.ML.Samples/Trainers.cs?range=5-8,12-70 "The SDCA regression example.")]
        /// ]]></format>
        /// </example>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[SDCA](~/Microsoft.ML.Samples/Trainers.cs?range=5-8,12-70 "The SDCA regression example.")]
        /// ]]></format>
        /// </example>
        public static Scalar<float> Sdca(this RegressionContext.RegressionTrainers ctx,
            Scalar<float> label, Vector<float> features, Scalar<float> weights = null,
            float? l2Const = null,
            float? l1Threshold = null,
            int? maxIterations = null,
            ISupportSdcaRegressionLoss loss = null,
            Action<LinearRegressionPredictor> onFit = null)
        {
            Contracts.CheckValue(label, nameof(label));
            Contracts.CheckValue(features, nameof(features));
            Contracts.CheckValueOrNull(weights);
            Contracts.CheckParam(!(l2Const < 0), nameof(l2Const), "Must not be negative, if specified.");
            Contracts.CheckParam(!(l1Threshold < 0), nameof(l1Threshold), "Must not be negative, if specified.");
            Contracts.CheckParam(!(maxIterations < 1), nameof(maxIterations), "Must be positive if specified");
            Contracts.CheckValueOrNull(loss);
            Contracts.CheckValueOrNull(onFit);

            var args = new SdcaRegressionTrainer.Arguments()
            {
                L2Const = l2Const,
                L1Threshold = l1Threshold,
                MaxIterations = maxIterations
            };
            if (loss != null)
                args.LossFunction = new TrivialRegressionLossFactory(loss);

            var rec = new TrainerEstimatorReconciler.Regression(
                (env, labelName, featuresName, weightsName) =>
                {
                    var trainer = new SdcaRegressionTrainer(env, args, featuresName, labelName, weightsName);
                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                    return trainer;
                }, label, features, weights);

            return rec.Score;
        }

        private sealed class TrivialRegressionLossFactory : ISupportSdcaRegressionLossFactory
        {
            private readonly ISupportSdcaRegressionLoss _loss;

            public TrivialRegressionLossFactory(ISupportSdcaRegressionLoss loss)
            {
                _loss = loss;
            }

            public ISupportSdcaRegressionLoss CreateComponent(IHostEnvironment env)
            {
                return _loss;
            }
        }
    }
    public static partial class BinaryClassificationTrainers
    {

        /// <summary>
        /// Predict a target using a linear binary classification model trained with the SDCA trainer, and log-loss.
        /// </summary>
        /// <param name="ctx">The binary classification context trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="l2Const">The L2 regularization hyperparameter.</param>
        /// <param name="l1Threshold">The L1 regularization hyperparameter. Higher values will tend to lead to more sparse model.</param>
        /// <param name="maxIterations">The maximum number of passes to perform over the data.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained, as well as the calibrator on top of that model. Note that this action cannot change the
        /// result in any way; it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The set of output columns including in order the predicted binary classification score (which will range
        /// from negative to positive infinity), the calibrated prediction (from 0 to 1), and the predicted label.</returns>
        public static (Scalar<float> score, Scalar<float> probability, Scalar<bool> predictedLabel) Sdca(
                    this BinaryClassificationContext.BinaryClassificationTrainers ctx,
                    Scalar<bool> label, Vector<float> features, Scalar<float> weights = null,
                    float? l2Const = null,
                    float? l1Threshold = null,
                    int? maxIterations = null,
                    Action<LinearBinaryPredictor, ParameterMixingCalibratedPredictor> onFit = null)
        {
            Contracts.CheckValue(label, nameof(label));
            Contracts.CheckValue(features, nameof(features));
            Contracts.CheckValueOrNull(weights);
            Contracts.CheckParam(!(l2Const < 0), nameof(l2Const), "Must not be negative, if specified.");
            Contracts.CheckParam(!(l1Threshold < 0), nameof(l1Threshold), "Must not be negative, if specified.");
            Contracts.CheckParam(!(maxIterations < 1), nameof(maxIterations), "Must be positive if specified");
            Contracts.CheckValueOrNull(onFit);

            var args = new LinearClassificationTrainer.Arguments()
            {
                L2Const = l2Const,
                L1Threshold = l1Threshold,
                MaxIterations = maxIterations,
            };

            var rec = new TrainerEstimatorReconciler.BinaryClassifier(
                (env, labelName, featuresName, weightsName) =>
                {
                    var trainer = new LinearClassificationTrainer(env, args, featuresName, labelName, weightsName);
                    if (onFit != null)
                    {
                        return trainer.WithOnFitDelegate(trans =>
                        {
                            // Under the default log-loss we assume a calibrated predictor.
                            var model = trans.Model;
                            var cali = (ParameterMixingCalibratedPredictor)model;
                            var pred = (LinearBinaryPredictor)cali.SubPredictor;
                            onFit(pred, cali);
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
        /// <param name="ctx">The binary classification context trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="loss">The custom loss.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="l2Const">The L2 regularization hyperparameter.</param>
        /// <param name="l1Threshold">The L1 regularization hyperparameter. Higher values will tend to lead to more sparse model.</param>
        /// <param name="maxIterations">The maximum number of passes to perform over the data.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained, as well as the calibrator on top of that model. Note that this action cannot change the
        /// result in any way; it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The set of output columns including in order the predicted binary classification score (which will range
        /// from negative to positive infinity), and the predicted label.</returns>
        /// <seealso cref="Sdca(BinaryClassificationContext.BinaryClassificationTrainers, Scalar{bool}, Vector{float}, Scalar{float}, float?, float?, int?, Action{LinearBinaryPredictor, ParameterMixingCalibratedPredictor})"/>
        public static (Scalar<float> score, Scalar<bool> predictedLabel) Sdca(
                this BinaryClassificationContext.BinaryClassificationTrainers ctx,
                Scalar<bool> label, Vector<float> features,
                ISupportSdcaClassificationLoss loss,
                Scalar<float> weights = null,
                float? l2Const = null,
                float? l1Threshold = null,
                int? maxIterations = null,
                Action<LinearBinaryPredictor> onFit = null
            )
        {
            Contracts.CheckValue(label, nameof(label));
            Contracts.CheckValue(features, nameof(features));
            Contracts.CheckValue(loss, nameof(loss));
            Contracts.CheckValueOrNull(weights);
            Contracts.CheckParam(!(l2Const < 0), nameof(l2Const), "Must not be negative, if specified.");
            Contracts.CheckParam(!(l1Threshold < 0), nameof(l1Threshold), "Must not be negative, if specified.");
            Contracts.CheckParam(!(maxIterations < 1), nameof(maxIterations), "Must be positive if specified");
            Contracts.CheckValueOrNull(onFit);

            bool hasProbs = loss is LogLoss;

            var args = new LinearClassificationTrainer.Arguments()
            {
                L2Const = l2Const,
                L1Threshold = l1Threshold,
                MaxIterations = maxIterations,
                LossFunction = new TrivialSdcaClassificationLossFactory(loss)
            };

            var rec = new TrainerEstimatorReconciler.BinaryClassifierNoCalibration(
                (env, labelName, featuresName, weightsName) =>
                {
                    var trainer = new LinearClassificationTrainer(env, args, featuresName, labelName, weightsName);
                    if (onFit != null)
                    {
                        return trainer.WithOnFitDelegate(trans =>
                        {
                            var model = trans.Model;
                            if (model is ParameterMixingCalibratedPredictor cali)
                                onFit((LinearBinaryPredictor)cali.SubPredictor);
                            else
                                onFit((LinearBinaryPredictor)model);
                        });
                    }
                    return trainer;
                }, label, features, weights, hasProbs);

            return rec.Output;
        }
    }

    public static partial class MultiClassClassificationTrainers {

        /// <summary>
        /// Predict a target using a linear multiclass classification model trained with the SDCA trainer.
        /// </summary>
        /// <param name="ctx">The multiclass classification context trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="loss">The custom loss.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="l2Const">The L2 regularization hyperparameter.</param>
        /// <param name="l1Threshold">The L1 regularization hyperparameter. Higher values will tend to lead to more sparse model.</param>
        /// <param name="maxIterations">The maximum number of passes to perform over the data.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained. Note that this action cannot change the
        /// result in any way; it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The set of output columns including in order the predicted per-class likelihoods (between 0 and 1, and summing up to 1), and the predicted label.</returns>
        public static (Vector<float> score, Key<uint, TVal> predictedLabel)
                Sdca<TVal>(this MulticlassClassificationContext.MulticlassClassificationTrainers ctx,
                    Key<uint, TVal> label,
                    Vector<float> features,
                    ISupportSdcaClassificationLoss loss = null,
                    Scalar<float> weights = null,
                    float? l2Const = null,
                    float? l1Threshold = null,
                    int? maxIterations = null,
                    Action<MulticlassLogisticRegressionPredictor> onFit = null)
        {
            Contracts.CheckValue(label, nameof(label));
            Contracts.CheckValue(features, nameof(features));
            Contracts.CheckValueOrNull(loss);
            Contracts.CheckValueOrNull(weights);
            Contracts.CheckParam(!(l2Const < 0), nameof(l2Const), "Must not be negative, if specified.");
            Contracts.CheckParam(!(l1Threshold < 0), nameof(l1Threshold), "Must not be negative, if specified.");
            Contracts.CheckParam(!(maxIterations < 1), nameof(maxIterations), "Must be positive if specified");
            Contracts.CheckValueOrNull(onFit);

            var args = new SdcaMultiClassTrainer.Arguments
            {
                L2Const = l2Const,
                L1Threshold = l1Threshold,
                MaxIterations = maxIterations
            };

            if (loss != null)
                args.LossFunction = new TrivialSdcaClassificationLossFactory(loss);

            var rec = new TrainerEstimatorReconciler.MulticlassClassifier<TVal>(
                (env, labelName, featuresName, weightsName) =>
                {
                    var trainer = new SdcaMultiClassTrainer(env, args, featuresName, labelName, weightsName);
                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                    return trainer;
                }, label, features, weights);

            return rec.Output;
        }
    }

    internal sealed class TrivialSdcaClassificationLossFactory : ISupportSdcaClassificationLossFactory
    {
        private readonly ISupportSdcaClassificationLoss _loss;

        public TrivialSdcaClassificationLossFactory(ISupportSdcaClassificationLoss loss)
        {
            _loss = loss;
        }

        public ISupportSdcaClassificationLoss CreateComponent(IHostEnvironment env)
        {
            return _loss;
        }
    }
}
