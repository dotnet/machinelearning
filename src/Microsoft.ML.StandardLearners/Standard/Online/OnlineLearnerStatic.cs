// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.StaticPipe.Runtime;
using System;

namespace Microsoft.ML.Trainers
{
    public static class OnlineLearnerStatic
    {
        /// <summary>
        /// Predict a target using a linear binary classification model trained with the SDCA trainer, and a custom loss.
        /// Note that because we cannot be sure that all loss functions will produce naturally calibrated outputs, setting
        /// a custom loss function will not produce a calibrated probability column.
        /// </summary>
        /// <param name="ctx">The binary classification context trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="lossFunction">The custom loss.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="learningRate">The learning Rate.</param>
        /// <param name="decreaseLearningRate">Decrease learning rate as iterations progress.</param>
        /// <param name="l2RegularizerWeight">L2 Regularization Weight.</param>
        /// <param name="numIterations">Number of training iterations through the data.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TTupleInShape, TTupleOutShape, TTransformer}.Fit(DataView{TTupleInShape})"/> method is called on the
        /// <see cref="Estimator{TTupleInShape, TTupleOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained, as well as the calibrator on top of that model. Note that this action cannot change the
        /// result in any way; it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The set of output columns including in order the predicted binary classification score (which will range
        /// from negative to positive infinity), and the predicted label.</returns>
        /// <seealso cref="AveragedPerceptronTrainer"/>.
        public static (Scalar<float> score, Scalar<bool> predictedLabel) AveragedPerceptron(
                this BinaryClassificationContext.BinaryClassificationTrainers ctx,
                IClassificationLoss lossFunction,
                Scalar<bool> label, Vector<float> features, Scalar<float> weights = null,
                float learningRate = AveragedLinearArguments.DefaultAveragedArgs.LearningRate,
                bool decreaseLearningRate = AveragedLinearArguments.DefaultAveragedArgs.DecreaseLearningRate,
                float l2RegularizerWeight = AveragedLinearArguments.DefaultAveragedArgs.L2RegularizerWeight,
                int numIterations = OnlineLinearArguments.DefaultArgs.NumIterations,
                Action<LinearBinaryPredictor> onFit = null
            )
        {
            Contracts.CheckValue(label, nameof(label));
            Contracts.CheckValue(features, nameof(features));
            Contracts.CheckValue(lossFunction, nameof(lossFunction));
            Contracts.CheckValueOrNull(weights);
            Contracts.CheckParam(learningRate > 0, nameof(learningRate), "Must be positive.");
            Contracts.CheckParam(0 <= l2RegularizerWeight && l2RegularizerWeight < 0.5, nameof(l2RegularizerWeight), "must be in range [0, 0.5)");

            Contracts.CheckParam(numIterations > 1, nameof(numIterations), "Must be greater than one, if specified.");
            Contracts.CheckValueOrNull(onFit);

            bool hasProbs = lossFunction is HingeLoss;

            var args = new AveragedPerceptronTrainer.Arguments()
            {
                LearningRate = learningRate,
                DecreaseLearningRate = decreaseLearningRate,
                L2RegularizerWeight = l2RegularizerWeight,
                NumIterations = numIterations
            };

            if (lossFunction != null)
                args.LossFunction = new TrivialClassificationLossFactory(lossFunction);

            var rec = new TrainerEstimatorReconciler.BinaryClassifierNoCalibration(
                (env, labelName, featuresName, weightsName) =>
                {
                    args.FeatureColumn = featuresName;
                    args.LabelColumn = labelName;
                    args.InitialWeights = weightsName;

                    var trainer = new AveragedPerceptronTrainer(env, args);

                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                    else
                        return trainer;

                    /*
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
                    */

                }, label, features, weights, hasProbs);

            return rec.Output;
        }

        /// <summary>
        /// Predict a target using a linear binary classification model trained with the AveragePerceptron trainer, and a custom loss.
        /// Note that because we cannot be sure that all loss functions will produce naturally calibrated outputs, setting
        /// a custom loss function will not produce a calibrated probability column.
        /// </summary>
        /// <param name="ctx">The binary classification context trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="lossFunction">The custom loss.</param>
        /// <param name="learningRate">The learning Rate.</param>
        /// <param name="decreaseLearningRate">Decrease learning rate as iterations progress.</param>
        /// <param name="l2RegularizerWeight">L2 Regularization Weight.</param>
        /// <param name="numIterations">Number of training iterations through the data.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TTupleInShape, TTupleOutShape, TTransformer}.Fit(DataView{TTupleInShape})"/> method is called on the
        /// <see cref="Estimator{TTupleInShape, TTupleOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained, as well as the calibrator on top of that model. Note that this action cannot change the
        /// result in any way; it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The set of output columns including in order the predicted binary classification score (which will range
        /// from negative to positive infinity), and the predicted label.</returns>
        /// <seealso cref="AveragedPerceptronTrainer"/>.
        /// <returns>The set of output columns including in order the predicted per-class likelihoods (between 0 and 1, and summing up to 1), and the predicted label.</returns>
        public static (Scalar<float> score, Scalar<float> probability, Scalar<bool> predictedLabel)
            AveragedPerceptron(this BinaryClassificationContext.BinaryClassificationTrainers ctx,
                Scalar<bool> label,
                Vector<float> features,
                IClassificationLoss lossFunction = null,
                Scalar<float> weights = null,
                float learningRate = AveragedLinearArguments.DefaultAveragedArgs.LearningRate,
                bool decreaseLearningRate = AveragedLinearArguments.DefaultAveragedArgs.DecreaseLearningRate,
                float l2RegularizerWeight = AveragedLinearArguments.DefaultAveragedArgs.L2RegularizerWeight,
                int numIterations = OnlineLinearArguments.DefaultArgs.NumIterations,
                Action<LinearBinaryPredictor> onFit = null)
        {
            Contracts.CheckValue(label, nameof(label));
            Contracts.CheckValue(features, nameof(features));
            Contracts.CheckValueOrNull(weights);
            Contracts.CheckParam(learningRate > 0, nameof(learningRate), "Must be positive.");
            Contracts.CheckParam(0 <= l2RegularizerWeight && l2RegularizerWeight < 0.5, nameof(l2RegularizerWeight), "must be in range [0, 0.5)");

            Contracts.CheckParam(numIterations > 1, nameof(numIterations), "Must be greater than one, if specified.");
            Contracts.CheckValueOrNull(onFit);

            var args = new AveragedPerceptronTrainer.Arguments()
            {
                LearningRate = learningRate,
                DecreaseLearningRate = decreaseLearningRate,
                L2RegularizerWeight = l2RegularizerWeight,
                NumIterations = numIterations
            };

            if (lossFunction != null)
                args.LossFunction = new TrivialClassificationLossFactory(lossFunction);

            var rec = new TrainerEstimatorReconciler.BinaryClassifier(
                (env, labelName, featuresName, weightsName) =>
                {
                    args.FeatureColumn = featuresName;
                    args.LabelColumn = labelName;
                    args.InitialWeights = weightsName;

                    var trainer = new AveragedPerceptronTrainer(env, args);
                    if (onFit != null)
                    {
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                        /*
                        return trainer.WithOnFitDelegate(trans =>
                        {
                            // Under the default log-loss we assume a calibrated predictor.
                            var model = trans.Model;
                            var cali = (ParameterMixingCalibratedPredictor)model;
                            var pred = (LinearBinaryPredictor)cali.SubPredictor;
                            onFit(pred, cali);
                        });
                        */
                    }
                    return trainer;
                }, label, features, weights);

            return rec.Output;
        }

        private sealed class TrivialClassificationLossFactory : ISupportClassificationLossFactory
        {
            private readonly IClassificationLoss _loss;

            public TrivialClassificationLossFactory(IClassificationLoss loss)
            {
                _loss = loss;
            }

            public IClassificationLoss CreateComponent(IHostEnvironment env)
            {
                return _loss;
            }
        }
    }
}
