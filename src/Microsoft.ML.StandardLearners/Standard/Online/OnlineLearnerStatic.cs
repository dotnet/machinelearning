// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.StaticPipe.Runtime;
using System;

namespace Microsoft.ML.StaticPipe
{
    /// <summary>
    /// Binary Classification trainer estimators.
    /// </summary>
    public static partial class BinaryClassificationTrainers
    {
        /// <summary>
        /// Predict a target using a linear binary classification model trained with the AveragedPerceptron trainer, and a custom loss.
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
                float learningRate = AveragedLinearArguments.AveragedDefaultArgs.LearningRate,
                bool decreaseLearningRate = AveragedLinearArguments.AveragedDefaultArgs.DecreaseLearningRate,
                float l2RegularizerWeight = AveragedLinearArguments.AveragedDefaultArgs.L2RegularizerWeight,
                int numIterations = AveragedLinearArguments.AveragedDefaultArgs.NumIterations,
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

                }, label, features, weights, hasProbs);

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

    /// <summary>
    /// Regression trainer estimators.
    /// </summary>
    public static partial class RegressionTrainers
    {
        /// <summary>
        /// Predict a target using a linear regression model trained with the <see cref="Microsoft.ML.Runtime.Learners.OnlineGradientDescentTrainer"/> trainer.
        /// </summary>
        /// <param name="ctx">The regression context trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="lossFunction">The custom loss. Defaults to <see cref="SquaredLoss"/> if not provided.</param>
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
        /// <seealso cref="OnlineGradientDescentTrainer"/>.
        /// <returns>The predicted output.</returns>
        public static Scalar<float> OnlineGradientDescent(this RegressionContext.RegressionTrainers ctx,
            Scalar<float> label,
            Vector<float> features,
            Scalar<float> weights = null,
            IRegressionLoss lossFunction = null,
            float learningRate = OnlineGradientDescentTrainer.Arguments.OgdDefaultArgs.LearningRate,
            bool decreaseLearningRate = OnlineGradientDescentTrainer.Arguments.OgdDefaultArgs.DecreaseLearningRate,
            float l2RegularizerWeight = OnlineGradientDescentTrainer.Arguments.OgdDefaultArgs.L2RegularizerWeight,
            int numIterations = OnlineGradientDescentTrainer.Arguments.OgdDefaultArgs.NumIterations,
            Action<LinearRegressionPredictor> onFit = null)
        {
            var rec = new TrainerEstimatorReconciler.Regression(
                (env, labelName, featuresName, weightsName) =>
                {
                    var trainer = new OnlineGradientDescentTrainer(env, labelName, featuresName, learningRate,
                        decreaseLearningRate, l2RegularizerWeight, numIterations, weightsName, lossFunction);

                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));

                    return trainer;
                }, label, features, weights);

            return rec.Score;
        }
    }
}
