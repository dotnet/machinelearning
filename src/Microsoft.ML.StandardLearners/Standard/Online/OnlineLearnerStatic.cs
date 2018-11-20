// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Trainers.Online;
using Microsoft.ML.StaticPipe.Runtime;
using System;

namespace Microsoft.ML.StaticPipe
{
    /// <summary>
    /// Binary Classification trainer estimators.
    /// </summary>
    public static class AveragedPerceptronExtensions
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
        /// <param name="l2RegularizerWeight">L2 regularization weight.</param>
        /// <param name="numIterations">Number of training iterations through the data.</param>
        /// <param name="advancedSettings">A delegate to supply more avdanced arguments to the algorithm.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained, as well as the calibrator on top of that model. Note that this action cannot change the
        /// result in any way; it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The set of output columns including in order the predicted binary classification score (which will range
        /// from negative to positive infinity), and the predicted label.</returns>
        /// <seealso cref="AveragedPerceptronTrainer"/>.
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[AveragedPerceptron](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Static/AveragedPerceptronBinaryClassification.cs)]
        /// ]]></format>
        /// </example>
        public static (Scalar<float> score, Scalar<bool> predictedLabel) AveragedPerceptron(
                this BinaryClassificationContext.BinaryClassificationTrainers ctx,
                Scalar<bool> label,
                Vector<float> features,
                Scalar<float> weights = null,
                IClassificationLoss lossFunction = null,
                float learningRate = AveragedLinearArguments.AveragedDefaultArgs.LearningRate,
                bool decreaseLearningRate = AveragedLinearArguments.AveragedDefaultArgs.DecreaseLearningRate,
                float l2RegularizerWeight = AveragedLinearArguments.AveragedDefaultArgs.L2RegularizerWeight,
                int numIterations = AveragedLinearArguments.AveragedDefaultArgs.NumIterations,
                Action<AveragedPerceptronTrainer.Arguments> advancedSettings = null,
                Action<LinearBinaryPredictor> onFit = null
            )
        {
            OnlineLinearStaticUtils.CheckUserParams(label, features, weights, learningRate, l2RegularizerWeight, numIterations, onFit, advancedSettings);

            bool hasProbs = lossFunction is LogLoss;

            var rec = new TrainerEstimatorReconciler.BinaryClassifierNoCalibration(
                (env, labelName, featuresName, weightsName) =>
                {

                    var trainer = new AveragedPerceptronTrainer(env, labelName, featuresName, weightsName, lossFunction,
                        learningRate, decreaseLearningRate, l2RegularizerWeight, numIterations, advancedSettings);

                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                    else
                        return trainer;

                }, label, features, weights, hasProbs);

            return rec.Output;
        }
    }

    /// <summary>
    /// Regression trainer estimators.
    /// </summary>
    public static class OnlineGradientDescentExtensions
    {
        /// <summary>
        /// Predict a target using a linear regression model trained with the <see cref="OnlineGradientDescentTrainer"/> trainer.
        /// </summary>
        /// <param name="ctx">The regression context trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="lossFunction">The custom loss. Defaults to <see cref="SquaredLoss"/> if not provided.</param>
        /// <param name="learningRate">The learning Rate.</param>
        /// <param name="decreaseLearningRate">Decrease learning rate as iterations progress.</param>
        /// <param name="l2RegularizerWeight">L2 regularization weight.</param>
        /// <param name="numIterations">Number of training iterations through the data.</param>
        /// <param name="advancedSettings">A delegate to supply more advanced arguments to the algorithm.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
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
            int numIterations = OnlineLinearArguments.OnlineDefaultArgs.NumIterations,
            Action<AveragedLinearArguments> advancedSettings = null,
            Action<LinearRegressionPredictor> onFit = null)
        {
            OnlineLinearStaticUtils.CheckUserParams(label, features, weights, learningRate, l2RegularizerWeight, numIterations, onFit, advancedSettings);
            Contracts.CheckValueOrNull(lossFunction);

            var rec = new TrainerEstimatorReconciler.Regression(
                (env, labelName, featuresName, weightsName) =>
                {
                    var trainer = new OnlineGradientDescentTrainer(env, labelName, featuresName, learningRate,
                        decreaseLearningRate, l2RegularizerWeight, numIterations, weightsName, lossFunction, advancedSettings);

                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));

                    return trainer;
                }, label, features, weights);

            return rec.Score;
        }
    }

    internal static class OnlineLinearStaticUtils{

        internal static void CheckUserParams(PipelineColumn label,
            PipelineColumn features,
            PipelineColumn weights,
            float learningRate,
            float l2RegularizerWeight,
            int numIterations,
            Delegate onFit,
            Delegate advancedArguments)
        {
            Contracts.CheckValue(label, nameof(label));
            Contracts.CheckValue(features, nameof(features));
            Contracts.CheckValueOrNull(weights);
            Contracts.CheckParam(learningRate > 0, nameof(learningRate), "Must be positive.");
            Contracts.CheckParam(0 <= l2RegularizerWeight && l2RegularizerWeight < 0.5, nameof(l2RegularizerWeight), "must be in range [0, 0.5)");
            Contracts.CheckParam(numIterations > 0, nameof(numIterations), "Must be positive, if specified.");
            Contracts.CheckValueOrNull(onFit);
            Contracts.CheckValueOrNull(advancedArguments);
        }
    }
}
