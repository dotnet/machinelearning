// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Trainers;
using System;

namespace Microsoft.ML
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
        /// <param name="advancedSettings">A delegate to supply more advanced arguments to the algorithm.</param>
        public static AveragedPerceptronTrainer AveragedPerceptron(
            this BinaryClassificationContext.BinaryClassificationTrainers ctx,
            string label = DefaultColumnNames.Label,
            string features = DefaultColumnNames.Features,
            string weights = null,
            IClassificationLoss lossFunction = null,
            float learningRate = AveragedLinearArguments.AveragedDefaultArgs.LearningRate,
            bool decreaseLearningRate = AveragedLinearArguments.AveragedDefaultArgs.DecreaseLearningRate,
            float l2RegularizerWeight = AveragedLinearArguments.AveragedDefaultArgs.L2RegularizerWeight,
            int numIterations = AveragedLinearArguments.AveragedDefaultArgs.NumIterations,
            Action<AveragedPerceptronTrainer.Arguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            var loss = new TrivialClassificationLossFactory(lossFunction ?? new LogLoss());
            return new AveragedPerceptronTrainer(env, label, features, weights, loss, learningRate, decreaseLearningRate, l2RegularizerWeight, numIterations, advancedSettings);
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
        public static OnlineGradientDescentTrainer OnlineGradientDescent(this RegressionContext.RegressionTrainers ctx,
            string label = DefaultColumnNames.Label,
            string features = DefaultColumnNames.Features,
            string weights = null,
            IRegressionLoss lossFunction = null,
            float learningRate = OnlineGradientDescentTrainer.Arguments.OgdDefaultArgs.LearningRate,
            bool decreaseLearningRate = OnlineGradientDescentTrainer.Arguments.OgdDefaultArgs.DecreaseLearningRate,
            float l2RegularizerWeight = AveragedLinearArguments.AveragedDefaultArgs.L2RegularizerWeight,
            int numIterations = OnlineLinearArguments.OnlineDefaultArgs.NumIterations,
            Action<AveragedLinearArguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new OnlineGradientDescentTrainer(env, label, features, learningRate, decreaseLearningRate, l2RegularizerWeight, numIterations, weights, lossFunction, advancedSettings);
        }
    }
}
