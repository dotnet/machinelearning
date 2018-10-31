// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Trainers;
using System;

namespace Microsoft.ML
{
    using Arguments = StochasticGradientDescentClassificationTrainer.Arguments;

    /// <summary>
    /// Binary Classification trainer estimators.
    /// </summary>
    public static class StochasticGradientDescentCatalog
    {
        /// <summary>
        ///  Predict a target using a linear binary classification model trained with the <see cref="StochasticGradientDescentClassificationTrainer"/> trainer.
        /// </summary>
        /// <param name="ctx">The binary classificaiton context trainer object.</param>
        /// <param name="label">The name of the label column.</param>
        /// <param name="features">The name of the feature column.</param>
        /// <param name="weights">The name for the example weight column.</param>
        /// <param name="maxIterations">The maximum number of iterations; set to 1 to simulate online learning.</param>
        /// <param name="initLearningRate">The initial learning rate used by SGD.</param>
        /// <param name="l2Weight">The L2 regularization constant.</param>
        /// <param name="loss">The loss function to use.</param>
        /// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        public static StochasticGradientDescentClassificationTrainer StochasticGradientDescent(this BinaryClassificationContext.BinaryClassificationTrainers ctx,
            string label = DefaultColumnNames.Label,
            string features = DefaultColumnNames.Features,
            string weights = null,
            int maxIterations = Arguments.Defaults.MaxIterations,
            double initLearningRate = Arguments.Defaults.InitLearningRate,
            float l2Weight = Arguments.Defaults.L2Weight,
            ISupportClassificationLossFactory loss = null,
            Action<Arguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new StochasticGradientDescentClassificationTrainer(env, features, label, weights, maxIterations, initLearningRate, l2Weight, loss, advancedSettings);
        }
    }
}
