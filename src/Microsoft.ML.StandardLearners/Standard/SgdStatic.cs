// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Trainers;
using Microsoft.ML.StaticPipe.Runtime;
using System;

namespace Microsoft.ML.StaticPipe
{
    using Arguments = StochasticGradientDescentClassificationTrainer.Arguments;

    /// <summary>
    /// Binary Classification trainer estimators.
    /// </summary>
    public static class SgdExtensions
    {
        /// <summary>
        ///  Predict a target using a linear binary classification model trained with the <see cref="Microsoft.ML.Trainers.StochasticGradientDescentClassificationTrainer"/> trainer.
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
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TTupleInShape, TTupleOutShape, TTransformer}.Fit(DataView{TTupleInShape})"/> method is called on the
        /// <see cref="Estimator{TTupleInShape, TTupleOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained.  Note that this action cannot change the result in any way; it is only a way for the caller to
        /// be informed about what was learnt.</param>
        /// <returns>The predicted output.</returns>
        public static (Scalar<float> score, Scalar<float> probability, Scalar<bool> predictedLabel) StochasticGradientDescentClassificationTrainer(this BinaryClassificationContext.BinaryClassificationTrainers ctx,
            Scalar<bool> label,
            Vector<float> features,
            Scalar<float> weights = null,
            int maxIterations = Arguments.Defaults.MaxIterations,
            double initLearningRate = Arguments.Defaults.InitLearningRate,
            float l2Weight = Arguments.Defaults.L2Weight,
            ISupportClassificationLossFactory loss = null,
            Action<Arguments> advancedSettings = null,
            Action<IPredictorWithFeatureWeights<float>> onFit = null)
        {
            var rec = new TrainerEstimatorReconciler.BinaryClassifier(
                (env, labelName, featuresName, weightsName) =>
                {
                    var trainer = new StochasticGradientDescentClassificationTrainer(env, labelName, featuresName, weightsName, maxIterations, initLearningRate, l2Weight, loss, advancedSettings);

                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                    return trainer;

                }, label, features, weights);

            return rec.Output;
        }
    }
}
