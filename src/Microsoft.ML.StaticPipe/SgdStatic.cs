// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.StaticPipe.Runtime;
using Microsoft.ML.Trainers;

namespace Microsoft.ML.StaticPipe
{
    using Options = StochasticGradientDescentClassificationTrainer.Options;

    /// <summary>
    /// Binary Classification trainer estimators.
    /// </summary>
    public static class SgdStaticExtensions
    {
        /// <summary>
        ///  Predict a target using a linear binary classification model trained with the <see cref="Microsoft.ML.Trainers.StochasticGradientDescentClassificationTrainer"/> trainer.
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
        public static (Scalar<float> score, Scalar<float> probability, Scalar<bool> predictedLabel) StochasticGradientDescentClassificationTrainer(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            Scalar<bool> label,
            Vector<float> features,
            Scalar<float> weights = null,
            int maxIterations = Options.Defaults.MaxIterations,
            double initLearningRate = Options.Defaults.InitLearningRate,
            float l2Weight = Options.Defaults.L2Weight,
            ISupportClassificationLossFactory loss = null,
            Action<IPredictorWithFeatureWeights<float>> onFit = null)
        {
            var rec = new TrainerEstimatorReconciler.BinaryClassifier(
                (env, labelName, featuresName, weightsName) =>
                {
                    var trainer = new StochasticGradientDescentClassificationTrainer(env, labelName, featuresName, weightsName, maxIterations, initLearningRate, l2Weight, loss);

                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                    return trainer;

                }, label, features, weights);

            return rec.Output;
        }

        /// <summary>
        ///  Predict a target using a linear binary classification model trained with the <see cref="Microsoft.ML.Trainers.StochasticGradientDescentClassificationTrainer"/> trainer.
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
        public static (Scalar<float> score, Scalar<float> probability, Scalar<bool> predictedLabel) StochasticGradientDescentClassificationTrainer(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            Scalar<bool> label,
            Vector<float> features,
            Scalar<float> weights,
            Options options,
            Action<IPredictorWithFeatureWeights<float>> onFit = null)
        {
            var rec = new TrainerEstimatorReconciler.BinaryClassifier(
                (env, labelName, featuresName, weightsName) =>
                {
                    options.FeatureColumn = featuresName;
                    options.LabelColumn = labelName;
                    options.WeightColumn = weightsName != null ? Optional<string>.Explicit(weightsName) : Optional<string>.Implicit(DefaultColumnNames.Weight);

                    var trainer = new StochasticGradientDescentClassificationTrainer(env, options);

                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                    return trainer;

                }, label, features, weights);

            return rec.Output;
        }
    }
}
