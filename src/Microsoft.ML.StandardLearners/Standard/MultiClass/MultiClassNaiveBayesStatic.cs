// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.StaticPipe.Runtime;
using Microsoft.ML.Trainers;

namespace Microsoft.ML.StaticPipe
{
    /// <summary>
    /// MultiClass Classification trainer estimators.
    /// </summary>
    public static partial class MultiClassClassificationTrainers
    {
        /// <summary>
        /// Predict a target using a linear multiclass classification model trained with the <see cref="MultiClassNaiveBayesTrainer"/> trainer.
        /// </summary>
        /// <param name="ctx">The multiclass classification context trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TTupleInShape, TTupleOutShape, TTransformer}.Fit(DataView{TTupleInShape})"/> method is called on the
        /// <see cref="Estimator{TTupleInShape, TTupleOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained. Note that this action cannot change the
        /// result in any way; it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The set of output columns including in order the predicted per-class likelihoods (between 0 and 1, and summing up to 1), and the predicted label.</returns>
        public static (Vector<float> score, Key<uint, TVal> predictedLabel)
            MultiClassNaiveBayesTrainer<TVal>(this MulticlassClassificationContext.MulticlassClassificationTrainers ctx,
            Key<uint, TVal> label,
            Vector<float> features,
            Action<MultiClassNaiveBayesPredictor> onFit = null)
        {
            Contracts.CheckValue(features, nameof(features));
            Contracts.CheckValue(label, nameof(label));
            Contracts.CheckValueOrNull(onFit);

            var rec = new TrainerEstimatorReconciler.MulticlassClassifier<TVal>(
                (env, labelName, featuresName, weightsName) =>
                {
                    var trainer = new MultiClassNaiveBayesTrainer(env, featuresName, labelName);

                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                    return trainer;
                }, label, features, null);

            return rec.Output;
        }
    }
}