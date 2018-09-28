// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.KMeans;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.StaticPipe.Runtime;
using System;

namespace Microsoft.ML.Trainers
{
    /// <summary>
    /// The trainer context extensions for the <see cref="KMeansPlusPlusTrainer"/>.
    /// </summary>
    public static class ClusteringTrainers
    {
        /// <summary>
        /// KMeans <see cref="ClusteringContext"/> extension method.
        /// </summary>
        /// <param name="ctx">The regression context trainer object.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="clustersCount">The number of clusters to use for KMeans.</param>
        /// <param name="advancedSettings">Algorithm advanced settings.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TTupleInShape, TTupleOutShape, TTransformer}.Fit(DataView{TTupleInShape})"/> method is called on the
        /// <see cref="Estimator{TTupleInShape, TTupleOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained.  Note that this action cannot change the result in any way; it is only a way for the caller to
        /// be informed about what was learnt.</param>
        /// <returns>The predicted output.</returns>
        public static (Vector<float> score, Key<uint> predictedLabel) KMeans(this ClusteringContext.ClusteringTrainers ctx,
           Vector<float> features, Scalar<float> weights = null,
           int clustersCount = KMeansPlusPlusTrainer.Defaults.K,
           Action<KMeansPlusPlusTrainer.Arguments> advancedSettings = null,
           Action<KMeansPredictor> onFit = null)
        {
               var rec = new TrainerEstimatorReconciler.Clustering(
               (env, featuresName, weightsName) =>
               {
                   var trainer = new KMeansPlusPlusTrainer(env, featuresName, clustersCount, weightsName, advancedSettings);

                   if (onFit != null)
                       return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                   else
                       return trainer;
               }, features, weights);

            return rec.Output;
        }
    }
}
