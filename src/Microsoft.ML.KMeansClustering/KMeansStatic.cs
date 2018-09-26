// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.KMeans;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.StaticPipe.Runtime;
using System;

namespace Microsoft.ML.Trainers
{
    public static class KMeansStatic
    {
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
