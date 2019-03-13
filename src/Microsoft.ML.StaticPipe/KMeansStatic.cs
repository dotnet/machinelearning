// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;

namespace Microsoft.ML.StaticPipe
{
    /// <summary>
    /// The trainer context extensions for the <see cref="KMeansTrainer"/>.
    /// </summary>
    public static class KMeansClusteringExtensions
    {
        /// <summary>
        /// KMeans <see cref="ClusteringCatalog"/> extension method.
        /// </summary>
        /// <param name="catalog">The clustering catalog trainer object.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="clustersCount">The number of clusters to use for KMeans.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained.  Note that this action cannot change the result in any way; it is only a way for the caller to
        /// be informed about what was learnt.</param>
        /// <returns>The predicted output.</returns>
        public static (Vector<float> score, Key<uint> predictedLabel) KMeans(this ClusteringCatalog.ClusteringTrainers catalog,
           Vector<float> features, Scalar<float> weights = null,
           int clustersCount = KMeansTrainer.Defaults.NumberOfClusters,
           Action<KMeansModelParameters> onFit = null)
        {
            Contracts.CheckValue(features, nameof(features));
            Contracts.CheckValueOrNull(weights);
            Contracts.CheckParam(clustersCount > 1, nameof(clustersCount), "If provided, must be greater than 1.");
            Contracts.CheckValueOrNull(onFit);

            var rec = new TrainerEstimatorReconciler.Clustering(
            (env, featuresName, weightsName) =>
            {
                var options = new KMeansTrainer.Options
                {
                    FeatureColumnName = featuresName,
                    NumberOfClusters = clustersCount,
                    ExampleWeightColumnName = weightsName
                };

                var trainer = new KMeansTrainer(env, options);

                if (onFit != null)
                    return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                else
                    return trainer;
            }, features, weights);

            return rec.Output;
        }

        /// <summary>
        /// KMeans <see cref="ClusteringCatalog"/> extension method.
        /// </summary>
        /// <param name="catalog">The regression catalog trainer object.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="options">Algorithm advanced settings.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained.  Note that this action cannot change the result in any way; it is only a way for the caller to
        /// be informed about what was learnt.</param>
        /// <returns>The predicted output.</returns>
        public static (Vector<float> score, Key<uint> predictedLabel) KMeans(this ClusteringCatalog.ClusteringTrainers catalog,
           Vector<float> features, Scalar<float> weights,
           KMeansTrainer.Options options,
           Action<KMeansModelParameters> onFit = null)
        {
            Contracts.CheckValueOrNull(onFit);
            Contracts.CheckValue(options, nameof(options));

            var rec = new TrainerEstimatorReconciler.Clustering(
            (env, featuresName, weightsName) =>
            {
                options.FeatureColumnName = featuresName;
                options.ExampleWeightColumnName = weightsName;

                var trainer = new KMeansTrainer(env, options);

                if (onFit != null)
                    return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                else
                    return trainer;
            }, features, weights);

            return rec.Output;
        }
    }
}
