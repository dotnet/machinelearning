// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Runtime
{
    /// <summary>
    /// Type of prediction task
    /// </summary>
    public enum PredictionKind
    {
        Unknown = 0,
        Custom = 1,

        BinaryClassification = 2,
        MultiClassClassification = 3,
        Regression = 4,
        MultiOutputRegression = 5,
        Ranking = 6,
        Recommendation = 7,
        AnomalyDetection = 8,
        Clustering = 9,
        SequenceClassification = 10,

        // More to be added later.
    }

    /// <summary>
    /// Weakly typed version of IPredictor.
    /// </summary>
    public interface IPredictor
    {
        /// <summary>
        /// Return the type of prediction task.
        /// </summary>
        PredictionKind PredictionKind { get; }
    }

    /// <summary>
    /// A predictor the produces values of the indicated type.
    /// REVIEW: Determine whether this is just a temporary shim or long term solution.
    /// </summary>
    public interface IPredictorProducing<out TResult> : IPredictor
    {
    }

    /// <summary>
    /// Strongly typed generic predictor that takes data instances (feature containers)
    /// and produces predictions for them.
    /// </summary>
    /// <typeparam name="TFeatures"> Type of features container (instance) on which to make predictions</typeparam>
    /// <typeparam name="TResult"> Type of prediction result</typeparam>
    public interface IPredictor<in TFeatures, out TResult> : IPredictorProducing<TResult>
    {
        /// <summary>
        /// Produce a prediction for provided features
        /// </summary>
        /// <param name="features"> Data instance </param>
        /// <returns> Prediction </returns>
        TResult Predict(TFeatures features);
    }

    /// <summary>
    /// A predictor that produces values and distributions of the indicated types.
    /// REVIEW: Determine whether this is just a temporary shim or long term solution.
    /// </summary>
    public interface IDistPredictorProducing<out TResult, out TResultDistribution> : IPredictorProducing<TResult>
    {
    }
}
