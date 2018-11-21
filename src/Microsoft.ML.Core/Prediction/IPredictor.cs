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

    /// <summary>
    /// Predictor that returns a probability distribution associated with a prediction result
    /// </summary>
    /// <typeparam name="TFeatures"> Type of features container (instance) on which to make predictions</typeparam>
    /// <typeparam name="TResult"> Type of prediction result</typeparam>
    /// <typeparam name="TResultDistribution"> Type of probability distribution associated  with the predicton</typeparam>
    public interface IDistributionPredictor<in TFeatures, TResult, out TResultDistribution>
        : IDistPredictorProducing<TResult, TResultDistribution>, IPredictor<TFeatures, TResult>
    {
        /// <summary>
        /// Return a probability distribution associated wtih the prediction.
        /// </summary>
        /// <param name="features">Data instance</param>
        /// <returns>Distribution associated with the prediction</returns>
        TResultDistribution PredictDistribution(TFeatures features);

        /// <summary>
        /// Return a probability distribution associated wtih the prediction, as well as the prediction.
        /// </summary>
        /// <param name="features">Data instance</param>
        /// <param name="result">Prediction</param>
        /// <returns>Distribution associated with the prediction</returns>
        TResultDistribution PredictDistribution(TFeatures features, out TResult result);
    }

    /// <summary>
    ///   Predictor that produces predictions for sets of instances at a time
    ///   for cases where this is more efficient than serial calls to Predict for each instance.
    /// </summary>
    /// <typeparam name="TFeatures"> Type of features container (instance) on which to make predictions</typeparam>
    /// <typeparam name="TFeaturesCollection"> Type of collection of instances </typeparam>
    /// <typeparam name="TResult"> Type of prediction result</typeparam>
    /// <typeparam name="TResultCollection"> Type of the collection of prediction results </typeparam>
    public interface IBulkPredictor<in TFeatures, in TFeaturesCollection, out TResult, out TResultCollection>
        : IPredictor<TFeatures, TResult>
    {
        /// <summary>
        /// Produce predictions for a set of instances
        /// </summary>
        /// <param name="featuresCollection">Collection of instances</param>
        /// <returns>Collection of predictions</returns>
        TResultCollection BulkPredict(TFeaturesCollection featuresCollection);
    }

    /// <summary>
    /// Predictor that can score sets of instances (presumably more efficiently)
    /// and returns a distribution associated with a prediction result.
    /// </summary>
    /// <typeparam name="TFeatures"> Type of features container (instance) on which to make predictions</typeparam>
    /// <typeparam name="TFeaturesCollection"> Type of collection of instances </typeparam>
    /// <typeparam name="TResult"> Type of prediction result</typeparam>
    /// <typeparam name="TResultDistribution"> Type of probability distribution associated  with the predicton</typeparam>
    /// <typeparam name="TResultCollection"> Type of the collection of prediction results </typeparam>
    /// <typeparam name="TResultDistributionCollection"> Type of the collection of distributions for prediction results </typeparam>
    public interface IBulkDistributionPredictor<in TFeatures, in TFeaturesCollection,
                        TResult, TResultCollection, out TResultDistribution, out TResultDistributionCollection>
        : IBulkPredictor<TFeatures, TFeaturesCollection, TResult, TResultCollection>,
          IDistributionPredictor<TFeatures, TResult, TResultDistribution>
    {
        /// <summary>
        /// Produce distributions over predictions for a set of instances
        /// </summary>
        /// <param name="featuresCollection">Collection of instances</param>
        /// <returns>Collection of prediction distributions</returns>
        TResultDistributionCollection BulkPredictDistribution(TFeaturesCollection featuresCollection);

        /// <summary>
        /// Produce distributions over predictions for a set of instances, along with actual prediction results
        /// </summary>
        /// <param name="featuresCollection">Collection of instances</param>
        /// <param name="resultCollection">Collection of prediction results</param>
        /// <returns>Collection of distributions associated with prediction results</returns>
        TResultDistributionCollection BulkPredictDistribution(TFeaturesCollection featuresCollection,
                                                              out TResultCollection resultCollection);
    }

#if FUTURE
    public interface IBulkPredictor<in TTestDataSet, out TResultSet> :
        IPredictor
    {
        // REVIEW: Should we also have versions where the caller supplies the "memory" to be filled in.
        TResultSet BulkPredict(TTestDataSet dataset);
    }

    public interface IBulkPredictor<in TTestDataset, out TResultSet, in TFeatures, out TResult>
        : IBulkPredictor<TTestDataset, TResultSet>, IPredictor<TFeatures, TResult>
    {
    }
#endif
}
