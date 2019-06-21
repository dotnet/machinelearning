// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.Extensions.ML
{
    /// <summary>
    /// Extension methods for <see cref="PredictionEnginePool{TextDataViewType, TPrediction}"/>.
    /// </summary>
    public static class PredictionEnginePoolExtensions
    {
        /// <summary>
        /// Run prediction pipeline on one example using a PredictionEngine from the pool.
        /// </summary>
        /// <param name="predictionEnginePool">
        /// The pool of PredictionEngine instances to get the PredictionEngine.
        /// </param>
        /// <param name="example">The example to run on.</param>
        /// <returns>The result of prediction. A new object is created for every call.</returns>
        public static TPrediction Predict<TData, TPrediction>(
            this PredictionEnginePool<TData, TPrediction> predictionEnginePool, TData example)
            where TData : class
            where TPrediction : class, new()
        {
            return predictionEnginePool.Predict(string.Empty, example);
        }

        /// <summary>
        /// Run prediction pipeline on one example using a PredictionEngine from the pool.
        /// </summary>
        /// <param name="predictionEnginePool">
        /// The pool of PredictionEngine instances to get the PredictionEngine.
        /// </param>
        /// <param name="modelName">
        /// The name of the model. Used when there are multiple models with the same input/output.
        /// </param>
        /// <param name="example">The example to run on.</param>
        /// <returns>The result of prediction. A new object is created for every call.</returns>
        public static TPrediction Predict<TData, TPrediction>(
            this PredictionEnginePool<TData, TPrediction> predictionEnginePool, string modelName, TData example)
            where TData : class
            where TPrediction : class, new()
        {
            var predictionEngine = predictionEnginePool.GetPredictionEngine(modelName);

            try
            {
                return predictionEngine.Predict(example);
            }
            finally
            {
                predictionEnginePool.ReturnPredictionEngine(modelName, predictionEngine);
            }
        }
    }
}
