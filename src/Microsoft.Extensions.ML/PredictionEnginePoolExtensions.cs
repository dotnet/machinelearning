// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.Extensions.ML
{
    public static class PredictionEnginePoolExtensions
    {
        public static TPrediction Predict<TData, TPrediction>(this PredictionEnginePool<TData, TPrediction> predictionEnginePool, TData dataSample) where TData : class where TPrediction : class, new()
        {
            return predictionEnginePool.Predict(string.Empty, dataSample);
        }

        public static TPrediction Predict<TData, TPrediction>(this PredictionEnginePool<TData, TPrediction> predictionEnginePool, string modelName, TData dataSample) where TData : class where TPrediction : class, new()
        {
            var predictionEngine = predictionEnginePool.GetPredictionEngine(modelName);

            try
            {
                return predictionEngine.Predict(dataSample);
            }
            finally
            {
                predictionEnginePool.ReturnPredictionEngine(modelName, predictionEngine);
            }
        }
    }
}
