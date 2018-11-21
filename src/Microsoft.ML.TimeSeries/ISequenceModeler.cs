// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

namespace Microsoft.ML.Runtime.TimeSeriesProcessing
{
    /// <summary>
    /// The base container class for the forecast result on a sequence of type <typeparamref name="T"/>.
    /// </summary>
    /// <typeparam name="T">The type of the elements in the sequence</typeparam>
    public abstract class ForecastResultBase<T>
    {
        public VBuffer<T> PointForecast;
    }

    /// <summary>
    /// The standard interface for modeling a sequence.
    /// </summary>
    /// <typeparam name="TInput">The type of the elements in the input sequence</typeparam>
    /// <typeparam name="TOutput">The type of the elements in the output sequence</typeparam>
    public interface ISequenceModeler<TInput, TOutput> : ICanSaveModel
    {
        /// <summary>
        /// Initializes the state of the modeler
        /// </summary>
        void InitState();

        /// <summary>
        /// Consumes one element from the input sequence.
        /// </summary>
        /// <param name="input">An element in the sequence</param>
        /// <param name="updateModel">determines whether the sequence model should be updated according to the input</param>
        void Consume(ref TInput input, bool updateModel = false);

        /// <summary>
        /// Trains the sequence model on a given sequence.
        /// </summary>
        /// <param name="data">The input sequence used for training</param>
        void Train(FixedSizeQueue<TInput> data);

        /// <summary>
        /// Trains the sequence model on a given sequence. The method accepts an object of RoleMappedData,
        /// and assumes the input column is the 'Feature' column of type TInput.
        /// </summary>
        /// <param name="data">The input sequence used for training</param>
        void Train(RoleMappedData data);

        /// <summary>
        /// Forecasts the next 'horizon' elements in the output sequence.
        /// </summary>
        /// <param name="result">The forecast result for the given horizon along with optional information depending on the algorithm</param>
        /// <param name="horizon">The forecast horizon</param>
        void Forecast(ref ForecastResultBase<TOutput> result, int horizon = 1);

        /// <summary>
        /// Predicts the next element in the output sequence.
        /// </summary>
        /// <param name="output">The output ref parameter the will contain the prediction result</param>
        void PredictNext(ref TOutput output);

        /// <summary>
        /// Creates a clone of the model.
        /// </summary>
        /// <returns>A clone of the object</returns>
        ISequenceModeler<TInput, TOutput> Clone();
    }
}
