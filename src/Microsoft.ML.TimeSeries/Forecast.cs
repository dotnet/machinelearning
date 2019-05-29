// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;

namespace Microsoft.ML.TimeSeries
{
    /// <summary>
    /// Interface for forecasting models.
    /// </summary>
    /// <typeparam name="T">The type of values that are forecasted.</typeparam>
    public interface ICanForecast<out T>
    {
        /// <summary>
        /// Train a forecasting model from an <see cref="IDataView"/>.
        /// </summary>
        /// <param name="dataView">Reference to the <see cref="IDataView"/></param>
        /// <param name="inputColumnName">Name of the input column to train the forecasing model.</param>
        void Train(IDataView dataView, string inputColumnName);

        /// <summary>
        /// Update a forecasting model with the new observations in the form of an <see cref="IDataView"/>.
        /// </summary>
        /// <param name="dataView">Reference to the observations as an <see cref="IDataView"/></param>
        /// <param name="inputColumnName">Name of the input column to update from.</param>
        void Update(IDataView dataView, string inputColumnName);

        /// <summary>
        /// Perform forecasting until a particular <paramref name="horizon"/>.
        /// </summary>
        /// <param name="horizon">Number of values to forecast.</param>
        /// <returns></returns>
        T[] Forecast(int horizon);

        /// <summary>
        /// Serialize the forecasting model to disk to preserve the state of forecasting model.
        /// </summary>
        /// <param name="env">Reference to <see cref="IHostEnvironment"/>, typically <see cref="MLContext"/></param>
        /// <param name="filePath">Name of the filepath to serialize the model to.</param>
        void Checkpoint(IHostEnvironment env, string filePath);

        /// <summary>
        /// Deserialize the forecasting model from disk.
        /// </summary>
        /// <param name="env">Reference to <see cref="IHostEnvironment"/>, typically <see cref="MLContext"/></param>
        /// <param name="filePath">Name of the filepath to deserialize the model from.</param>
        ICanForecast<T> LoadFrom(IHostEnvironment env, string filePath);
    }
}
