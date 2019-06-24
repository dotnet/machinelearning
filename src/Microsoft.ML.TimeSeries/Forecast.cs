// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Data;
using static Microsoft.ML.Transforms.TimeSeries.AdaptiveSingularSpectrumSequenceModeler;

namespace Microsoft.ML.TimeSeries
{
    /// <summary>
    /// Interface for forecasting models.
    /// </summary>
    /// <typeparam name="T">The type of values that are forecasted.</typeparam>
    public interface ICanForecast<T> : ICanSaveModel
    {
        /// <summary>
        /// Train a forecasting model from an <see cref="IDataView"/>.
        /// </summary>
        /// <param name="dataView">Training data.</param>
        void Train(IDataView dataView);

        /// <summary>
        /// Update a forecasting model with the new observations in the form of an <see cref="IDataView"/>.
        /// </summary>
        /// <param name="dataView">Reference to the observations as an <see cref="IDataView"/></param>
        /// <param name="inputColumnName">Name of the input column to update from.</param>
        void Update(IDataView dataView, string inputColumnName = null);

        /// <summary>
        /// Perform forecasting until a particular <paramref name="horizon"/>.
        /// </summary>
        /// <param name="horizon">Number of values to forecast.</param>
        /// <returns>Forecasted values.</returns>
        T[] Forecast(int horizon);

        /// <summary>
        /// Perform forecasting until a particular <paramref name="horizon"/> and also computes confidence intervals.
        /// </summary>
        /// <param name="horizon">Number of values to forecast.</param>
        /// <param name="forecast">Forecasted values</param>
        /// <param name="confidenceIntervalLowerBounds">Lower bound confidence intervals of forecasted values.</param>
        /// <param name="confidenceIntervalUpperBounds">Upper bound confidence intervals of forecasted values.</param>
        /// <param name="confidenceLevel">Forecast confidence level.</param>
        void ForecastWithConfidenceIntervals(int horizon, out T[] forecast, out float[] confidenceIntervalLowerBounds, out float[] confidenceIntervalUpperBounds, float confidenceLevel = 0.95f);
    }

    public static class ForecastExtensions
    {
        /// <summary>
        /// Load a <see cref="ICanForecast{T}"/> model.
        /// </summary>
        /// <typeparam name="T">The type of <see cref="ICanForecast{T}"/>, usually float.</typeparam>
        /// <param name="catalog"><see cref="ModelOperationsCatalog"/></param>
        /// <param name="filePath">File path to load the model from.</param>
        /// <returns><see cref="ICanForecast{T}"/> model.</returns>
        public static ICanForecast<T> LoadForecastingModel<T>(this ModelOperationsCatalog catalog, string filePath)
        {
            var env = CatalogUtils.GetEnvironment(catalog);
            using (var file = File.OpenRead(filePath))
            {
                using (var rep = RepositoryReader.Open(file, env))
                {
                    ModelLoadContext.LoadModel<ICanForecast<T>, SignatureLoadModel>(env, out var model, rep, LoaderSignature);
                    return model;
                }
            }
        }

        /// <summary>
        /// Save a <see cref="ICanForecast{T}"/> model to a file specified by <paramref name="filePath"/>
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="catalog"><see cref="ModelOperationsCatalog"/></param>
        /// <param name="model"><see cref="ICanForecast{T}"/> model to save.</param>
        /// <param name="filePath">File path to save the model to.</param>
        public static void SaveForecastingModel<T>(this ModelOperationsCatalog catalog, ICanForecast<T> model, string filePath)
        {
            var env = CatalogUtils.GetEnvironment(catalog);
            using (var file = File.Create(filePath))
            {
                using (var ch = env.Start("Saving forecasting model."))
                {
                    using (var rep = RepositoryWriter.CreateNew(file, ch))
                    {
                        ModelSaveContext.SaveModel(rep, model, LoaderSignature);
                        rep.Commit();
                    }
                }
            }
        }
    }
}
