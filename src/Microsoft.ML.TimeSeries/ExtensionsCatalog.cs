﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.TimeSeriesProcessing;

namespace Microsoft.ML
{
    public static class TimeSeriesCatalog
    {
        /// <summary>
        /// Create a new instance of <see cref="IidChangePointEstimator"/>
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// Column is a vector of type double and size 4. The vector contains Alert, Raw Score, P-Value and Martingale score as first four values.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="confidence">The confidence for change point detection in the range [0, 100].</param>
        /// <param name="changeHistoryLength">The length of the sliding window on p-values for computing the martingale score.</param>
        /// <param name="martingale">The martingale used for scoring.</param>
        /// <param name="eps">The epsilon parameter for the Power martingale.</param>
        public static IidChangePointEstimator IidChangePointEstimator(this TransformsCatalog catalog, string outputColumnName, string inputColumnName,
            int confidence, int changeHistoryLength, MartingaleType martingale = MartingaleType.Power, double eps = 0.1)
            => new IidChangePointEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, confidence, changeHistoryLength, inputColumnName, martingale, eps);

        /// <summary>
        /// Create a new instance of <see cref="IidSpikeEstimator"/>
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/></param>.
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="confidence">The confidence for spike detection in the range [0, 100].</param>
        /// <param name="pvalueHistoryLength">The size of the sliding window for computing the p-value.</param>
        /// <param name="side">The argument that determines whether to detect positive or negative anomalies, or both.</param>
        public static IidSpikeEstimator IidSpikeEstimator(this TransformsCatalog catalog, string outputColumnName, string inputColumnName,
             int confidence, int pvalueHistoryLength, AnomalySide side = AnomalySide.TwoSided)
            => new IidSpikeEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, confidence, pvalueHistoryLength, inputColumnName, side);

        /// <summary>
        /// Create a new instance of <see cref="SsaChangePointEstimator"/>
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// Column is a vector of type double and size 4. The vector contains Alert, Raw Score, P-Value and Martingale score as first four values.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="confidence">The confidence for change point detection in the range [0, 100].</param>
        /// <param name="trainingWindowSize">The number of points from the beginning of the sequence used for training.</param>
        /// <param name="changeHistoryLength">The size of the sliding window for computing the p-value.</param>
        /// <param name="seasonalityWindowSize">An upper bound on the largest relevant seasonality in the input time-series.</param>
        /// <param name="errorFunction">The function used to compute the error between the expected and the observed value.</param>
        /// <param name="martingale">The martingale used for scoring.</param>
        /// <param name="eps">The epsilon parameter for the Power martingale.</param>
        public static SsaChangePointEstimator SsaChangePointEstimator(this TransformsCatalog catalog, string outputColumnName, string inputColumnName,
            int confidence, int changeHistoryLength, int trainingWindowSize, int seasonalityWindowSize, ErrorFunction errorFunction = ErrorFunction.SignedDifference,
            MartingaleType martingale = MartingaleType.Power, double eps = 0.1)
            => new SsaChangePointEstimator(CatalogUtils.GetEnvironment(catalog), new SsaChangePointDetector.Options
            {
                Name = outputColumnName,
                Source = inputColumnName ?? outputColumnName,
                Confidence = confidence,
                ChangeHistoryLength = changeHistoryLength,
                TrainingWindowSize = trainingWindowSize,
                SeasonalWindowSize = seasonalityWindowSize,
                Martingale = martingale,
                PowerMartingaleEpsilon = eps,
                ErrorFunction = errorFunction
            });

        /// <summary>
        /// Create a new instance of <see cref="SsaSpikeEstimator"/>
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.
        /// <param name="confidence">The confidence for spike detection in the range [0, 100].</param>
        /// <param name="pvalueHistoryLength">The size of the sliding window for computing the p-value.</param>
        /// <param name="trainingWindowSize">The number of points from the beginning of the sequence used for training.</param>
        /// <param name="seasonalityWindowSize">An upper bound on the largest relevant seasonality in the input time-series.</param>
        /// The vector contains Alert, Raw Score, P-Value as first three values.</param>
        /// <param name="side">The argument that determines whether to detect positive or negative anomalies, or both.</param>
        /// <param name="errorFunction">The function used to compute the error between the expected and the observed value.</param>
        public static SsaSpikeEstimator SsaSpikeEstimator(this TransformsCatalog catalog, string outputColumnName, string inputColumnName, int confidence, int pvalueHistoryLength,
            int trainingWindowSize, int seasonalityWindowSize, AnomalySide side = AnomalySide.TwoSided, ErrorFunction errorFunction = ErrorFunction.SignedDifference)
            => new SsaSpikeEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, confidence, pvalueHistoryLength, trainingWindowSize, seasonalityWindowSize, inputColumnName, side, errorFunction);
    }
}
