// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;

namespace Microsoft.ML
{
    public static class TimeSeriesCatalog
    {
        /// <summary>
        /// Create <see cref="IidChangePointEstimator"/>, which predicts change points in an
        /// <a href="https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables">independent identically distributed (i.i.d.)</a>
        /// time series based on adaptive kernel density estimations and martingale scores.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// The column data is a vector of <see cref="System.Double"/>. The vector contains 4 elements: alert (non-zero value means a change point), raw score, p-Value and martingale score.</param>
        /// <param name="inputColumnName">Name of column to transform. The column data must be <see cref="System.Single"/>. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="confidence">The confidence for change point detection in the range [0, 100].</param>
        /// <param name="changeHistoryLength">The length of the sliding window on p-values for computing the martingale score.</param>
        /// <param name="martingale">The martingale used for scoring.</param>
        /// <param name="eps">The epsilon parameter for the Power martingale.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[DetectIidChangePoint](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/TimeSeries/DetectIidChangePointBatchPrediction.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static IidChangePointEstimator DetectIidChangePoint(this TransformsCatalog catalog, string outputColumnName, string inputColumnName,
            int confidence, int changeHistoryLength, MartingaleType martingale = MartingaleType.Power, double eps = 0.1)
            => new IidChangePointEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, confidence, changeHistoryLength, inputColumnName, martingale, eps);

        /// <summary>
        /// Create <see cref="IidSpikeEstimator"/>, which predicts spikes in
        /// <a href="https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables">independent identically distributed (i.i.d.)</a>
        /// time series based on adaptive kernel density estimations and martingale scores.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// The column data is a vector of <see cref="System.Double"/>. The vector contains 3 elements: alert (non-zero value means a spike), raw score, and p-value.</param>
        /// <param name="inputColumnName">Name of column to transform. The column data must be <see cref="System.Single"/>.
        /// If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="confidence">The confidence for spike detection in the range [0, 100].</param>
        /// <param name="pvalueHistoryLength">The size of the sliding window for computing the p-value.</param>
        /// <param name="side">The argument that determines whether to detect positive or negative anomalies, or both.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[DetectIidSpike](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/TimeSeries/DetectIidSpikeBatchPrediction.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static IidSpikeEstimator DetectIidSpike(this TransformsCatalog catalog, string outputColumnName, string inputColumnName,
             int confidence, int pvalueHistoryLength, AnomalySide side = AnomalySide.TwoSided)
            => new IidSpikeEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, confidence, pvalueHistoryLength, inputColumnName, side);

        /// <summary>
        /// Create <see cref="SsaChangePointEstimator"/>, which predicts change points in time series
        /// using <a href="https://en.wikipedia.org/wiki/Singular_spectrum_analysis">Singular Spectrum Analysis (SSA)</a>.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// The column data is a vector of <see cref="System.Double"/>. The vector contains 4 elements: alert (non-zero value means a change point), raw score, p-Value and martingale score.</param>
        /// <param name="inputColumnName">Name of column to transform. The column data must be <see cref="System.Single"/>.
        /// If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="confidence">The confidence for change point detection in the range [0, 100].</param>
        /// <param name="trainingWindowSize">The number of points from the beginning of the sequence used for training.</param>
        /// <param name="changeHistoryLength">The size of the sliding window for computing the p-value.</param>
        /// <param name="seasonalityWindowSize">An upper bound on the largest relevant seasonality in the input time-series.</param>
        /// <param name="errorFunction">The function used to compute the error between the expected and the observed value.</param>
        /// <param name="martingale">The martingale used for scoring.</param>
        /// <param name="eps">The epsilon parameter for the Power martingale.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[DetectChangePointBySsa](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/TimeSeries/DetectChangePointBySsaBatchPrediction.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static SsaChangePointEstimator DetectChangePointBySsa(this TransformsCatalog catalog, string outputColumnName, string inputColumnName,
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
        /// Create <see cref="SsaSpikeEstimator"/>, which predicts spikes in time series
        /// using <a href="https://en.wikipedia.org/wiki/Singular_spectrum_analysis">Singular Spectrum Analysis (SSA)</a>.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// The column data is a vector of <see cref="System.Double"/>. The vector contains 3 elements: alert (non-zero value means a spike), raw score, and p-value.</param>
        /// <param name="inputColumnName">Name of column to transform. The column data must be <see cref="System.Single"/>.
        /// If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="confidence">The confidence for spike detection in the range [0, 100].</param>
        /// <param name="pvalueHistoryLength">The size of the sliding window for computing the p-value.</param>
        /// <param name="trainingWindowSize">The number of points from the beginning of the sequence used for training.</param>
        /// <param name="seasonalityWindowSize">An upper bound on the largest relevant seasonality in the input time-series.</param>
        /// <param name="side">The argument that determines whether to detect positive or negative anomalies, or both.</param>
        /// <param name="errorFunction">The function used to compute the error between the expected and the observed value.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[DetectSpikeBySsa](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/TimeSeries/DetectSpikeBySsaBatchPrediction.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static SsaSpikeEstimator DetectSpikeBySsa(this TransformsCatalog catalog, string outputColumnName, string inputColumnName, int confidence, int pvalueHistoryLength,
            int trainingWindowSize, int seasonalityWindowSize, AnomalySide side = AnomalySide.TwoSided, ErrorFunction errorFunction = ErrorFunction.SignedDifference)
            => new SsaSpikeEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, confidence, pvalueHistoryLength, trainingWindowSize, seasonalityWindowSize, inputColumnName, side, errorFunction);

        /// <summary>
        /// Create <see cref="SrCnnAnomalyEstimator"/>, which detects timeseries anomalies using SRCNN algorithm.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// The column data is a vector of <see cref="System.Double"/>. The vector contains 3 elements: alert (1 means anomaly while 0 means normal), raw score, and magnitude of spectual residual.</param>
        /// <param name="inputColumnName">Name of column to transform. The column data must be <see cref="System.Single"/>.</param>
        /// <param name="windowSize">The size of the sliding window for computing spectral residual.</param>
        /// <param name="backAddWindowSize">The number of points to add back of training window. No more than windowSize, usually keep default value.</param>
        /// <param name="lookaheadWindowSize">The number of pervious points used in prediction. No more than windowSize, usually keep default value.</param>
        /// <param name="averageingWindowSize">The size of sliding window to generate a saliency map for the series. No more than windowSize, usually keep default value.</param>
        /// <param name="judgementWindowSize">The size of sliding window to calculate the anomaly score for each data point. No more than windowSize.</param>
        /// <param name="threshold">The threshold to determine anomaly, score larger than the threshold is considered as anomaly. Should be in (0,1)</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[DetectAnomalyBySrCnn](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/TimeSeries/DetectAnomalyBySrCnn.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static SrCnnAnomalyEstimator DetectAnomalyBySrCnn(this TransformsCatalog catalog, string outputColumnName, string inputColumnName,
            int windowSize=64, int backAddWindowSize=5, int lookaheadWindowSize=5, int averageingWindowSize=3, int judgementWindowSize=21, double threshold=0.3)
            => new SrCnnAnomalyEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, windowSize, backAddWindowSize, lookaheadWindowSize, averageingWindowSize, judgementWindowSize, threshold, inputColumnName);
    }
}
