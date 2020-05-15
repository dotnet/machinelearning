// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.TimeSeries;
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
        /// <a href="https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables"> independent identically distributed (i.i.d.)</a>
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
        /// <param name="backAddWindowSize">The number of points to add back of training window. No more than <paramref name="windowSize"/>, usually keep default value.</param>
        /// <param name="lookaheadWindowSize">The number of pervious points used in prediction. No more than <paramref name="windowSize"/>, usually keep default value.</param>
        /// <param name="averageingWindowSize">The size of sliding window to generate a saliency map for the series. No more than <paramref name="windowSize"/>, usually keep default value.</param>
        /// <param name="judgementWindowSize">The size of sliding window to calculate the anomaly score for each data point. No more than <paramref name="windowSize"/>.</param>
        /// <param name="threshold">The threshold to determine anomaly, score larger than the threshold is considered as anomaly. Should be in (0,1)</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[DetectAnomalyBySrCnn](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/TimeSeries/DetectAnomalyBySrCnn.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static SrCnnAnomalyEstimator DetectAnomalyBySrCnn(this TransformsCatalog catalog, string outputColumnName, string inputColumnName,
            int windowSize = 64, int backAddWindowSize = 5, int lookaheadWindowSize = 5, int averageingWindowSize = 3, int judgementWindowSize = 21, double threshold = 0.3)
            => new SrCnnAnomalyEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, windowSize, backAddWindowSize, lookaheadWindowSize, averageingWindowSize, judgementWindowSize, threshold, inputColumnName);

        /// <summary>
        /// Create <see cref="SrCnnAnomalyEstimator"/>, which detects timeseries anomalies using SRCNN algorithm.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="input">...</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// The column data is a vector of <see cref="System.Double"/>. The vector contains 3 elements: alert (1 means anomaly while 0 means normal), raw score, and magnitude of spectual residual.</param>
        /// <param name="inputColumnName">Name of column to transform. The column data must be <see cref="System.Single"/>.</param>
        /// <param name="threshold">The threshold to determine anomaly, score larger than the threshold is considered as anomaly. Should be in (0,1)</param>
        /// <param name="batchSize">.Divide the input data into batches to fit SrCnn model. Must be -1 or a positive integer no less than 12. Default value is 1024.</param>
        /// <param name="sensitivity">The sensitivity of boundaries. Must be in the interval (0, 100).</param>
        /// <param name="detectMode">The detect mode of the SrCnn model.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[DetectAnomalyBySrCnn](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/TimeSeries/DetectAnomalyBySrCnnBatchPrediction.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static IDataView BatchDetectAnomalyBySrCnn(this DataOperationsCatalog catalog, IDataView input, string outputColumnName, string inputColumnName,
            double threshold = 0.3, int batchSize = 1024, double sensitivity = 99, SrCnnDetectMode detectMode = SrCnnDetectMode.AnomalyAndMargin)
            => new SrCnnBatchAnomalyDetector(CatalogUtils.GetEnvironment(catalog), input, inputColumnName, outputColumnName, threshold, batchSize, sensitivity, detectMode);

        /// <summary>
        /// Create <see cref="RootCause"/>, which localizes root causes using decision tree algorithm.
        /// </summary>
        /// <param name="catalog">The anomaly detection catalog.</param>
        /// <param name="src">Root cause's input. The data is an instance of <see cref="Microsoft.ML.TimeSeries.RootCauseLocalizationInput"/>.</param>
        /// <param name="beta">Beta is a weight parameter for user to choose. It is used when score is calculated for each root cause item. The range of beta should be in [0,1]. For a larger beta, root cause point which has a large difference between value and expected value will get a high score. On the contrary, for a small beta, root cause items which has a high relative change will get a high score.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[LocalizeRootCause](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/TimeSeries/LocalizeRootCause.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static RootCause LocalizeRootCause(this AnomalyDetectionCatalog catalog, RootCauseLocalizationInput src, double beta = 0.5)
        {
            IHostEnvironment host = CatalogUtils.GetEnvironment(catalog);

            //check the root cause input
            CheckRootCauseInput(host, src);

            //check beta
            host.CheckUserArg(beta >= 0 && beta <= 1, nameof(beta), "Must be in [0,1]");

            //find out the root cause
            RootCauseAnalyzer analyzer = new RootCauseAnalyzer(src, beta);
            RootCause dst = analyzer.Analyze();
            return dst;
        }

        private static void CheckRootCauseInput(IHostEnvironment host, RootCauseLocalizationInput src)
        {
            host.CheckUserArg(src.Slices.Count >= 1, nameof(src.Slices), "Must has more than one item");

            bool containsAnomalyTimestamp = false;
            foreach (MetricSlice slice in src.Slices)
            {
                if (slice.TimeStamp.Equals(src.AnomalyTimestamp))
                {
                    containsAnomalyTimestamp = true;
                }
            }
            host.CheckUserArg(containsAnomalyTimestamp, nameof(src.Slices), "Has no points in the given anomaly timestamp");
        }

        /// <summary>
        /// Singular Spectrum Analysis (SSA) model for univariate time-series forecasting.
        /// For the details of the model, refer to http://arxiv.org/pdf/1206.6910.pdf.
        /// </summary>
        /// <param name="catalog">Catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.
        /// The vector contains Alert, Raw Score, P-Value as first three values.</param>
        /// <param name="windowSize">The length of the window on the series for building the trajectory matrix (parameter L).</param>
        /// <param name="seriesLength">The length of series that is kept in buffer for modeling (parameter N).</param>
        /// <param name="trainSize">The length of series from the beginning used for training.</param>
        /// <param name="horizon">The number of values to forecast.</param>
        /// <param name="isAdaptive">The flag determing whether the model is adaptive.</param>
        /// <param name="discountFactor">The discount factor in [0,1] used for online updates.</param>
        /// <param name="rankSelectionMethod">The rank selection method.</param>
        /// <param name="rank">The desired rank of the subspace used for SSA projection (parameter r). This parameter should be in the range in [1, windowSize].
        /// If set to null, the rank is automatically determined based on prediction error minimization.</param>
        /// <param name="maxRank">The maximum rank considered during the rank selection process. If not provided (i.e. set to null), it is set to windowSize - 1.</param>
        /// <param name="shouldStabilize">The flag determining whether the model should be stabilized.</param>
        /// <param name="shouldMaintainInfo">The flag determining whether the meta information for the model needs to be maintained.</param>
        /// <param name="maxGrowth">The maximum growth on the exponential trend.</param>
        /// <param name="confidenceLowerBoundColumn">The name of the confidence interval lower bound column. If not specified then confidence intervals will not be calculated.</param>
        /// <param name="confidenceUpperBoundColumn">The name of the confidence interval upper bound column. If not specified then confidence intervals will not be calculated.</param>
        /// <param name="confidenceLevel">The confidence level for forecasting.</param>
        /// <param name="variableHorizon">Set this to true if horizon will change after training(at prediction time).</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[Forecasting](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/TimeSeries/Forecasting.cs)]
        /// [!code-csharp[ForecastingWithConfidenceInterval](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/TimeSeries/ForecastingWithConfidenceInterval.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static SsaForecastingEstimator ForecastBySsa(
            this ForecastingCatalog catalog, string outputColumnName, string inputColumnName, int windowSize, int seriesLength, int trainSize, int horizon,
            bool isAdaptive = false, float discountFactor = 1, RankSelectionMethod rankSelectionMethod = RankSelectionMethod.Exact, int? rank = null,
            int? maxRank = null, bool shouldStabilize = true, bool shouldMaintainInfo = false, GrowthRatio? maxGrowth = null, string confidenceLowerBoundColumn = null,
            string confidenceUpperBoundColumn = null, float confidenceLevel = 0.95f, bool variableHorizon = false) =>
            new SsaForecastingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName, windowSize, seriesLength, trainSize,
                horizon, isAdaptive, discountFactor, rankSelectionMethod, rank, maxRank, shouldStabilize, shouldMaintainInfo, maxGrowth, confidenceLowerBoundColumn,
                confidenceUpperBoundColumn, confidenceLevel, variableHorizon);
    }
}
