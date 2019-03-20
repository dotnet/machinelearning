// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// The MetricsStatistics class computes summary statistics over multiple observations of a metric.
    /// </summary>
    public sealed class MetricStatistics
    {
        private readonly SummaryStatistics _statistic;

        /// <summary>
        /// Get the mean value for the metric.
        /// </summary>
        public double Mean => _statistic.Mean;

        /// <summary>
        /// Get the standard deviation for the metric.
        /// </summary>
        public double StandardDeviation => (_statistic.RawCount <= 1) ? 0 : _statistic.SampleStdDev;

        /// <summary>
        /// Get the standard error of the mean for the metric.
        /// </summary>
        public double StandardError => (_statistic.RawCount <= 1) ? 0 : _statistic.StandardErrorMean;

        /// <summary>
        /// Get the count for the number of samples used. Useful for interpreting
        /// the standard deviation and the stardard error and building confidence intervals.
        /// </summary>
        public int Count => (int)_statistic.RawCount;

        internal MetricStatistics()
        {
            _statistic = new SummaryStatistics();
        }

        /// <summary>
        /// Add another metric to the set of observations.
        /// </summary>
        /// <param name="metric">The metric being accumulated</param>
        internal void Add(double metric)
        {
            _statistic.Add(metric);
        }
    }

    internal static class MetricsStatisticsUtils
    {
        public static void AddToEach(IReadOnlyList<double> src, IReadOnlyList<MetricStatistics> dest)
        {
            Contracts.Assert(src.Count == dest.Count);

            for (int i = 0; i < dest.Count; i++)
                dest[i].Add(src[i]);
        }

        public static MetricStatistics[] InitializeArray(int length)
        {
            var array = new MetricStatistics[length];
            for (int i = 0; i < array.Length; i++)
                array[i] = new MetricStatistics();

            return array;
        }
    }

    /// <summary>
    /// This interface handles the accumulation of summary statistics over multiple observations of
    /// evaluation metrics.
    /// </summary>
    /// <typeparam name="T">The metric results type, such as <see cref="RegressionMetrics"/>.</typeparam>
    internal interface IMetricsStatistics<T>
    {
        void Add(T metrics);
    }

    /// <summary>
    /// The <see cref="RegressionMetricsStatistics"/> class holds summary
    /// statistics over multiple observations of <see cref="RegressionMetrics"/>.
    /// </summary>
    public sealed class RegressionMetricsStatistics : IMetricsStatistics<RegressionMetrics>
    {
        /// <summary>
        /// Summary statistics for <see cref="RegressionMetrics.MeanAbsoluteError"/>.
        /// </summary>
        public MetricStatistics MeanAbsoluteError { get; }

        /// <summary>
        /// Summary statistics for <see cref="RegressionMetrics.MeanSquaredError"/>.
        /// </summary>
        public MetricStatistics MeanSquaredError { get; }

        /// <summary>
        /// Summary statistics for <see cref="RegressionMetrics.RootMeanSquaredError"/>.
        /// </summary>
        public MetricStatistics RootMeanSquaredError { get; }

        /// <summary>
        /// Summary statistics for <see cref="RegressionMetrics.LossFunction"/>.
        /// </summary>
        public MetricStatistics LossFunction { get; }

        /// <summary>
        /// Summary statistics for <see cref="RegressionMetrics.RSquared"/>.
        /// </summary>
        public MetricStatistics RSquared { get; }

        internal RegressionMetricsStatistics()
        {
            MeanAbsoluteError = new MetricStatistics();
            MeanSquaredError = new MetricStatistics();
            RootMeanSquaredError = new MetricStatistics();
            LossFunction = new MetricStatistics();
            RSquared = new MetricStatistics();
        }

        /// <summary>
        /// Add a set of evaluation metrics to the set of observations.
        /// </summary>
        /// <param name="metrics">The observed regression evaluation metric</param>
        void IMetricsStatistics<RegressionMetrics>.Add(RegressionMetrics metrics)
        {
            MeanAbsoluteError.Add(metrics.MeanAbsoluteError);
            MeanSquaredError.Add(metrics.MeanSquaredError);
            RootMeanSquaredError.Add(metrics.RootMeanSquaredError);
            LossFunction.Add(metrics.LossFunction);
            RSquared.Add(metrics.RSquared);
        }
    }

    /// <summary>
    /// The <see cref="BinaryClassificationMetricsStatistics"/> class holds summary
    /// statistics over multiple observations of <see cref="BinaryClassificationMetrics"/>.
    /// </summary>
    public sealed class BinaryClassificationMetricsStatistics : IMetricsStatistics<BinaryClassificationMetrics>
    {
        /// <summary>
        /// Summary Statistics for <see cref="BinaryClassificationMetrics.AreaUnderRocCurve"/>.
        /// </summary>
        public MetricStatistics AreaUnderRocCurve { get; }

        /// <summary>
        /// Summary Statistics for <see cref="BinaryClassificationMetrics.Accuracy"/>.
        /// </summary>
        public MetricStatistics Accuracy { get; }

        /// <summary>
        /// Summary statistics for <see cref="BinaryClassificationMetrics.PositivePrecision"/>.
        /// </summary>
        public MetricStatistics PositivePrecision { get; }

        /// <summary>
        /// Summary statistics for <see cref="BinaryClassificationMetrics.PositiveRecall"/>.
        /// </summary>
        public MetricStatistics PositiveRecall { get; }

        /// <summary>
        /// Summary statistics for <see cref="BinaryClassificationMetrics.NegativePrecision"/>.
        /// </summary>
        public MetricStatistics NegativePrecision { get; }

        /// <summary>
        /// Summary statistics for <see cref="BinaryClassificationMetrics.NegativeRecall"/>.
        /// </summary>
        public MetricStatistics NegativeRecall { get; }

        /// <summary>
        /// Summary statistics for <see cref="BinaryClassificationMetrics.F1Score"/>.
        /// </summary>
        public MetricStatistics F1Score { get; }

        /// <summary>
        /// Summary statistics for <see cref="BinaryClassificationMetrics.AreaUnderPrecisionRecallCurve"/>.
        /// </summary>
        public MetricStatistics AreaUnderPrecisionRecallCurve { get; }

        internal BinaryClassificationMetricsStatistics()
        {
            AreaUnderRocCurve = new MetricStatistics();
            Accuracy = new MetricStatistics();
            PositivePrecision = new MetricStatistics();
            PositiveRecall = new MetricStatistics();
            NegativePrecision = new MetricStatistics();
            NegativeRecall = new MetricStatistics();
            F1Score = new MetricStatistics();
            AreaUnderPrecisionRecallCurve = new MetricStatistics();
        }

        /// <summary>
        /// Add a set of evaluation metrics to the set of observations.
        /// </summary>
        /// <param name="metrics">The observed binary classification evaluation metric</param>
        void IMetricsStatistics<BinaryClassificationMetrics>.Add(BinaryClassificationMetrics metrics)
        {
            AreaUnderRocCurve.Add(metrics.AreaUnderRocCurve);
            Accuracy.Add(metrics.Accuracy);
            PositivePrecision.Add(metrics.PositivePrecision);
            PositiveRecall.Add(metrics.PositiveRecall);
            NegativePrecision.Add(metrics.NegativePrecision);
            NegativeRecall.Add(metrics.NegativeRecall);
            F1Score.Add(metrics.F1Score);
            AreaUnderPrecisionRecallCurve.Add(metrics.AreaUnderPrecisionRecallCurve);
        }
    }

    /// <summary>
    /// The <see cref="MulticlassClassificationMetricsStatistics"/> class holds summary
    /// statistics over multiple observations of <see cref="MulticlassClassificationMetrics"/>.
    /// </summary>
    public sealed class MulticlassClassificationMetricsStatistics : IMetricsStatistics<MulticlassClassificationMetrics>
    {
        /// <summary>
        /// Summary statistics for <see cref="MulticlassClassificationMetrics.MacroAccuracy"/>.
        /// </summary>
        public MetricStatistics MacroAccuracy { get; }

        /// <summary>
        /// Summary statistics for <see cref="MulticlassClassificationMetrics.MicroAccuracy"/>.
        /// </summary>
        public MetricStatistics MicroAccuracy { get; }

        /// <summary>
        /// Summary statistics for <see cref="MulticlassClassificationMetrics.LogLoss"/>.
        /// </summary>
        public MetricStatistics LogLoss { get; }

        /// <summary>
        /// Summary statistics for <see cref="MulticlassClassificationMetrics.LogLossReduction"/>.
        /// </summary>
        public MetricStatistics LogLossReduction { get; }

        /// <summary>
        /// Summary statistics for <see cref="MulticlassClassificationMetrics.TopKAccuracy"/>.
        /// </summary>
        public MetricStatistics TopKAccuracy { get; }

        /// <summary>
        /// Summary statistics for <see cref="MulticlassClassificationMetrics.PerClassLogLoss"/>.
        /// </summary>
        public IReadOnlyList<MetricStatistics> PerClassLogLoss { get; private set; }

        internal MulticlassClassificationMetricsStatistics()
        {
            MacroAccuracy = new MetricStatistics();
            MicroAccuracy = new MetricStatistics();
            LogLoss = new MetricStatistics();
            LogLossReduction = new MetricStatistics();
            TopKAccuracy = new MetricStatistics();
        }

        /// <summary>
        /// Add a set of evaluation metrics to the set of observations.
        /// </summary>
        /// <param name="metrics">The observed binary classification evaluation metric</param>
        void IMetricsStatistics<MulticlassClassificationMetrics>.Add(MulticlassClassificationMetrics metrics)
        {
            MacroAccuracy.Add(metrics.MacroAccuracy);
            MicroAccuracy.Add(metrics.MicroAccuracy);
            LogLoss.Add(metrics.LogLoss);
            LogLossReduction.Add(metrics.LogLossReduction);
            TopKAccuracy.Add(metrics.TopKAccuracy);

            if (PerClassLogLoss == null)
                PerClassLogLoss = MetricsStatisticsUtils.InitializeArray(metrics.PerClassLogLoss.Count);
            MetricsStatisticsUtils.AddToEach(metrics.PerClassLogLoss, PerClassLogLoss);
        }
    }

    /// <summary>
    /// The <see cref="RankingMetricsStatistics"/> class holds summary
    /// statistics over multiple observations of <see cref="RankingMetrics"/>.
    /// </summary>
    public sealed class RankingMetricsStatistics : IMetricsStatistics<RankingMetrics>
    {
        /// <summary>
        /// Summary statistics for <see cref="RankingMetrics.DiscountedCumulativeGains"/>.
        /// </summary>
        public IReadOnlyList<MetricStatistics> DiscountedCumulativeGains { get; private set; }

        /// <summary>
        /// Summary statistics for <see cref="RankingMetrics.NormalizedDiscountedCumulativeGains"/>.
        /// </summary>
        public IReadOnlyList<MetricStatistics> NormalizedDiscountedCumulativeGains { get; private set; }

        internal RankingMetricsStatistics()
        {
        }

        /// <summary>
        /// Add a set of evaluation metrics to the set of observations.
        /// </summary>
        /// <param name="metrics">The observed regression evaluation metric</param>
        void IMetricsStatistics<RankingMetrics>.Add(RankingMetrics metrics)
        {
            if (DiscountedCumulativeGains == null)
                DiscountedCumulativeGains = MetricsStatisticsUtils.InitializeArray(metrics.DiscountedCumulativeGains.Count);

            if (NormalizedDiscountedCumulativeGains == null)
                NormalizedDiscountedCumulativeGains = MetricsStatisticsUtils.InitializeArray(metrics.NormalizedDiscountedCumulativeGains.Count);

            MetricsStatisticsUtils.AddToEach(metrics.DiscountedCumulativeGains, DiscountedCumulativeGains);
            MetricsStatisticsUtils.AddToEach(metrics.NormalizedDiscountedCumulativeGains, NormalizedDiscountedCumulativeGains);
        }
    }
}
