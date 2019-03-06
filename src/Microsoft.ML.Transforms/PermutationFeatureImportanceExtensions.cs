// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Immutable;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    public static class PermutationFeatureImportanceExtensions
    {
        #region Regression
        /// <summary>
        /// Permutation Feature Importance (PFI) for Regression
        /// </summary>
        /// <remarks>
        /// <para>
        /// Permutation feature importance (PFI) is a technique to determine the global importance of features in a trained
        /// machine learning model. PFI is a simple yet powerful technique motivated by Breiman in his Random Forest paper, section 10
        /// (Breiman. <a href='https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf'>&quot;Random Forests.&quot;</a> Machine Learning, 2001.)
        /// The advantage of the PFI method is that it is model agnostic -- it works with any model that can be
        /// evaluated -- and it can use any dataset, not just the training set, to compute feature importance metrics.
        /// </para>
        /// <para>
        /// PFI works by taking a labeled dataset, choosing a feature, and permuting the values
        /// for that feature across all the examples, so that each example now has a random value for the feature and
        /// the original values for all other features. The evalution metric (e.g. R-squared) is then calculated
        /// for this modified dataset, and the change in the evaluation metric from the original dataset is computed.
        /// The larger the change in the evaluation metric, the more important the feature is to the model.
        /// PFI works by performing this permutation analysis across all the features of a model, one after another.
        /// </para>
        /// <para>
        /// In this implementation, PFI computes the change in all possible regression evaluation metrics for each feature, and an
        /// <code>ImmutableArray</code> of <code>RegressionMetrics</code> objects is returned. See the sample below for an
        /// example of working with these results to analyze the feature importance of a model.
        /// </para>
        /// </remarks>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[PFI](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/PermutationFeatureImportance/PFIRegressionExample.cs)]
        /// ]]>
        /// </format>
        /// </example>
        /// <param name="catalog">The regression catalog.</param>
        /// <param name="model">The model to evaluate.</param>
        /// <param name="data">The evaluation data set.</param>
        /// <param name="label">Label column name.</param>
        /// <param name="features">Feature column name.</param>
        /// <param name="useFeatureWeightFilter">Use features weight to pre-filter features.</param>
        /// <param name="topExamples">Limit the number of examples to evaluate on. null means examples (up to ~ 2 bln) from input will be used.</param>
        /// <param name="permutationCount">The number of permutations to perform.</param>
        /// <returns>Array of per-feature 'contributions' to the score.</returns>
        public static ImmutableArray<RegressionMetricsStatistics>
            PermutationFeatureImportance<TModel>(
                this RegressionCatalog catalog,
                IPredictionTransformer<TModel> model,
                IDataView data,
                string label = DefaultColumnNames.Label,
                string features = DefaultColumnNames.Features,
                bool useFeatureWeightFilter = false,
                int? topExamples = null,
                int permutationCount = 1)
        {
            return PermutationFeatureImportance<TModel, RegressionMetrics, RegressionMetricsStatistics>.GetImportanceMetricsMatrix(
                            catalog.GetEnvironment(),
                            model,
                            data,
                            () => new RegressionMetricsStatistics(),
                            idv => catalog.Evaluate(idv, label),
                            RegressionDelta,
                            features,
                            permutationCount,
                            useFeatureWeightFilter,
                            topExamples);
        }

        private static RegressionMetrics RegressionDelta(
            RegressionMetrics a, RegressionMetrics b)
        {
            return new RegressionMetrics(
                l1: a.MeanAbsoluteError - b.MeanAbsoluteError,
                l2: a.MeanSquaredError - b.MeanSquaredError,
                rms: a.RootMeanSquaredError - b.RootMeanSquaredError,
                lossFunction: a.LossFunction - b.LossFunction,
                rSquared: a.RSquared - b.RSquared);
        }
        #endregion

        #region Binary Classification
        /// <summary>
        /// Permutation Feature Importance (PFI) for Binary Classification
        /// </summary>
        /// <remarks>
        /// <para>
        /// Permutation feature importance (PFI) is a technique to determine the global importance of features in a trained
        /// machine learning model. PFI is a simple yet powerful technique motivated by Breiman in his Random Forest paper, section 10
        /// (Breiman. <a href='https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf'>&quot;Random Forests.&quot;</a> Machine Learning, 2001.)
        /// The advantage of the PFI method is that it is model agnostic -- it works with any model that can be
        /// evaluated -- and it can use any dataset, not just the training set, to compute feature importance metrics.
        /// </para>
        /// <para>
        /// PFI works by taking a labeled dataset, choosing a feature, and permuting the values
        /// for that feature across all the examples, so that each example now has a random value for the feature and
        /// the original values for all other features. The evalution metric (e.g. AUC) is then calculated
        /// for this modified dataset, and the change in the evaluation metric from the original dataset is computed.
        /// The larger the change in the evaluation metric, the more important the feature is to the model.
        /// PFI works by performing this permutation analysis across all the features of a model, one after another.
        /// </para>
        /// <para>
        /// In this implementation, PFI computes the change in all possible binary classification evaluation metrics for each feature, and an
        /// <code>ImmutableArray</code> of <code>BinaryClassificationMetrics</code> objects is returned. See the sample below for an
        /// example of working with these results to analyze the feature importance of a model.
        /// </para>
        /// </remarks>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[PFI](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/PermutationFeatureImportance/PfiBinaryClassificationExample.cs)]
        /// ]]>
        /// </format>
        /// </example>
        /// <param name="catalog">The binary classification catalog.</param>
        /// <param name="model">The model to evaluate.</param>
        /// <param name="data">The evaluation data set.</param>
        /// <param name="label">Label column name.</param>
        /// <param name="features">Feature column name.</param>
        /// <param name="useFeatureWeightFilter">Use features weight to pre-filter features.</param>
        /// <param name="topExamples">Limit the number of examples to evaluate on. null means examples (up to ~ 2 bln) from input will be used.</param>
        /// <param name="permutationCount">The number of permutations to perform.</param>
        /// <returns>Array of per-feature 'contributions' to the score.</returns>
        public static ImmutableArray<BinaryClassificationMetricsStatistics>
            PermutationFeatureImportance<TModel>(
                this BinaryClassificationCatalog catalog,
                IPredictionTransformer<TModel> model,
                IDataView data,
                string label = DefaultColumnNames.Label,
                string features = DefaultColumnNames.Features,
                bool useFeatureWeightFilter = false,
                int? topExamples = null,
                int permutationCount = 1)
        {
            return PermutationFeatureImportance<TModel, BinaryClassificationMetrics, BinaryClassificationMetricsStatistics>.GetImportanceMetricsMatrix(
                            catalog.GetEnvironment(),
                            model,
                            data,
                            () => new BinaryClassificationMetricsStatistics(),
                            idv => catalog.Evaluate(idv, label),
                            BinaryClassifierDelta,
                            features,
                            permutationCount,
                            useFeatureWeightFilter,
                            topExamples);
        }

        private static BinaryClassificationMetrics BinaryClassifierDelta(
            BinaryClassificationMetrics a, BinaryClassificationMetrics b)
        {
            return new BinaryClassificationMetrics(
                auc: a.AreaUnderRocCurve - b.AreaUnderRocCurve,
                accuracy: a.Accuracy - b.Accuracy,
                positivePrecision: a.PositivePrecision - b.PositivePrecision,
                positiveRecall: a.PositiveRecall - b.PositiveRecall,
                negativePrecision: a.NegativePrecision - b.NegativePrecision,
                negativeRecall: a.NegativeRecall - b.NegativeRecall,
                f1Score: a.F1Score - b.F1Score,
                auprc: a.AreaUnderPrecisionRecallCurve - b.AreaUnderPrecisionRecallCurve);
        }

        #endregion Binary Classification

        #region Multiclass Classification
        /// <summary>
        /// Permutation Feature Importance (PFI) for MulticlassClassification
        /// </summary>
        /// <remarks>
        /// <para>
        /// Permutation feature importance (PFI) is a technique to determine the global importance of features in a trained
        /// machine learning model. PFI is a simple yet powerful technique motivated by Breiman in his Random Forest paper, section 10
        /// (Breiman. <a href='https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf'>&quot;Random Forests.&quot;</a> Machine Learning, 2001.)
        /// The advantage of the PFI method is that it is model agnostic -- it works with any model that can be
        /// evaluated -- and it can use any dataset, not just the training set, to compute feature importance metrics.
        /// </para>
        /// <para>
        /// PFI works by taking a labeled dataset, choosing a feature, and permuting the values
        /// for that feature across all the examples, so that each example now has a random value for the feature and
        /// the original values for all other features. The evalution metric (e.g. micro-accuracy) is then calculated
        /// for this modified dataset, and the change in the evaluation metric from the original dataset is computed.
        /// The larger the change in the evaluation metric, the more important the feature is to the model.
        /// PFI works by performing this permutation analysis across all the features of a model, one after another.
        /// </para>
        /// <para>
        /// In this implementation, PFI computes the change in all possible multiclass classification evaluation metrics for each feature, and an
        /// <code>ImmutableArray</code> of <code>MultiClassClassifierMetrics</code> objects is returned. See the sample below for an
        /// example of working with these results to analyze the feature importance of a model.
        /// </para>
        /// </remarks>
        /// <param name="catalog">The clustering catalog.</param>
        /// <param name="model">The model to evaluate.</param>
        /// <param name="data">The evaluation data set.</param>
        /// <param name="label">Label column name.</param>
        /// <param name="features">Feature column name.</param>
        /// <param name="useFeatureWeightFilter">Use features weight to pre-filter features.</param>
        /// <param name="topExamples">Limit the number of examples to evaluate on. null means examples (up to ~ 2 bln) from input will be used.</param>
        /// <param name="permutationCount">The number of permutations to perform.</param>
        /// <returns>Array of per-feature 'contributions' to the score.</returns>
        public static ImmutableArray<MultiClassClassifierMetricsStatistics>
            PermutationFeatureImportance<TModel>(
                this MulticlassClassificationCatalog catalog,
                IPredictionTransformer<TModel> model,
                IDataView data,
                string label = DefaultColumnNames.Label,
                string features = DefaultColumnNames.Features,
                bool useFeatureWeightFilter = false,
                int? topExamples = null,
                int permutationCount = 1)
        {
            return PermutationFeatureImportance<TModel, MultiClassClassifierMetrics, MultiClassClassifierMetricsStatistics>.GetImportanceMetricsMatrix(
                            catalog.GetEnvironment(),
                            model,
                            data,
                            () => new MultiClassClassifierMetricsStatistics(),
                            idv => catalog.Evaluate(idv, label),
                            MulticlassClassificationDelta,
                            features,
                            permutationCount,
                            useFeatureWeightFilter,
                            topExamples);
        }

        private static MultiClassClassifierMetrics MulticlassClassificationDelta(
            MultiClassClassifierMetrics a, MultiClassClassifierMetrics b)
        {
            if (a.TopK != b.TopK)
                Contracts.Assert(a.TopK == b.TopK, "TopK to compare must be the same length.");

            var perClassLogLoss = ComputeArrayDeltas(a.PerClassLogLoss, b.PerClassLogLoss);

            return new MultiClassClassifierMetrics(
                accuracyMicro: a.MicroAccuracy - b.MicroAccuracy,
                accuracyMacro: a.MacroAccuracy - b.MacroAccuracy,
                logLoss: a.LogLoss - b.LogLoss,
                logLossReduction: a.LogLossReduction - b.LogLossReduction,
                topK: a.TopK,
                topKAccuracy: a.TopKAccuracy - b.TopKAccuracy,
                perClassLogLoss: perClassLogLoss
                );
        }

        #endregion

        #region Ranking
        /// <summary>
        /// Permutation Feature Importance (PFI) for Ranking
        /// </summary>
        /// <remarks>
        /// <para>
        /// Permutation feature importance (PFI) is a technique to determine the global importance of features in a trained
        /// machine learning model. PFI is a simple yet powerful technique motivated by Breiman in his Random Forest paper, section 10
        /// (Breiman. <a href='https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf'>&quot;Random Forests.&quot;</a> Machine Learning, 2001.)
        /// The advantage of the PFI method is that it is model agnostic -- it works with any model that can be
        /// evaluated -- and it can use any dataset, not just the training set, to compute feature importance metrics.
        /// </para>
        /// <para>
        /// PFI works by taking a labeled dataset, choosing a feature, and permuting the values
        /// for that feature across all the examples, so that each example now has a random value for the feature and
        /// the original values for all other features. The evalution metric (e.g. NDCG) is then calculated
        /// for this modified dataset, and the change in the evaluation metric from the original dataset is computed.
        /// The larger the change in the evaluation metric, the more important the feature is to the model.
        /// PFI works by performing this permutation analysis across all the features of a model, one after another.
        /// </para>
        /// <para>
        /// In this implementation, PFI computes the change in all possible ranking evaluation metrics for each feature, and an
        /// <code>ImmutableArray</code> of <code>RankingMetrics</code> objects is returned. See the sample below for an
        /// example of working with these results to analyze the feature importance of a model.
        /// </para>
        /// </remarks>
        /// <param name="catalog">The clustering catalog.</param>
        /// <param name="model">The model to evaluate.</param>
        /// <param name="data">The evaluation data set.</param>
        /// <param name="label">Label column name.</param>
        /// <param name="groupId">GroupId column name</param>
        /// <param name="features">Feature column name.</param>
        /// <param name="useFeatureWeightFilter">Use features weight to pre-filter features.</param>
        /// <param name="topExamples">Limit the number of examples to evaluate on. null means examples (up to ~ 2 bln) from input will be used.</param>
        /// <param name="permutationCount">The number of permutations to perform.</param>
        /// <returns>Array of per-feature 'contributions' to the score.</returns>
        public static ImmutableArray<RankingMetricsStatistics>
            PermutationFeatureImportance<TModel>(
                this RankingCatalog catalog,
                IPredictionTransformer<TModel> model,
                IDataView data,
                string label = DefaultColumnNames.Label,
                string groupId = DefaultColumnNames.GroupId,
                string features = DefaultColumnNames.Features,
                bool useFeatureWeightFilter = false,
                int? topExamples = null,
                int permutationCount = 1)
        {
            return PermutationFeatureImportance<TModel, RankingMetrics, RankingMetricsStatistics>.GetImportanceMetricsMatrix(
                            catalog.GetEnvironment(),
                            model,
                            data,
                            () => new RankingMetricsStatistics(),
                            idv => catalog.Evaluate(idv, label, groupId),
                            RankingDelta,
                            features,
                            permutationCount,
                            useFeatureWeightFilter,
                            topExamples);
        }

        private static RankingMetrics RankingDelta(
            RankingMetrics a, RankingMetrics b)
        {
            var dcg = ComputeArrayDeltas(a.DiscountedCumulativeGains, b.DiscountedCumulativeGains);
            var ndcg = ComputeArrayDeltas(a.NormalizedDiscountedCumulativeGains, b.NormalizedDiscountedCumulativeGains);

            return new RankingMetrics(dcg: dcg, ndcg: ndcg);
        }

        #endregion

        #region Helpers

        private static double[] ComputeArrayDeltas(double[] a, double[] b)
        {
            Contracts.Assert(a.Length == b.Length, "Arrays to compare must be of the same length.");

            var delta = new double[a.Length];
            for (int i = 0; i < a.Length; i++)
                delta[i] = a[i] - b[i];
            return delta;
        }

        #endregion
    }
}

namespace Microsoft.ML.Data
{
    #region MetricsStatistics

    /// <summary>
    /// The MetricsStatistics class computes summary statistics over multiple observations of a metric.
    /// </summary>
    public sealed class MetricStatistics
    {
        private readonly SummaryStatistics _statistic;

        /// <summary>
        /// Get the mean value for the metric
        /// </summary>
        public double Mean => _statistic.Mean;

        /// <summary>
        /// Get the standard deviation for the metric
        /// </summary>
        public double StandardDeviation => (_statistic.RawCount <= 1) ? 0 : _statistic.SampleStdDev;

        /// <summary>
        /// Get the standard error of the mean for the metric
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
        /// Add another metric to the set of observations
        /// </summary>
        /// <param name="metric">The metric being accumulated</param>
        internal void Add(double metric)
        {
            _statistic.Add(metric);
        }
    }

    internal static class MetricsStatisticsUtils
    {
        public static void AddArray(double[] src, MetricStatistics[] dest)
        {
            Contracts.Assert(src.Length == dest.Length, "Array sizes do not match.");

            for (int i = 0; i < dest.Length; i++)
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
        /// Summary Statistics for L1
        /// </summary>
        public MetricStatistics MeanAbsoluteError { get; }

        /// <summary>
        /// Summary Statistics for L2
        /// </summary>
        public MetricStatistics MeanSquaredError { get; }

        /// <summary>
        /// Summary statistics for the root mean square loss (or RMS).
        /// </summary>
        public MetricStatistics RootMeanSquaredError { get; }

        /// <summary>
        /// Summary statistics for the user-supplied loss function.
        /// </summary>
        public MetricStatistics LossFunction { get; }

        /// <summary>
        /// Summary statistics for the R squared value.
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
        /// Summary Statistics for AUC
        /// </summary>
        public MetricStatistics AreaUnderRocCurve { get; }

        /// <summary>
        /// Summary Statistics for Accuracy
        /// </summary>
        public MetricStatistics Accuracy { get; }

        /// <summary>
        /// Summary statistics for Positive Precision
        /// </summary>
        public MetricStatistics PositivePrecision { get; }

        /// <summary>
        /// Summary statistics for Positive Recall
        /// </summary>
        public MetricStatistics PositiveRecall { get; }

        /// <summary>
        /// Summary statistics for Negative Precision.
        /// </summary>
        public MetricStatistics NegativePrecision { get; }

        /// <summary>
        /// Summary statistics for Negative Recall.
        /// </summary>
        public MetricStatistics NegativeRecall { get; }

        /// <summary>
        /// Summary statistics for F1Score.
        /// </summary>
        public MetricStatistics F1Score { get; }

        /// <summary>
        /// Summary statistics for AUPRC.
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
    /// The <see cref="MultiClassClassifierMetricsStatistics"/> class holds summary
    /// statistics over multiple observations of <see cref="MultiClassClassifierMetrics"/>.
    /// </summary>
    public sealed class MultiClassClassifierMetricsStatistics : IMetricsStatistics<MultiClassClassifierMetrics>
    {
        /// <summary>
        /// Summary Statistics for Micro-Accuracy
        /// </summary>
        public MetricStatistics AccuracyMacro { get; }

        /// <summary>
        /// Summary Statistics for Micro-Accuracy
        /// </summary>
        public MetricStatistics AccuracyMicro { get; }

        /// <summary>
        /// Summary statistics for Log Loss
        /// </summary>
        public MetricStatistics LogLoss { get; }

        /// <summary>
        /// Summary statistics for Log Loss Reduction
        /// </summary>
        public MetricStatistics LogLossReduction { get; }

        /// <summary>
        /// Summary statistics for Top K Accuracy
        /// </summary>
        public MetricStatistics TopKAccuracy { get; }

        /// <summary>
        /// Summary statistics for Per Class Log Loss
        /// </summary>
        public MetricStatistics[] PerClassLogLoss { get; private set; }

        internal MultiClassClassifierMetricsStatistics()
        {
            AccuracyMacro = new MetricStatistics();
            AccuracyMicro = new MetricStatistics();
            LogLoss = new MetricStatistics();
            LogLossReduction = new MetricStatistics();
            TopKAccuracy = new MetricStatistics();
        }

        /// <summary>
        /// Add a set of evaluation metrics to the set of observations.
        /// </summary>
        /// <param name="metrics">The observed binary classification evaluation metric</param>
        void IMetricsStatistics<MultiClassClassifierMetrics>.Add(MultiClassClassifierMetrics metrics)
        {
            AccuracyMacro.Add(metrics.MacroAccuracy);
            AccuracyMicro.Add(metrics.MicroAccuracy);
            LogLoss.Add(metrics.LogLoss);
            LogLossReduction.Add(metrics.LogLossReduction);
            TopKAccuracy.Add(metrics.TopKAccuracy);

            if (PerClassLogLoss == null)
                PerClassLogLoss = MetricsStatisticsUtils.InitializeArray(metrics.PerClassLogLoss.Length);
            MetricsStatisticsUtils.AddArray(metrics.PerClassLogLoss, PerClassLogLoss);
        }
    }

    /// <summary>
    /// The <see cref="RankingMetricsStatistics"/> class holds summary
    /// statistics over multiple observations of <see cref="RankingMetrics"/>.
    /// </summary>
    public sealed class RankingMetricsStatistics : IMetricsStatistics<RankingMetrics>
    {
        /// <summary>
        /// Summary Statistics for DCG
        /// </summary>
        public MetricStatistics[] DiscountedCumulativeGains { get; private set; }

        /// <summary>
        /// Summary Statistics for L2
        /// </summary>
        public MetricStatistics[] NormalizedDiscountedCumulativeGains { get; private set; }

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
                DiscountedCumulativeGains = MetricsStatisticsUtils.InitializeArray(metrics.DiscountedCumulativeGains.Length);

            if (NormalizedDiscountedCumulativeGains == null)
                NormalizedDiscountedCumulativeGains = MetricsStatisticsUtils.InitializeArray(metrics.NormalizedDiscountedCumulativeGains.Length);

            MetricsStatisticsUtils.AddArray(metrics.DiscountedCumulativeGains, DiscountedCumulativeGains);
            MetricsStatisticsUtils.AddArray(metrics.NormalizedDiscountedCumulativeGains, NormalizedDiscountedCumulativeGains);
        }
    }

    #endregion
}
