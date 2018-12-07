// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Transforms;
using System.Collections.Immutable;

namespace Microsoft.ML
{
    public static class PermutationFeatureImportanceExtensions
    {
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
        /// the original values for all other features. The evalution metric (e.g. AUC or R-squared) is then calculated
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
        /// [!code-csharp[PFI](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/PermutationFeatureImportance.cs)]
        /// ]]>
        /// </format>
        /// </example>
        /// <param name="ctx">The regression context.</param>
        /// <param name="model">The model to evaluate.</param>
        /// <param name="data">The evaluation data set.</param>
        /// <param name="label">Label column name.</param>
        /// <param name="features">Feature column names.</param>
        /// <param name="useFeatureWeightFilter">Use features weight to pre-filter features.</param>
        /// <param name="topExamples">Limit the number of examples to evaluate on. null means examples (up to ~ 2 bln) from input will be used.</param>
        /// <param name="permutationCount">The number of permutations to perform.</param>
        /// <returns>Array of per-feature 'contributions' to the score.</returns>
        public static ImmutableArray<RegressionMetricsStatistics>
            PermutationFeatureImportance(
                this RegressionContext ctx,
                IPredictionTransformer<IPredictor> model,
                IDataView data,
                string label = DefaultColumnNames.Label,
                string features = DefaultColumnNames.Features,
                bool useFeatureWeightFilter = false,
                int? topExamples = null,
                int permutationCount = 1)
        {
            return PermutationFeatureImportance<RegressionMetrics, RegressionMetricsStatistics>.GetImportanceMetricsMatrix(
                            CatalogUtils.GetEnvironment(ctx),
                            model,
                            data,
                            idv => ctx.Evaluate(idv, label),
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
                l1: a.L1 - b.L1,
                l2: a.L2 - b.L2,
                rms: a.Rms - b.Rms,
                lossFunction: a.LossFn - b.LossFn,
                rSquared: a.RSquared - b.RSquared);
        }

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
        /// the original values for all other features. The evalution metric (e.g. AUC or R-squared) is then calculated
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
        /// [!code-csharp[PFI](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/PermutationFeatureImportance.cs)]
        /// ]]>
        /// </format>
        /// </example>
        /// <param name="ctx">The binary classification context.</param>
        /// <param name="model">The model to evaluate.</param>
        /// <param name="data">The evaluation data set.</param>
        /// <param name="label">Label column name.</param>
        /// <param name="features">Feature column names.</param>
        /// <param name="useFeatureWeightFilter">Use features weight to pre-filter features.</param>
        /// <param name="topExamples">Limit the number of examples to evaluate on. null means examples (up to ~ 2 bln) from input will be used.</param>
        /// <param name="permutationCount">The number of permutations to perform.</param>
        /// <returns>Array of per-feature 'contributions' to the score.</returns>
        public static ImmutableArray<BinaryClassificationMetricsStatistics>
            PermutationFeatureImportance(
                this BinaryClassificationContext ctx,
                IPredictionTransformer<IPredictor> model,
                IDataView data,
                string label = DefaultColumnNames.Label,
                string features = DefaultColumnNames.Features,
                bool useFeatureWeightFilter = false,
                int? topExamples = null,
                int permutationCount = 1)
        {
            return PermutationFeatureImportance<BinaryClassificationMetrics, BinaryClassificationMetricsStatistics>.GetImportanceMetricsMatrix(
                            CatalogUtils.GetEnvironment(ctx),
                            model,
                            data,
                            idv => ctx.Evaluate(idv, label),
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
                auc: a.Auc - b.Auc,
                accuracy: a.Accuracy - b.Accuracy,
                positivePrecision: a.PositivePrecision - b.PositivePrecision,
                positiveRecall: a.PositiveRecall - b.PositiveRecall,
                negativePrecision: a.NegativePrecision - b.NegativePrecision,
                negativeRecall: a.NegativeRecall - b.NegativeRecall,
                f1Score: a.F1Score - b.F1Score,
                auprc: a.Auprc - b.Auprc);
        }
    }

    public sealed class MetricStatistics
    {
        /// <summary>
        /// Get the mean value for the metric
        /// </summary>
        public double Mean => _statistic.Mean;

        /// <summary>
        /// Get the standard deviation for the metric
        /// </summary>
        public double StandardDeviation => ComputeStandardDeviation();

        /// <summary>
        /// Get the standard error of the mean for the metric
        /// </summary>
        public double StandardError => ComputeStandardError();

        private SummaryStatistics _statistic;

        internal MetricStatistics()
        {
            _statistic = new SummaryStatistics();
        }

        /// <summary>
        /// Add another metric
        /// </summary>
        /// <param name="metric">The metric being accumulated</param>
        internal void Add(double metric)
        {
            _statistic.Add(metric);
        }

        private double ComputeStandardDeviation()
        {
            double standardDeviation = 0;
            // Protect against a divid-by-zero
            if (_statistic.RawCount > 2)
                standardDeviation = _statistic.SampleStdDev;

            return standardDeviation;
        }

        private double ComputeStandardError()
        {
            double standardError = 0;
            // Protect against a divid-by-zero
            if (_statistic.RawCount > 2)
                standardError = _statistic.StandardErrorMean;

            return standardError;
        }
    }

    public abstract class MetricsStatisticsBase<T>{
        internal MetricsStatisticsBase()
        {
        }

        public abstract void Add(T metrics);
    }

    public sealed class RegressionMetricsStatistics : MetricsStatisticsBase<RegressionMetrics>
    {
        /// <summary>
        /// Summary Statistics for L1
        /// </summary>
        public MetricStatistics L1 { get; }

        /// <summary>
        /// Summary Statistics for L2
        /// </summary>
        public MetricStatistics L2 { get; }

        /// <summary>
        /// Summary statistics for the root mean square loss (or RMS).
        /// </summary>
        public MetricStatistics Rms { get; }

        /// <summary>
        /// Summary statistics for the user-supplied loss function.
        /// </summary>
        public MetricStatistics LossFn { get; }

        /// <summary>
        /// Summary statistics for the R squared value.
        /// </summary>
        public MetricStatistics RSquared { get; }

        public RegressionMetricsStatistics()
        {
            L1 = new MetricStatistics();
            L2 = new MetricStatistics();
            Rms = new MetricStatistics();
            LossFn = new MetricStatistics();
            RSquared = new MetricStatistics();
        }

        public override void Add(RegressionMetrics metrics)
        {
            L1.Add(metrics.L1);
            L2.Add(metrics.L2);
            Rms.Add(metrics.Rms);
            LossFn.Add(metrics.LossFn);
            RSquared.Add(metrics.RSquared);
        }
    }

    public sealed class BinaryClassificationMetricsStatistics : MetricsStatisticsBase<BinaryClassificationMetrics>
    {
        /// <summary>
        /// Summary Statistics for L1
        /// </summary>
        public MetricStatistics Auc { get; }

        /// <summary>
        /// Summary Statistics for L2
        /// </summary>
        public MetricStatistics Accuracy { get; }

        /// <summary>
        /// Summary statistics for the root mean square loss (or RMS).
        /// </summary>
        public MetricStatistics PositivePrecision { get; }

        /// <summary>
        /// Summary statistics for the user-supplied loss function.
        /// </summary>
        public MetricStatistics PositiveRecall { get; }

        /// <summary>
        /// Summary statistics for the R squared value.
        /// </summary>
        public MetricStatistics NegativePrecision { get; }

        /// <summary>
        /// Summary statistics for the R squared value.
        /// </summary>
        public MetricStatistics NegativeRecall { get; }

        /// <summary>
        /// Summary statistics for the R squared value.
        /// </summary>
        public MetricStatistics F1Score { get; }

        /// <summary>
        /// Summary statistics for the R squared value.
        /// </summary>
        public MetricStatistics Auprc { get; }

        public BinaryClassificationMetricsStatistics()
        {
            Auc = new MetricStatistics();
            Accuracy = new MetricStatistics();
            PositivePrecision = new MetricStatistics();
            PositiveRecall = new MetricStatistics();
            NegativePrecision = new MetricStatistics();
            NegativeRecall = new MetricStatistics();
            F1Score = new MetricStatistics();
            Auprc = new MetricStatistics();
        }

        public override void Add(BinaryClassificationMetrics metrics)
        {
            Auc.Add(metrics.Auc);
            Accuracy.Add(metrics.Accuracy);
            PositivePrecision.Add(metrics.PositivePrecision);
            PositiveRecall.Add(metrics.PositiveRecall);
            NegativePrecision.Add(metrics.NegativePrecision);
            NegativeRecall.Add(metrics.NegativeRecall);
            F1Score.Add(metrics.F1Score);
            Auprc.Add(metrics.Auprc);
        }
    }
}
