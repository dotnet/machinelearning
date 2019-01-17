// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Immutable;
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
            PermutationFeatureImportance(
                this RegressionCatalog catalog,
                IPredictionTransformer<IPredictor> model,
                IDataView data,
                string label = DefaultColumnNames.Label,
                string features = DefaultColumnNames.Features,
                bool useFeatureWeightFilter = false,
                int? topExamples = null,
                int permutationCount = 1)
        {
            return PermutationFeatureImportance<RegressionMetrics, RegressionMetricsStatistics>.GetImportanceMetricsMatrix(
                            CatalogUtils.GetEnvironment(catalog),
                            model,
                            data,
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
                l1: a.L1 - b.L1,
                l2: a.L2 - b.L2,
                rms: a.Rms - b.Rms,
                lossFunction: a.LossFn - b.LossFn,
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
            PermutationFeatureImportance(
                this BinaryClassificationCatalog catalog,
                IPredictionTransformer<IPredictor> model,
                IDataView data,
                string label = DefaultColumnNames.Label,
                string features = DefaultColumnNames.Features,
                bool useFeatureWeightFilter = false,
                int? topExamples = null,
                int permutationCount = 1)
        {
            return PermutationFeatureImportance<BinaryClassificationMetrics, BinaryClassificationMetricsStatistics>.GetImportanceMetricsMatrix(
                            CatalogUtils.GetEnvironment(catalog),
                            model,
                            data,
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
                auc: a.Auc - b.Auc,
                accuracy: a.Accuracy - b.Accuracy,
                positivePrecision: a.PositivePrecision - b.PositivePrecision,
                positiveRecall: a.PositiveRecall - b.PositiveRecall,
                negativePrecision: a.NegativePrecision - b.NegativePrecision,
                negativeRecall: a.NegativeRecall - b.NegativeRecall,
                f1Score: a.F1Score - b.F1Score,
                auprc: a.Auprc - b.Auprc);
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
            PermutationFeatureImportance(
                this MulticlassClassificationCatalog catalog,
                IPredictionTransformer<IPredictor> model,
                IDataView data,
                string label = DefaultColumnNames.Label,
                string features = DefaultColumnNames.Features,
                bool useFeatureWeightFilter = false,
                int? topExamples = null,
                int permutationCount = 1)
        {
            return PermutationFeatureImportance<MultiClassClassifierMetrics, MultiClassClassifierMetricsStatistics>.GetImportanceMetricsMatrix(
                            CatalogUtils.GetEnvironment(catalog),
                            model,
                            data,
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
                accuracyMicro: a.AccuracyMicro - b.AccuracyMicro,
                accuracyMacro: a.AccuracyMacro - b.AccuracyMacro,
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
        public static ImmutableArray<RankerMetricsStatistics>
            PermutationFeatureImportance(
                this RankingCatalog catalog,
                IPredictionTransformer<IPredictor> model,
                IDataView data,
                string label = DefaultColumnNames.Label,
                string groupId = DefaultColumnNames.GroupId,
                string features = DefaultColumnNames.Features,
                bool useFeatureWeightFilter = false,
                int? topExamples = null,
                int permutationCount = 1)
        {
            return PermutationFeatureImportance<RankerMetrics, RankerMetricsStatistics>.GetImportanceMetricsMatrix(
                            CatalogUtils.GetEnvironment(catalog),
                            model,
                            data,
                            idv => catalog.Evaluate(idv, label, groupId),
                            RankingDelta,
                            features,
                            permutationCount,
                            useFeatureWeightFilter,
                            topExamples);
        }

        private static RankerMetrics RankingDelta(
            RankerMetrics a, RankerMetrics b)
        {
            var dcg = ComputeArrayDeltas(a.Dcg, b.Dcg);
            var ndcg = ComputeArrayDeltas(a.Ndcg, b.Ndcg);

            return new RankerMetrics(dcg: dcg, ndcg: ndcg);
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
        public int Count => (int) _statistic.RawCount;

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

    /// <summary>
    /// The MetricsStatisticsBase class is the base class for computing summary
    /// statistics over multiple observations of model evaluation metrics.
    /// </summary>
    /// <typeparam name="T">The EvaluationMetric type, such as RegressionMetrics</typeparam>
    public abstract class MetricsStatisticsBase<T>{
        internal MetricsStatisticsBase()
        {
        }

        public abstract void Add(T metrics);

        protected static void AddArray(double[] src, MetricStatistics[] dest)
        {
            Contracts.Assert(src.Length == dest.Length, "Array sizes do not match.");

            for (int i = 0; i < dest.Length; i++)
                dest[i].Add(src[i]);
        }

        protected MetricStatistics[] InitializeArray(int length)
        {
            var array = new MetricStatistics[length];
            for (int i = 0; i < array.Length; i++)
                array[i] = new MetricStatistics();

            return array;
        }
    }

    /// <summary>
    /// The RegressionMetricsStatistics class is computes summary
    /// statistics over multiple observations of regression evaluation metrics.
    /// </summary>
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

        /// <summary>
        /// Add a set of evaluation metrics to the set of observations.
        /// </summary>
        /// <param name="metrics">The observed regression evaluation metric</param>
        public override void Add(RegressionMetrics metrics)
        {
            L1.Add(metrics.L1);
            L2.Add(metrics.L2);
            Rms.Add(metrics.Rms);
            LossFn.Add(metrics.LossFn);
            RSquared.Add(metrics.RSquared);
        }
    }

    /// <summary>
    /// The BinaryClassificationMetricsStatistics class is computes summary
    /// statistics over multiple observations of binary classification evaluation metrics.
    /// </summary>
    public sealed class BinaryClassificationMetricsStatistics : MetricsStatisticsBase<BinaryClassificationMetrics>
    {
        /// <summary>
        /// Summary Statistics for AUC
        /// </summary>
        public MetricStatistics Auc { get; }

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

        /// <summary>
        /// Add a set of evaluation metrics to the set of observations.
        /// </summary>
        /// <param name="metrics">The observed binary classification evaluation metric</param>
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

    /// <summary>
    /// The MultiClassClassifierMetricsStatistics class is computes summary
    /// statistics over multiple observations of binary classification evaluation metrics.
    /// </summary>
    public sealed class MultiClassClassifierMetricsStatistics : MetricsStatisticsBase<MultiClassClassifierMetrics>
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

        public MultiClassClassifierMetricsStatistics()
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
        public override void Add(MultiClassClassifierMetrics metrics)
        {
            AccuracyMacro.Add(metrics.AccuracyMacro);
            AccuracyMicro.Add(metrics.AccuracyMicro);
            LogLoss.Add(metrics.LogLoss);
            LogLossReduction.Add(metrics.LogLossReduction);
            TopKAccuracy.Add(metrics.TopKAccuracy);

            if (PerClassLogLoss == null)
                PerClassLogLoss = InitializeArray(metrics.PerClassLogLoss.Length);
            AddArray(metrics.PerClassLogLoss, PerClassLogLoss);
        }
    }

    /// <summary>
    /// The RankerMetricsStatistics class is computes summary
    /// statistics over multiple observations of regression evaluation metrics.
    /// </summary>
    public sealed class RankerMetricsStatistics : MetricsStatisticsBase<RankerMetrics>
    {
        /// <summary>
        /// Summary Statistics for DCG
        /// </summary>
        public MetricStatistics[] Dcg { get; private set; }

        /// <summary>
        /// Summary Statistics for L2
        /// </summary>
        public MetricStatistics[] Ndcg { get; private set; }

        /// <summary>
        /// Add a set of evaluation metrics to the set of observations.
        /// </summary>
        /// <param name="metrics">The observed regression evaluation metric</param>
        public override void Add(RankerMetrics metrics)
        {
            if (Dcg == null)
                Dcg = InitializeArray(metrics.Dcg.Length);

            if (Ndcg == null)
                Ndcg = InitializeArray(metrics.Ndcg.Length);

            AddArray(metrics.Dcg, Dcg);
            AddArray(metrics.Ndcg, Ndcg);
        }
    }

    #endregion
}
