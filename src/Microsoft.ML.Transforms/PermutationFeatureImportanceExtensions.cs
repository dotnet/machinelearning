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
        /// <param name="numPermutations">The number of permutations to perform.</param>
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
                int numPermutations = 1)
        {
            return PermutationFeatureImportance<RegressionMetrics, RegressionMetricsStatistics>.GetImportanceMetricsMatrix(
                            CatalogUtils.GetEnvironment(ctx),
                            model,
                            data,
                            idv => ctx.Evaluate(idv, label),
                            RegressionDelta,
                            features,
                            numPermutations,
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
        /// <param name="numPermutations">The number of permutations to perform.</param>
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
                int numPermutations = 1)
        {
            return PermutationFeatureImportance<BinaryClassificationMetrics, BinaryClassificationMetricsStatistics>.GetImportanceMetricsMatrix(
                            CatalogUtils.GetEnvironment(ctx),
                            model,
                            data,
                            idv => ctx.Evaluate(idv, label),
                            BinaryClassifierDelta,
                            features,
                            numPermutations,
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

    public sealed class SimpleStatistics
    {
        public readonly double Mean;
        public readonly double StandardDeviation;

        public SimpleStatistics(double mean, double standardDeviation)
        {
            Mean = mean;
            StandardDeviation = standardDeviation;
        }
    }

    public abstract class MetricsStatisticsBase<T>{
        internal MetricsStatisticsBase()
        {
        }

        public abstract void Add(T metrics);

        internal SimpleStatistics GetStatistics(SummaryStatistics summaryStatistics)
        {
            double standardDeviation = 0;
            // Protect against a divid-by-zero
            if (summaryStatistics.RawCount > 2)
                standardDeviation = summaryStatistics.SampleStdDev;

            return new SimpleStatistics(summaryStatistics.Mean, standardDeviation);
        }
    }

    public sealed class RegressionMetricsStatistics : MetricsStatisticsBase<RegressionMetrics>
    {
        /// <summary>
        /// Summary Statistics for L1
        /// </summary>
        public SimpleStatistics L1 => GetStatistics(_l1);
        private SummaryStatistics _l1;

        /// <summary>
        /// Summary Statistics for L2
        /// </summary>
        public SimpleStatistics L2 => GetStatistics(_l2);

        private SummaryStatistics _l2;

        /// <summary>
        /// Summary statistics for the root mean square loss (or RMS).
        /// </summary>
        public SimpleStatistics Rms => GetStatistics(_rms);

        private SummaryStatistics _rms;

        /// <summary>
        /// Summary statistics for the user-supplied loss function.
        /// </summary>
        public SimpleStatistics LossFn => GetStatistics(_lossFn);

        private SummaryStatistics _lossFn;

        /// <summary>
        /// Summary statistics for the R squared value.
        /// </summary>
        public SimpleStatistics RSquared => GetStatistics(_rSquared);

        private SummaryStatistics _rSquared;

        public RegressionMetricsStatistics()
        {
            _l1 = new SummaryStatistics();
            _l2 = new SummaryStatistics();
            _rms = new SummaryStatistics();
            _lossFn = new SummaryStatistics();
            _rSquared = new SummaryStatistics();
        }

        public override void Add(RegressionMetrics metrics)
        {
            _l1.Add(metrics.L1);
            _l2.Add(metrics.L2);
            _rms.Add(metrics.Rms);
            _lossFn.Add(metrics.LossFn);
            _rSquared.Add(metrics.RSquared);
        }
    }

    public sealed class BinaryClassificationMetricsStatistics : MetricsStatisticsBase<BinaryClassificationMetrics>
    {
        /// <summary>
        /// Summary Statistics for L1
        /// </summary>
        public SimpleStatistics Auc => GetStatistics(_auc);
        private SummaryStatistics _auc;

        /// <summary>
        /// Summary Statistics for L2
        /// </summary>
        public SimpleStatistics Accuracy => GetStatistics(_accuracy);

        private SummaryStatistics _accuracy;

        /// <summary>
        /// Summary statistics for the root mean square loss (or RMS).
        /// </summary>
        public SimpleStatistics PositivePrecision => GetStatistics(_positivePrecision);

        private SummaryStatistics _positivePrecision;

        /// <summary>
        /// Summary statistics for the user-supplied loss function.
        /// </summary>
        public SimpleStatistics PositiveRecall => GetStatistics(_positiveRecall);

        private SummaryStatistics _positiveRecall;

        /// <summary>
        /// Summary statistics for the R squared value.
        /// </summary>
        public SimpleStatistics NegativePrecision => GetStatistics(_negativePrecision);

        private SummaryStatistics _negativePrecision;

        /// <summary>
        /// Summary statistics for the R squared value.
        /// </summary>
        public SimpleStatistics NegativeRecall => GetStatistics(_negativeRecall);

        private SummaryStatistics _negativeRecall;

        /// <summary>
        /// Summary statistics for the R squared value.
        /// </summary>
        public SimpleStatistics F1Score => GetStatistics(_f1Score);

        private SummaryStatistics _f1Score;

        /// <summary>
        /// Summary statistics for the R squared value.
        /// </summary>
        public SimpleStatistics Auprc => GetStatistics(_auprc);

        private SummaryStatistics _auprc;

        public BinaryClassificationMetricsStatistics()
        {
            _auc = new SummaryStatistics();
            _accuracy = new SummaryStatistics();
            _positivePrecision = new SummaryStatistics();
            _positiveRecall = new SummaryStatistics();
            _negativePrecision = new SummaryStatistics();
            _negativeRecall = new SummaryStatistics();
            _f1Score = new SummaryStatistics();
            _auprc = new SummaryStatistics();
        }

        public override void Add(BinaryClassificationMetrics metrics)
        {
            _auc.Add(metrics.Auc);
            _accuracy.Add(metrics.Accuracy);
            _positivePrecision.Add(metrics.PositivePrecision);
            _positiveRecall.Add(metrics.PositiveRecall);
            _negativePrecision.Add(metrics.NegativePrecision);
            _negativeRecall.Add(metrics.NegativeRecall);
            _f1Score.Add(metrics.F1Score);
            _auprc.Add(metrics.Auprc);
        }
    }
}
