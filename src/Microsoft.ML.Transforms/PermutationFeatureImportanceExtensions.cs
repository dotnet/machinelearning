// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Transforms;
using System.Collections.Generic;
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
        /// <returns>Array of per-feature 'contributions' to the score.</returns>
        public static ImmutableArray<RegressionMetricsDelta>
            PermutationFeatureImportance(
                this RegressionContext ctx,
                IPredictionTransformer<IPredictor> model,
                IDataView data,
                string label = DefaultColumnNames.Label,
                string features = DefaultColumnNames.Features,
                bool useFeatureWeightFilter = false,
                int? topExamples = null)
        {
            return PermutationFeatureImportance<RegressionMetrics, RegressionMetricsDelta>.GetImportanceMetricsMatrix(
                            CatalogUtils.GetEnvironment(ctx),
                            model,
                            data,
                            idv => ctx.Evaluate(idv, label),
                            RegressionDelta,
                            AverageRegressionMetrics,
                            features,
                            useFeatureWeightFilter,
                            topExamples);
        }

        private static RegressionMetricsDelta RegressionDelta(
            RegressionMetrics a, RegressionMetrics b)
        {
            return new RegressionMetricsDelta(
                l1: a.L1 - b.L1,
                l2: a.L2 - b.L2,
                rms: a.Rms - b.Rms,
                lossFunction: a.LossFn - b.LossFn,
                rSquared: a.RSquared - b.RSquared);
        }

        private static RegressionMetricsDelta AverageRegressionMetrics(
            IEnumerable<RegressionMetricsDelta> metricsDeltas)
        {
            var l1 = new SummaryStatistics();
            var l2 = new SummaryStatistics();
            var rms = new SummaryStatistics();
            var lossFunction = new SummaryStatistics();
            var rsquared = new SummaryStatistics();

            foreach (var delta in metricsDeltas)
            {
                l1.Add(delta.L1);
                l2.Add(delta.L2);
                rms.Add(delta.Rms);
                lossFunction.Add(delta.LossFn);
                rsquared.Add(delta.RSquared);
            }

            return new RegressionMetricsDelta(
                l1: l1.Mean,
                l1Std: l1.SampleStdDev,
                l2: l2.Mean,
                l2Std: l2.SampleStdDev,
                rms: rms.Mean,
                rmsStd: rms.SampleStdDev,
                lossFunction: lossFunction.Mean,
                lossFunctionStd: lossFunction.SampleStdDev,
                rSquared: rsquared.Mean,
                rSquaredStd: rsquared.SampleStdDev);
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
        /// <returns>Array of per-feature 'contributions' to the score.</returns>
        public static ImmutableArray<BinaryClassificationMetrics>
            PermutationFeatureImportance(
                this BinaryClassificationContext ctx,
                IPredictionTransformer<IPredictor> model,
                IDataView data,
                string label = DefaultColumnNames.Label,
                string features = DefaultColumnNames.Features,
                bool useFeatureWeightFilter = false,
                int? topExamples = null)
        {
            return PermutationFeatureImportance<BinaryClassificationMetrics, BinaryClassificationMetrics>.GetImportanceMetricsMatrix(
                            CatalogUtils.GetEnvironment(ctx),
                            model,
                            data,
                            idv => ctx.Evaluate(idv, label),
                            BinaryClassifierDelta,
                            AverageMetrics,
                            features,
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

        private static BinaryClassificationMetrics AverageMetrics(
            IEnumerable<BinaryClassificationMetrics> metricsDeltas)
        {
            return new BinaryClassificationMetrics(
                auc: 0,
                accuracy: 0,
                positivePrecision: 0,
                positiveRecall: 0,
                negativePrecision: 0,
                negativeRecall: 0,
                f1Score: 0,
                auprc: 0);
        }
    }

    public sealed class RegressionMetricsDelta
    {
        /// <summary>
        /// Gets the difference of the absolute loss of the model.
        /// </summary>
        /// <remarks>
        /// The absolute loss is defined as
        /// L1 = (1/m) * sum( abs( yi - y&apos;i))
        /// where m is the number of instances in the test set.
        /// y'i are the predicted labels for each instance.
        /// yi are the correct labels of each instance.
        /// </remarks>
        public double L1 { get; }

        public double L1Std { get;  }

        /// <summary>
        /// Gets the squared loss of the model.
        /// </summary>
        /// <remarks>
        /// The squared loss is defined as
        /// L2 = (1/m) * sum(( yi - y&apos;i)^2)
        /// where m is the number of instances in the test set.
        /// y'i are the predicted labels for each instance.
        /// yi are the correct labels of each instance.
        /// </remarks>
        public double L2 { get; }

        public double L2Std { get; }

        /// <summary>
        /// Gets the root mean square loss (or RMS) which is the square root of the L2 loss.
        /// </summary>
        public double Rms { get; }

        public double RmsStd { get; }

        /// <summary>
        /// Gets the result of user defined loss function.
        /// </summary>
        /// <remarks>
        /// This is the average of a loss function defined by the user,
        /// computed over all the instances in the test set.
        /// </remarks>
        public double LossFn { get; }

        public double LossFnStd { get; }

        /// <summary>
        /// Gets the R squared value of the model, which is also known as
        /// the coefficient of determination​.
        /// </summary>
        public double RSquared { get; }

        public double RSquaredsStd { get; }

        [BestFriend]
        internal RegressionMetricsDelta(double l1, double l2, double rms, double lossFunction, double rSquared,
            double l1Std = 0, double l2Std = 0, double rmsStd = 0,
            double lossFunctionStd = 0, double rSquaredStd = 0)
        {
            L1 = l1;
            L1Std = l1Std;
            L2 = l2;
            L2Std = l2Std;
            Rms = rms;
            RmsStd = rmsStd;
            LossFn = lossFunction;
            LossFnStd = lossFunctionStd;
            RSquared = rSquared;
            RSquaredsStd = rSquaredStd;
        }
    }
}
