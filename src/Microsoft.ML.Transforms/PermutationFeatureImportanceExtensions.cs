// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Collections.Immutable;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
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
        /// <param name="predictionTransformer">The model on which to evaluate feature importance.</param>
        /// <param name="data">The evaluation data set.</param>
        /// <param name="labelColumnName">Label column name.</param>
        /// <param name="useFeatureWeightFilter">Use features weight to pre-filter features.</param>
        /// <param name="numberOfExamplesToUse">Limit the number of examples to evaluate on. <cref langword="null"/> means up to ~2 bln examples from <paramref param="data"/> will be used.</param>
        /// <param name="permutationCount">The number of permutations to perform.</param>
        /// <returns>Array of per-feature 'contributions' to the score.</returns>
        public static ImmutableArray<RegressionMetricsStatistics>
            PermutationFeatureImportance<TModel>(
                this RegressionCatalog catalog,
                ISingleFeaturePredictionTransformer<TModel> predictionTransformer,
                IDataView data,
                string labelColumnName = DefaultColumnNames.Label,
                bool useFeatureWeightFilter = false,
                int? numberOfExamplesToUse = null,
                int permutationCount = 1) where TModel : class
        {
            return PermutationFeatureImportance<TModel, RegressionMetrics, RegressionMetricsStatistics>.GetImportanceMetricsMatrix(
                            catalog.GetEnvironment(),
                            predictionTransformer,
                            data,
                            () => new RegressionMetricsStatistics(),
                            idv => catalog.Evaluate(idv, labelColumnName),
                            RegressionDelta,
                            predictionTransformer.FeatureColumnName,
                            permutationCount,
                            useFeatureWeightFilter,
                            numberOfExamplesToUse);
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
        /// <param name="predictionTransformer">The model on which to evaluate feature importance.</param>
        /// <param name="data">The evaluation data set.</param>
        /// <param name="labelColumnName">Label column name.</param>
        /// <param name="useFeatureWeightFilter">Use features weight to pre-filter features.</param>
        /// <param name="numberOfExamplesToUse">Limit the number of examples to evaluate on. <cref langword="null"/> means up to ~2 bln examples from <paramref param="data"/> will be used.</param>
        /// <param name="permutationCount">The number of permutations to perform.</param>
        /// <returns>Array of per-feature 'contributions' to the score.</returns>
        public static ImmutableArray<BinaryClassificationMetricsStatistics>
            PermutationFeatureImportance<TModel>(
                this BinaryClassificationCatalog catalog,
                ISingleFeaturePredictionTransformer<TModel> predictionTransformer,
                IDataView data,
                string labelColumnName = DefaultColumnNames.Label,
                bool useFeatureWeightFilter = false,
                int? numberOfExamplesToUse = null,
                int permutationCount = 1) where TModel : class
        {
            return PermutationFeatureImportance<TModel, BinaryClassificationMetrics, BinaryClassificationMetricsStatistics>.GetImportanceMetricsMatrix(
                            catalog.GetEnvironment(),
                            predictionTransformer,
                            data,
                            () => new BinaryClassificationMetricsStatistics(),
                            idv => catalog.Evaluate(idv, labelColumnName),
                            BinaryClassifierDelta,
                            predictionTransformer.FeatureColumnName,
                            permutationCount,
                            useFeatureWeightFilter,
                            numberOfExamplesToUse);
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
        /// <code>ImmutableArray</code> of <code>MulticlassClassificationMetrics</code> objects is returned. See the sample below for an
        /// example of working with these results to analyze the feature importance of a model.
        /// </para>
        /// </remarks>
        /// <param name="catalog">The clustering catalog.</param>
        /// <param name="predictionTransformer">The model on which to evaluate feature importance.</param>
        /// <param name="data">The evaluation data set.</param>
        /// <param name="labelColumnName">Label column name.</param>
        /// <param name="useFeatureWeightFilter">Use features weight to pre-filter features.</param>
        /// <param name="numberOfExamplesToUse">Limit the number of examples to evaluate on. <cref langword="null"/> means up to ~2 bln examples from <paramref param="data"/> will be used.</param>
        /// <param name="permutationCount">The number of permutations to perform.</param>
        /// <returns>Array of per-feature 'contributions' to the score.</returns>
        public static ImmutableArray<MulticlassClassificationMetricsStatistics>
            PermutationFeatureImportance<TModel>(
                this MulticlassClassificationCatalog catalog,
                ISingleFeaturePredictionTransformer<TModel> predictionTransformer,
                IDataView data,
                string labelColumnName = DefaultColumnNames.Label,
                bool useFeatureWeightFilter = false,
                int? numberOfExamplesToUse = null,
                int permutationCount = 1) where TModel : class
        {
            return PermutationFeatureImportance<TModel, MulticlassClassificationMetrics, MulticlassClassificationMetricsStatistics>.GetImportanceMetricsMatrix(
                            catalog.GetEnvironment(),
                            predictionTransformer,
                            data,
                            () => new MulticlassClassificationMetricsStatistics(),
                            idv => catalog.Evaluate(idv, labelColumnName),
                            MulticlassClassificationDelta,
                            predictionTransformer.FeatureColumnName,
                            permutationCount,
                            useFeatureWeightFilter,
                            numberOfExamplesToUse);
        }

        private static MulticlassClassificationMetrics MulticlassClassificationDelta(
            MulticlassClassificationMetrics a, MulticlassClassificationMetrics b)
        {
            if (a.TopK != b.TopK)
                Contracts.Assert(a.TopK == b.TopK, "TopK to compare must be the same length.");

            var perClassLogLoss = ComputeSequenceDeltas(a.PerClassLogLoss, b.PerClassLogLoss);

            return new MulticlassClassificationMetrics(
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
        /// <param name="predictionTransformer">The model on which to evaluate feature importance.</param>
        /// <param name="data">The evaluation data set.</param>
        /// <param name="labelColumnName">Label column name.</param>
        /// <param name="rowGroupColumnName">GroupId column name</param>
        /// <param name="useFeatureWeightFilter">Use features weight to pre-filter features.</param>
        /// <param name="numberOfExamplesToUse">Limit the number of examples to evaluate on. <cref langword="null"/> means up to ~2 bln examples from <paramref param="data"/> will be used.</param>
        /// <param name="permutationCount">The number of permutations to perform.</param>
        /// <returns>Array of per-feature 'contributions' to the score.</returns>
        public static ImmutableArray<RankingMetricsStatistics>
            PermutationFeatureImportance<TModel>(
                this RankingCatalog catalog,
                ISingleFeaturePredictionTransformer<TModel> predictionTransformer,
                IDataView data,
                string labelColumnName = DefaultColumnNames.Label,
                string rowGroupColumnName = DefaultColumnNames.GroupId,
                bool useFeatureWeightFilter = false,
                int? numberOfExamplesToUse = null,
                int permutationCount = 1) where TModel : class
        {
            return PermutationFeatureImportance<TModel, RankingMetrics, RankingMetricsStatistics>.GetImportanceMetricsMatrix(
                            catalog.GetEnvironment(),
                            predictionTransformer,
                            data,
                            () => new RankingMetricsStatistics(),
                            idv => catalog.Evaluate(idv, labelColumnName, rowGroupColumnName),
                            RankingDelta,
                            predictionTransformer.FeatureColumnName,
                            permutationCount,
                            useFeatureWeightFilter,
                            numberOfExamplesToUse);
        }

        private static RankingMetrics RankingDelta(
            RankingMetrics a, RankingMetrics b)
        {
            var dcg = ComputeSequenceDeltas(a.DiscountedCumulativeGains, b.DiscountedCumulativeGains);
            var ndcg = ComputeSequenceDeltas(a.NormalizedDiscountedCumulativeGains, b.NormalizedDiscountedCumulativeGains);

            return new RankingMetrics(dcg: dcg, ndcg: ndcg);
        }

        #endregion

        #region Helpers

        private static double[] ComputeSequenceDeltas(IReadOnlyList<double> a, IReadOnlyList<double> b)
        {
            Contracts.Assert(a.Count == b.Count);

            var delta = new double[a.Count];
            for (int i = 0; i < a.Count; i++)
                delta[i] = a[i] - b[i];
            return delta;
        }

        #endregion
    }
}
