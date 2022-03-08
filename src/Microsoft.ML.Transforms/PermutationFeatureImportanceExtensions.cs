// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    /// <summary>
    /// Collection of extension methods used by <see cref="RegressionCatalog"/>,
    ///  <see cref="BinaryClassificationCatalog"/>, <see cref="MulticlassClassificationCatalog"/>,
    ///  and <see cref="RankingCatalog"/> to create instances of permutation feature importance components.
    /// </summary>
    public static class PermutationFeatureImportanceExtensions
    {
        #region Regression
        /// <summary>
        /// Permutation Feature Importance (PFI) for Regression.
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
        /// the original values for all other features. The evaluation metric (e.g. R-squared) is then calculated
        /// for this modified dataset, and the change in the evaluation metric from the original dataset is computed.
        /// The larger the change in the evaluation metric, the more important the feature is to the model.
        /// PFI works by performing this permutation analysis across all the features of a model, one after another.
        /// </para>
        /// <para>
        /// In this implementation, PFI computes the change in all possible regression evaluation metrics for each feature, and an
        /// <see cref="ImmutableArray"/> of <see cref="RegressionMetrics"/> objects is returned. See the sample below for an
        /// example of working with these results to analyze the feature importance of a model.
        /// </para>
        /// </remarks>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[PermutationFeatureImportance](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Regression/PermutationFeatureImportance.cs)]
        /// ]]>
        /// </format>
        /// </example>
        /// <param name="catalog">The regression catalog.</param>
        /// <param name="predictionTransformer">The model on which to evaluate feature importance.</param>
        /// <param name="data">The evaluation data set.</param>
        /// <param name="labelColumnName">Label column name. The column data must be <see cref="System.Single"/>.</param>
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

        /// <summary>
        /// Permutation Feature Importance (PFI) for Regression.
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
        /// the original values for all other features. The evaluation metric (e.g. R-squared) is then calculated
        /// for this modified dataset, and the change in the evaluation metric from the original dataset is computed.
        /// The larger the change in the evaluation metric, the more important the feature is to the model.
        /// PFI works by performing this permutation analysis across all the features of a model, one after another.
        /// </para>
        /// <para>
        /// In this implementation, PFI computes the change in all possible regression evaluation metrics for each feature, and an
        /// <see cref="ImmutableArray"/> of <see cref="RegressionMetrics"/> objects is returned. See the sample below for an
        /// example of working with these results to analyze the feature importance of a model.
        /// </para>
        /// </remarks>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[PermutationFeatureImportance](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Regression/PermutationFeatureImportance.cs)]
        /// ]]>
        /// </format>
        /// </example>
        /// <param name="catalog">The regression catalog.</param>
        /// <param name="model">The model on which to evaluate feature importance.</param>
        /// <param name="data">The evaluation data set.</param>
        /// <param name="labelColumnName">Label column name. The column data must be <see cref="System.Single"/>.</param>
        /// <param name="useFeatureWeightFilter">Use features weight to pre-filter features.</param>
        /// <param name="numberOfExamplesToUse">Limit the number of examples to evaluate on. <cref langword="null"/> means up to ~2 bln examples from <paramref param="data"/> will be used.</param>
        /// <param name="permutationCount">The number of permutations to perform.</param>
        /// <returns>Dictionary mapping each feature to its per-feature 'contributions' to the score.</returns>
        public static ImmutableDictionary<string, RegressionMetricsStatistics>
            PermutationFeatureImportance(
                this RegressionCatalog catalog,
                ITransformer model,
                IDataView data,
                string labelColumnName = DefaultColumnNames.Label,
                bool useFeatureWeightFilter = false,
                int? numberOfExamplesToUse = null,
                int permutationCount = 1)
        {
            Contracts.CheckValue(catalog, nameof(catalog));

            var env = catalog.GetEnvironment();
            Contracts.CheckValue(env, nameof(env));

            env.CheckValue(data, nameof(data));
            env.CheckValue(model, nameof(model));

            RegressionMetricsStatistics resultInitializer() => new();
            RegressionMetrics evaluationFunc(IDataView idv) => catalog.Evaluate(idv, labelColumnName);

            return PermutationFeatureImportance(
                env,
                model,
                data,
                resultInitializer,
                evaluationFunc,
                RegressionDelta,
                permutationCount,
                useFeatureWeightFilter,
                numberOfExamplesToUse
                );
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
        /// Permutation Feature Importance (PFI) for Binary Classification.
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
        /// the original values for all other features. The evaluation metric (e.g. AUC) is then calculated
        /// for this modified dataset, and the change in the evaluation metric from the original dataset is computed.
        /// The larger the change in the evaluation metric, the more important the feature is to the model.
        /// PFI works by performing this permutation analysis across all the features of a model, one after another.
        /// </para>
        /// <para>
        /// In this implementation, PFI computes the change in all possible binary classification evaluation metrics for each feature, and an
        /// <see cref="ImmutableArray"/> of <see cref="BinaryClassificationMetrics"/> objects is returned. See the sample below for an
        /// example of working with these results to analyze the feature importance of a model.
        /// </para>
        /// </remarks>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[PermutationFeatureImportance](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/PermutationFeatureImportance.cs)]
        /// ]]>
        /// </format>
        /// </example>
        /// <param name="catalog">The binary classification catalog.</param>
        /// <param name="predictionTransformer">The model on which to evaluate feature importance.</param>
        /// <param name="data">The evaluation data set.</param>
        /// <param name="labelColumnName">Label column name. The column data must be <see cref="System.Boolean"/>.</param>
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
                idv => catalog.EvaluateNonCalibrated(idv, labelColumnName),
                BinaryClassifierDelta,
                predictionTransformer.FeatureColumnName,
                permutationCount,
                useFeatureWeightFilter,
                numberOfExamplesToUse);
        }

        /// <summary>
        /// Permutation Feature Importance (PFI) for Binary Classification.
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
        /// the original values for all other features. The evaluation metric (e.g. AUC) is then calculated
        /// for this modified dataset, and the change in the evaluation metric from the original dataset is computed.
        /// The larger the change in the evaluation metric, the more important the feature is to the model.
        /// PFI works by performing this permutation analysis across all the features of a model, one after another.
        /// </para>
        /// <para>
        /// In this implementation, PFI computes the change in all possible binary classification evaluation metrics for each feature, and an
        /// <see cref="ImmutableArray"/> of <see cref="BinaryClassificationMetrics"/> objects is returned. See the sample below for an
        /// example of working with these results to analyze the feature importance of a model.
        /// </para>
        /// </remarks>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[PermutationFeatureImportance](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/PermutationFeatureImportance.cs)]
        /// ]]>
        /// </format>
        /// </example>
        /// <param name="catalog">The binary classification catalog.</param>
        /// <param name="model">The model on which to evaluate feature importance.</param>
        /// <param name="data">The evaluation data set.</param>
        /// <param name="labelColumnName">Label column name. The column data must be <see cref="System.Boolean"/>.</param>
        /// <param name="useFeatureWeightFilter">Use features weight to pre-filter features.</param>
        /// <param name="numberOfExamplesToUse">Limit the number of examples to evaluate on. <cref langword="null"/> means up to ~2 bln examples from <paramref param="data"/> will be used.</param>
        /// <param name="permutationCount">The number of permutations to perform.</param>
        /// <returns>Dictionary mapping each feature to its per-feature 'contributions' to the score.</returns>
        public static ImmutableDictionary<string, BinaryClassificationMetricsStatistics>
            PermutationFeatureImportanceNonCalibrated(
                this BinaryClassificationCatalog catalog,
                ITransformer model,
                IDataView data,
                string labelColumnName = DefaultColumnNames.Label,
                bool useFeatureWeightFilter = false,
                int? numberOfExamplesToUse = null,
                int permutationCount = 1)
        {
            Contracts.CheckValue(catalog, nameof(catalog));

            var env = catalog.GetEnvironment();
            Contracts.CheckValue(env, nameof(env));

            env.CheckValue(data, nameof(data));
            env.CheckValue(model, nameof(model));

            BinaryClassificationMetricsStatistics resultInitializer() => new();
            BinaryClassificationMetrics evaluationFunc(IDataView idv) => catalog.EvaluateNonCalibrated(idv, labelColumnName);

            return PermutationFeatureImportance(
                env,
                model,
                data,
                resultInitializer,
                evaluationFunc,
                BinaryClassifierDelta,
                permutationCount,
                useFeatureWeightFilter,
                numberOfExamplesToUse
                );
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
        /// Permutation Feature Importance (PFI) for MulticlassClassification.
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
        /// the original values for all other features. The evaluation metric (e.g. micro-accuracy) is then calculated
        /// for this modified dataset, and the change in the evaluation metric from the original dataset is computed.
        /// The larger the change in the evaluation metric, the more important the feature is to the model.
        /// PFI works by performing this permutation analysis across all the features of a model, one after another.
        /// </para>
        /// <para>
        /// In this implementation, PFI computes the change in all possible multiclass classification evaluation metrics for each feature, and an
        /// <see cref="ImmutableArray"/> of <see cref="MulticlassClassificationMetrics"/> objects is returned. See the sample below for an
        /// example of working with these results to analyze the feature importance of a model.
        /// </para>
        /// </remarks>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[PermutationFeatureImportance](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/MulticlassClassification/PermutationFeatureImportance.cs)]
        /// ]]>
        /// </format>
        /// </example>
        /// <param name="catalog">The multiclass classification catalog.</param>
        /// <param name="predictionTransformer">The model on which to evaluate feature importance.</param>
        /// <param name="data">The evaluation data set.</param>
        /// <param name="labelColumnName">Label column name. The column data must be <see cref="KeyDataViewType"/>.</param>
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

        /// <summary>
        /// Permutation Feature Importance (PFI) for MulticlassClassification.
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
        /// the original values for all other features. The evaluation metric (e.g. micro-accuracy) is then calculated
        /// for this modified dataset, and the change in the evaluation metric from the original dataset is computed.
        /// The larger the change in the evaluation metric, the more important the feature is to the model.
        /// PFI works by performing this permutation analysis across all the features of a model, one after another.
        /// </para>
        /// <para>
        /// In this implementation, PFI computes the change in all possible multiclass classification evaluation metrics for each feature, and an
        /// <see cref="ImmutableArray"/> of <see cref="MulticlassClassificationMetrics"/> objects is returned. See the sample below for an
        /// example of working with these results to analyze the feature importance of a model.
        /// </para>
        /// </remarks>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[PermutationFeatureImportance](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/MulticlassClassification/PermutationFeatureImportance.cs)]
        /// ]]>
        /// </format>
        /// </example>
        /// <param name="catalog">The multiclass classification catalog.</param>
        /// <param name="model">The model on which to evaluate feature importance.</param>
        /// <param name="data">The evaluation data set.</param>
        /// <param name="labelColumnName">Label column name. The column data must be <see cref="KeyDataViewType"/>.</param>
        /// <param name="useFeatureWeightFilter">Use features weight to pre-filter features.</param>
        /// <param name="numberOfExamplesToUse">Limit the number of examples to evaluate on. <cref langword="null"/> means up to ~2 bln examples from <paramref param="data"/> will be used.</param>
        /// <param name="permutationCount">The number of permutations to perform.</param>
        /// <returns>Dictionary mapping each feature to its per-feature 'contributions' to the score.</returns>
        public static ImmutableDictionary<string, MulticlassClassificationMetricsStatistics>
            PermutationFeatureImportance(
                this MulticlassClassificationCatalog catalog,
                ITransformer model,
                IDataView data,
                string labelColumnName = DefaultColumnNames.Label,
                bool useFeatureWeightFilter = false,
                int? numberOfExamplesToUse = null,
                int permutationCount = 1)
        {
            Contracts.CheckValue(catalog, nameof(catalog));

            var env = catalog.GetEnvironment();
            Contracts.CheckValue(env, nameof(env));

            env.CheckValue(data, nameof(data));
            env.CheckValue(model, nameof(model));

            MulticlassClassificationMetricsStatistics resultInitializer() => new();
            MulticlassClassificationMetrics evaluationFunc(IDataView idv) => catalog.Evaluate(idv, labelColumnName);

            return PermutationFeatureImportance(
                env,
                model,
                data,
                resultInitializer,
                evaluationFunc,
                MulticlassClassificationDelta,
                permutationCount,
                useFeatureWeightFilter,
                numberOfExamplesToUse
                );
        }

        private static MulticlassClassificationMetrics MulticlassClassificationDelta(
            MulticlassClassificationMetrics a, MulticlassClassificationMetrics b)
        {
            if (a.TopKPredictionCount != b.TopKPredictionCount)
                Contracts.Assert(a.TopKPredictionCount == b.TopKPredictionCount, "TopK to compare must be the same length.");

            var perClassLogLoss = ComputeSequenceDeltas(a.PerClassLogLoss, b.PerClassLogLoss);

            return new MulticlassClassificationMetrics(
                accuracyMicro: a.MicroAccuracy - b.MicroAccuracy,
                accuracyMacro: a.MacroAccuracy - b.MacroAccuracy,
                logLoss: a.LogLoss - b.LogLoss,
                logLossReduction: a.LogLossReduction - b.LogLossReduction,
                topKPredictionCount: a.TopKPredictionCount,
                topKAccuracies: a?.TopKAccuracyForAllK?.Zip(b.TopKAccuracyForAllK, (a, b) => a - b)?.ToArray(),
                perClassLogLoss: perClassLogLoss
                );
        }

        #endregion

        #region Ranking
        /// <summary>
        /// Permutation Feature Importance (PFI) for Ranking.
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
        /// the original values for all other features. The evaluation metric (e.g. NDCG) is then calculated
        /// for this modified dataset, and the change in the evaluation metric from the original dataset is computed.
        /// The larger the change in the evaluation metric, the more important the feature is to the model.
        /// PFI works by performing this permutation analysis across all the features of a model, one after another.
        /// </para>
        /// <para>
        /// In this implementation, PFI computes the change in all possible ranking evaluation metrics for each feature, and an
        /// <see cref="ImmutableArray"/> of <see cref="RankingMetrics"/> objects is returned. See the sample below for an
        /// example of working with these results to analyze the feature importance of a model.
        /// </para>
        /// </remarks>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[PermutationFeatureImportance](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Ranking/PermutationFeatureImportance.cs)]
        /// ]]>
        /// </format>
        /// </example>
        /// <param name="catalog">The ranking catalog.</param>
        /// <param name="predictionTransformer">The model on which to evaluate feature importance.</param>
        /// <param name="data">The evaluation data set.</param>
        /// <param name="labelColumnName">Label column name. The column data must be <see cref="System.Single"/> or <see cref="KeyDataViewType"/>.</param>
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

        /// <summary>
        /// Permutation Feature Importance (PFI) for Ranking.
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
        /// the original values for all other features. The evaluation metric (e.g. NDCG) is then calculated
        /// for this modified dataset, and the change in the evaluation metric from the original dataset is computed.
        /// The larger the change in the evaluation metric, the more important the feature is to the model.
        /// PFI works by performing this permutation analysis across all the features of a model, one after another.
        /// </para>
        /// <para>
        /// In this implementation, PFI computes the change in all possible ranking evaluation metrics for each feature, and an
        /// <see cref="ImmutableArray"/> of <see cref="RankingMetrics"/> objects is returned. See the sample below for an
        /// example of working with these results to analyze the feature importance of a model.
        /// </para>
        /// </remarks>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[PermutationFeatureImportance](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Ranking/PermutationFeatureImportance.cs)]
        /// ]]>
        /// </format>
        /// </example>
        /// <param name="catalog">The ranking catalog.</param>
        /// <param name="model">The model on which to evaluate feature importance.</param>
        /// <param name="data">The evaluation data set.</param>
        /// <param name="labelColumnName">Label column name. The column data must be <see cref="System.Single"/> or <see cref="KeyDataViewType"/>.</param>
        /// <param name="rowGroupColumnName">GroupId column name</param>
        /// <param name="useFeatureWeightFilter">Use features weight to pre-filter features.</param>
        /// <param name="numberOfExamplesToUse">Limit the number of examples to evaluate on. <cref langword="null"/> means up to ~2 bln examples from <paramref param="data"/> will be used.</param>
        /// <param name="permutationCount">The number of permutations to perform.</param>
        /// <returns>Dictionary mapping each feature to its per-feature 'contributions' to the score.</returns>
        public static ImmutableDictionary<string, RankingMetricsStatistics>
            PermutationFeatureImportance(
                this RankingCatalog catalog,
                ITransformer model,
                IDataView data,
                string labelColumnName = DefaultColumnNames.Label,
                string rowGroupColumnName = DefaultColumnNames.GroupId,
                bool useFeatureWeightFilter = false,
                int? numberOfExamplesToUse = null,
                int permutationCount = 1)
        {
            Contracts.CheckValue(catalog, nameof(catalog));

            var env = catalog.GetEnvironment();
            Contracts.CheckValue(env, nameof(env));

            env.CheckValue(data, nameof(data));
            env.CheckValue(model, nameof(model));

            RankingMetricsStatistics resultInitializer() => new();
            RankingMetrics evaluationFunc(IDataView idv) => catalog.Evaluate(idv, labelColumnName, rowGroupColumnName);

            return PermutationFeatureImportance(
                env,
                model,
                data,
                resultInitializer,
                evaluationFunc,
                RankingDelta,
                permutationCount,
                useFeatureWeightFilter,
                numberOfExamplesToUse
                );
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

        private static ImmutableDictionary<string, TResult>
            PermutationFeatureImportance<TMetric, TResult>(
                IHostEnvironment env,
                ITransformer model,
                IDataView data,
                Func<TResult> resultInitializer,
                Func<IDataView, TMetric> evaluationFunc,
                Func<TMetric, TMetric, TMetric> deltaFunc,
                int permutationCount,
                bool useFeatureWeightFilter,
                int? numberOfExamplesToUse) where TResult : IMetricsStatistics<TMetric>
        {
            env.CheckValue(data, nameof(data));
            env.CheckValue(model, nameof(model));

            ISingleFeaturePredictionTransformer lastTransformer = null;

            if (model is ITransformerChainAccessor chain)
            {
                foreach (var transformer in chain.Transformers.Reverse())
                {
                    if (transformer is ISingleFeaturePredictionTransformer singlePredictionTransformer)
                    {
                        lastTransformer = singlePredictionTransformer;
                        break;
                    }
                }
            }
            else lastTransformer = model as ISingleFeaturePredictionTransformer;

            env.CheckValue(lastTransformer, nameof(lastTransformer), "The model provided does not have a compatible predictor");

            string featureColumnName = lastTransformer.FeatureColumnName;
            var predictionTransformerGenericType = GetImplementedIPredictionTransformer(lastTransformer.GetType());

            Type[] types = { predictionTransformerGenericType.GenericTypeArguments[0], typeof(TMetric), typeof(TResult) };
            Type pfiGenericType = typeof(PermutationFeatureImportance<,,>).MakeGenericType(types);

            object[] param = { env,
                lastTransformer,
                data,
                resultInitializer,
                evaluationFunc,
                deltaFunc,
                featureColumnName,
                permutationCount,
                useFeatureWeightFilter,
                numberOfExamplesToUse
            };

            MethodInfo mi = pfiGenericType.GetMethod("GetImportanceMetricsMatrix", BindingFlags.Static | BindingFlags.Public);
            var permutationFeatureImportance = (ImmutableArray<TResult>)mi.Invoke(null, param);

            VBuffer<ReadOnlyMemory<char>> nameBuffer = default;
            data.Schema[featureColumnName].Annotations.GetValue("SlotNames", ref nameBuffer);
            var featureColumnNames = nameBuffer.DenseValues().ToList();

            var output = new Dictionary<string, TResult>();
            for (int i = 0; i < permutationFeatureImportance.Length; i++)
            {
                var name = featureColumnNames[i].ToString();

                // If the slot wasn't given a name, default to just the slot number.
                if (string.IsNullOrEmpty(name))
                {
                    name = $"Slot {i}";
                }
                output.Add(name, permutationFeatureImportance[i]);
            }

            return output.ToImmutableDictionary();
        }

        private static Type GetImplementedIPredictionTransformer(Type type)
        {
            foreach (Type iType in type.GetInterfaces())
            {
                if (iType.IsGenericType && iType.GetGenericTypeDefinition() == typeof(IPredictionTransformer<>))
                {
                    return iType;
                }
            }

            throw new ArgumentException($"Type IPredictionTransformer not implemented by provided type, {type}", nameof(type));
        }

        #endregion
    }
}
