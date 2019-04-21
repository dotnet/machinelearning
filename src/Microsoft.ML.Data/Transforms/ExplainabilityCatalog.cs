﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    using FeatureContributionDefaults = FeatureContributionCalculatingEstimator.Defaults;

    /// <summary>
    /// Collection of extension methods for <see cref="TransformsCatalog"/> to create instances of model explainability components.
    /// </summary>
    public static class ExplainabilityCatalog
    {
        /// <summary>
        /// Create a <see cref="FeatureContributionCalculatingEstimator"/> that computes model-specific contribution scores for
        /// each feature of the input vector.
        /// </summary>
        /// <param name="catalog">The transforms catalog.</param>
        /// <param name="predictionTransformer">A <see cref="ISingleFeaturePredictionTransformer{TModel}"/> that supports Feature Contribution Calculation,
        /// and which will also be used for scoring.</param>
        /// <param name="numberOfPositiveContributions">The number of positive contributions to report, sorted from highest magnitude to lowest magnitude.
        /// Note that if there are fewer features with positive contributions than <paramref name="numberOfPositiveContributions"/>, the rest will be returned as zeros.</param>
        /// <param name="numberOfNegativeContributions">The number of negative contributions to report, sorted from highest magnitude to lowest magnitude.
        /// Note that if there are fewer features with negative contributions than <paramref name="numberOfNegativeContributions"/>, the rest will be returned as zeros.</param>
        /// <param name="normalize">Whether the feature contributions should be normalized to the [-1, 1] interval.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[CalculateFeatureContribution](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/CalculateFeatureContribution.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static FeatureContributionCalculatingEstimator CalculateFeatureContribution(this TransformsCatalog catalog,
            ISingleFeaturePredictionTransformer<ICalculateFeatureContribution> predictionTransformer,
            int numberOfPositiveContributions = FeatureContributionDefaults.NumberOfPositiveContributions,
            int numberOfNegativeContributions = FeatureContributionDefaults.NumberOfNegativeContributions,
            bool normalize = FeatureContributionDefaults.Normalize)
            => new FeatureContributionCalculatingEstimator(CatalogUtils.GetEnvironment(catalog), predictionTransformer.Model, numberOfPositiveContributions, numberOfNegativeContributions, predictionTransformer.FeatureColumnName, normalize);

        /// <summary>
        /// Create a <see cref="FeatureContributionCalculatingEstimator"/> that computes model-specific contribution scores for
        /// each feature of the input vector.
        /// </summary>
        /// <param name="catalog">The transforms catalog.</param>
        /// <param name="predictionTransformer">A <see cref="ISingleFeaturePredictionTransformer{TModel}"/> that supports Feature Contribution Calculation,
        /// and which will also be used for scoring.</param>
        /// <param name="numberOfPositiveContributions">The number of positive contributions to report, sorted from highest magnitude to lowest magnitude.
        /// Note that if there are fewer features with positive contributions than <paramref name="numberOfPositiveContributions"/>, the rest will be returned as zeros.</param>
        /// <param name="numberOfNegativeContributions">The number of negative contributions to report, sorted from highest magnitude to lowest magnitude.
        /// Note that if there are fewer features with negative contributions than <paramref name="numberOfNegativeContributions"/>, the rest will be returned as zeros.</param>
        /// <param name="normalize">Whether the feature contributions should be normalized to the [-1, 1] interval.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[CalculateFeatureContributionCalibrated](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/CalculateFeatureContributionCalibrated.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static FeatureContributionCalculatingEstimator CalculateFeatureContribution<TModelParameters, TCalibrator>(this TransformsCatalog catalog,
            ISingleFeaturePredictionTransformer<CalibratedModelParametersBase<TModelParameters, TCalibrator>> predictionTransformer,
            int numberOfPositiveContributions = FeatureContributionDefaults.NumberOfPositiveContributions,
            int numberOfNegativeContributions = FeatureContributionDefaults.NumberOfNegativeContributions,
            bool normalize = FeatureContributionDefaults.Normalize)
            where TModelParameters : class, ICalculateFeatureContribution
            where TCalibrator : class, ICalibrator
            => new FeatureContributionCalculatingEstimator(CatalogUtils.GetEnvironment(catalog), predictionTransformer.Model.SubModel, numberOfPositiveContributions, numberOfNegativeContributions, predictionTransformer.FeatureColumnName, normalize);
    }
}
