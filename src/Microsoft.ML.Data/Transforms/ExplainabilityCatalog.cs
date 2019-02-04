// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Internal.Internallearn;

namespace Microsoft.ML
{
    using FeatureContributionDefaults = FeatureContributionCalculatingEstimator.Defaults;

    public static class ExplainabilityCatalog
    {
        /// <summary>
        /// Feature Contribution Calculation computes model-specific contribution scores for each feature.
        /// Note that this functionality is not supported by all the models. See <see cref="FeatureContributionCalculatingTransformer"/> for a list of the suported models.
        /// </summary>
        /// <param name="catalog">The model explainability operations catalog.</param>
        /// <param name="modelParameters">Trained model parameters that support Feature Contribution Calculation and which will be used for scoring.</param>
        /// <param name="featureColumn">The name of the feature column that will be used as input.</param>
        /// <param name="numPositiveContributions">The number of positive contributions to report, sorted from highest magnitude to lowest magnitude.
        /// Note that if there are fewer features with positive contributions than <paramref name="numPositiveContributions"/>, the rest will be returned as zeros.</param>
        /// <param name="numNegativeContributions">The number of negative contributions to report, sorted from highest magnitude to lowest magnitude.
        /// Note that if there are fewer features with negative contributions than <paramref name="numNegativeContributions"/>, the rest will be returned as zeros.</param>
        /// <param name="normalize">Whether the feature contributions should be normalized to the [-1, 1] interval.</param>
        public static FeatureContributionCalculatingEstimator FeatureContributionCalculation(this ModelOperationsCatalog.ExplainabilityTransforms catalog,
            ICalculateFeatureContribution modelParameters,
            string featureColumn = DefaultColumnNames.Features,
            int numPositiveContributions = FeatureContributionDefaults.NumPositiveContributions,
            int numNegativeContributions = FeatureContributionDefaults.NumNegativeContributions,
            bool normalize = FeatureContributionDefaults.Normalize)
            => new FeatureContributionCalculatingEstimator(CatalogUtils.GetEnvironment(catalog), modelParameters, featureColumn, numPositiveContributions, numNegativeContributions, normalize);
    }
}
