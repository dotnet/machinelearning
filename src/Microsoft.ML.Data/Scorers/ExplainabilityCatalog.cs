// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Internallearn;

namespace Microsoft.ML
{
    using FeatureContributionDefaults = FeatureContributionCalculatingEstimator.Defaults;

    public static class ExplainabilityCatalog
    {
        /// <summary>
        /// Feature Contribution Calculation scores the model on an input dataset and
        /// computes model-specific contribution scores for each feature. Note that this functionality is not supported by all the predictos.
        /// See <see cref="FeatureContributionCalculatingTransformer"/> for a list of the suported predictors.
        /// </summary>
        /// <param name="catalog">The model explainability operations catalog.</param>
        /// <param name="predictor">Trained model parameters that support Feature Contribution Calculation and which will be used for scoring.</param>
        /// <param name="featureColumn">The name of the feature column that will be used as input.</param>
        /// <param name="top">The number of top contributing features for each data sample that will be retained in the FeatureContribution column.</param>
        /// <param name="bottom">The number of least contributing features for each data sample that will be retained in the FeatureContribution column.</param>
        /// <param name="normalize">Whether the feature contributions should be normalized to the [-1, 1] interval.</param>
        /// <param name="stringify">Since the features are converted to numbers before the algorithms use them, if you want the contributions presented as
        /// "feature name:feature contribution" pairs for each feature, set stringify to <langword>true</langword></param>
        public static FeatureContributionCalculatingEstimator FeatureContributionCalculation(this ModelOperationsCatalog.ExplainabilityTransforms catalog,
            IFeatureContributionMappable predictor,
            string featureColumn = DefaultColumnNames.Features,
            int top = FeatureContributionDefaults.Top,
            int bottom = FeatureContributionDefaults.Bottom,
            bool normalize = FeatureContributionDefaults.Normalize,
            bool stringify = FeatureContributionDefaults.Stringify)
            => new FeatureContributionCalculatingEstimator(CatalogUtils.GetEnvironment(catalog), predictor, featureColumn, top, bottom, normalize, stringify);
    }
}
