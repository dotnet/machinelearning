// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.Trainers
{
    /// <summary>
    /// Interface for mapping input values to corresponding feature contributions.
    /// This interface is commonly implemented by predictors.
    /// </summary>
    [BestFriend]
    internal interface IFeatureContributionMapper : IPredictor
    {
        /// <summary>
        /// Get a delegate for mapping Contributions to Features.
        /// Result will contain vector with topN positive contributions(if available) and
        /// bottomN negative contributions (if available).
        /// For example linear predictor will have both negative and positive contributions.
        /// For trees we will not have negative contributions, so bottom param will be ignored.
        /// If normalization is requested that resulting values will be normalized to [-1, 1].
        /// </summary>
        ValueMapper<TSrc, VBuffer<float>> GetFeatureContributionMapper<TSrc, TDst>(int top, int bottom, bool normalize);
    }

    /// <summary>
    /// Allows support for feature contribution calculation by model parameters.
    /// </summary>
    public interface ICalculateFeatureContribution
    {
        FeatureContributionCalculator FeatureContributionCalculator { get; }
    }

    /// <summary>
    /// Support for feature contribution calculation.
    /// </summary>
    public sealed class FeatureContributionCalculator
    {
        internal IFeatureContributionMapper ContributionMapper { get; }
        internal FeatureContributionCalculator(IFeatureContributionMapper contributionMapper) => ContributionMapper = contributionMapper;
    }
}
