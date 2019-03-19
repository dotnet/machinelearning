// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Functional.Tests.Datasets
{
    /// <summary>
    /// A class to hold a feature column.
    /// </summary>
    internal sealed class FeatureColumn
    {
        public float[] Features { get; set; }
    }

    /// <summary>
    /// A class to hold the output of FeatureContributionCalculator.
    /// </summary>
    internal sealed class FeatureContributionOutput
    {
        public float[] FeatureContributions { get; set; }
    }

    /// <summary>
    /// A class to hold the Score column.
    /// </summary>
    internal sealed class ScoreColumn
    {
        public float Score { get; set; }
    }

    /// <summary>
    /// A class to hold a vector Score column.
    /// </summary>
    internal sealed class VectorScoreColumn
    {
        public float[] Score { get; set; }
    }
}
