// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.Functional.Tests.Datasets
{
    /// <summary>
    /// A class to hold the output of FeatureContributionCalculator
    /// </summary>
    internal sealed class FeatureContributionOutput
    {
        public float[] FeatureContributions { get; set; }
    }
}
