// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.Functional.Tests.Datasets
{
    /// <summary>
    /// A schematized class for loading the HousingRegression dataset.
    /// </summary>
    internal sealed class FeatureContributionOutput
    {
        public float[] FeatureContributions { get; set; }
    }
}
