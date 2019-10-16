// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML.Tests.Datasets
{
    /// <summary>
    /// A class describing the MovieRecommendation test dataset.
    /// </summary>
    internal sealed class MovieRecommendation
    {
        [LoadColumn(0)]
        public uint UserId { get; set; }

        [LoadColumn(1)]
        public uint MovieId { get; set; }

        [LoadColumn(2)]
        public float Rating { get; set; }
    }
}
