// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.Functional.Tests.Datasets
{
    /// <summary>
    /// A class containing one property per <see cref="DataKind"/>.
    /// </summary>
    /// <remarks>
    /// This class has annotations for automatic deserialization from a file, and contains helper methods
    /// for reading from a file and for generating a random dataset as an IEnumerable.
    /// </remarks>
    internal sealed class MnistOneClass
    {
        [LoadColumn(0)]
        public float Label { get; set; }

        [LoadColumn(1, 784), VectorType(784)]
        public float[] Features { get; set; }
    }
}
