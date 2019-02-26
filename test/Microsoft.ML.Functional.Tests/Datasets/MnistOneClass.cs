// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.Functional.Tests.Datasets
{
    internal sealed class MnistOneClass
    {
        private const int _featureLength = 783;

        public float Label { get; set; }

        public float[] Features { get; set; }

        public static TextLoader GetTextLoader(MLContext mlContext, bool hasHeader, char separatorChar)
        {
            return mlContext.Data.CreateTextLoader(
                    new[] {
                        new TextLoader.Column("Label", DataKind.Single, 0),
                        new TextLoader.Column("Features", DataKind.Single, 1, 1 + _featureLength)
                    },
                separatorChar: separatorChar,
                hasHeader: hasHeader,
                allowSparse: true);
        }
    }
}
