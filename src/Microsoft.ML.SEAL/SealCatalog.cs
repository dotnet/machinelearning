// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.SEAL
{
    public static class SealCatalog
    {
        public static SealEstimator EncryptFeatures(this TransformsCatalog catalog,
            bool encrypt,
            double scale,
            ulong polyModDegree,
            string sealKeyFilePath,
            IEnumerable<int> bitSizes,
            string outputColumnName,
            string inputColumnName = null)
            => new SealEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                encrypt, scale, polyModDegree, sealKeyFilePath, bitSizes, outputColumnName, inputColumnName);
    }
}
