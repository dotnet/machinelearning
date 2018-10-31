// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms.Categorical;

namespace Microsoft.ML
{

    /// <summary>
    /// Extensions for KeyToVectorMappingEstimator.
    /// </summary>
    public static class ToBinaryVectorCatalog
    {
        public static KeyToBinaryVectorMappingEstimator ToBinaryVector(this TransformsCatalog.CategoricalTransforms catalog, params KeyToBinaryVectorTransform.ColumnInfo[] columns)
        => new KeyToBinaryVectorMappingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        public static KeyToBinaryVectorMappingEstimator ToBinaryVector(this TransformsCatalog.CategoricalTransforms catalog, string name, string source = null)
            => new KeyToBinaryVectorMappingEstimator(CatalogUtils.GetEnvironment(catalog), name, source);
    }
}
