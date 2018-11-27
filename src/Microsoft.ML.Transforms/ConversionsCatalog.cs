// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms.Conversions;

namespace Microsoft.ML
{
    /// <summary>
    /// Extensions for the <see cref="TransformsCatalog.ConversionTransforms"/>.
    /// </summary>
    public static class ConversionsCatalog
    {

        /// <summary>
        ///  Convert the key types back to binary verctor.
        /// </summary>
        /// <param name="catalog">The categorical transform's catalog.</param>
        /// <param name="columns">The input column.</param>
        public static KeyToBinaryVectorMappingEstimator MapKeyToBinaryVector(this TransformsCatalog.ConversionTransforms catalog,
            params KeyToBinaryVectorMappingTransformer.ColumnInfo[] columns)
            => new KeyToBinaryVectorMappingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        ///  Convert the key types back to binary verctor.
        /// </summary>
        /// <param name="catalog">The categorical transform's catalog.</param>
        /// <param name="inputColumn">The name of the input column of the transformation.</param>
        /// <param name="outputColumn">The name of the column produced by the transformation.</param>
        public static KeyToBinaryVectorMappingEstimator MapKeyToBinaryVector(this TransformsCatalog.ConversionTransforms catalog,
            string inputColumn,
            string outputColumn = null)
            => new KeyToBinaryVectorMappingEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn);
    }
}
