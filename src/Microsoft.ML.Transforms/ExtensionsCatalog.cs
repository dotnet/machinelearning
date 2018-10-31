// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Conversions;

namespace Microsoft.ML
{
    public static class MissingValueIndicatorCatalog
    {
        /// <summary>
        /// Initializes a new instance of <see cref="MissingValueIndicatorEstimator"/>
        /// </summary>
        /// <param name="catalog">The categorical transform's catalog.</param>
        /// <param name="columns">The names of the input columns of the transformation and the corresponding names for the output columns.</param>
        public static MissingValueIndicatorEstimator IndicateMissingValues(this TransformsCatalog catalog,
            params (string inputColumn, string outputColumn)[] columns)
            => new MissingValueIndicatorEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Initializes a new instance of <see cref="MissingValueIndicatorEstimator"/>
        /// </summary>
        /// <param name="catalog">The categorical transform's catalog.</param>
        /// <param name="inputColumn">The name of the input column of the transformation.</param>
        /// <param name="outputColumn">The name of the column produced by the transformation.</param>
        public static MissingValueIndicatorEstimator IndicateMissingValues(this TransformsCatalog catalog,
            string inputColumn,
            string outputColumn = null)
            => new MissingValueIndicatorEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn);
    }

    /// <summary>
    /// Extensions for KeyToVectorMappingEstimator.
    /// </summary>
    public static class ToBinaryVectorCatalog
    {
        /// <summary>
        ///  Convert the key types back to binary verctor.
        /// </summary>
        /// <param name="catalog">The categorical transform's catalog.</param>
        /// <param name="columns">The input column.</param>
        /// <returns></returns>
        public static KeyToBinaryVectorMappingEstimator MapKeyToBinaryVector(this TransformsCatalog.CategoricalTransforms catalog,
            params KeyToBinaryVectorTransform.ColumnInfo[] columns)
        => new KeyToBinaryVectorMappingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        ///  Convert the key types back to binary verctor.
        /// </summary>
        /// <param name="catalog">The categorical transform's catalog.</param>
        /// <param name="inputColumn">The name of the input column of the transformation.</param>
        /// <param name="outputColumn">The name of the column produced by the transformation.</param>
        /// <returns></returns>
        public static KeyToBinaryVectorMappingEstimator MapKeyToBinaryVector(this TransformsCatalog.CategoricalTransforms catalog,
            string inputColumn,
            string outputColumn = null)
            => new KeyToBinaryVectorMappingEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn);
    }
}
