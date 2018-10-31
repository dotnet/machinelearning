// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms.Categorical;

namespace Microsoft.ML
{

    public static class ToValueCatalog
    {
        /// <summary>
        /// Convert the key types back to their original values.
        /// </summary>
        /// <param name="catalog">The categorical transform's catalog.</param>
        /// <param name="inputColumn">Name of the input column.</param>
        public static KeyToValueEstimator ToValue(this TransformsCatalog.CategoricalTransforms catalog, string inputColumn)
            => new KeyToValueEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn);

        /// <summary>
        ///  Convert the key types (name of the column specified in the first item of the tuple) back to their original values
        ///  (named as specified in the second item of the tuple).
        /// </summary>
        /// <param name="catalog">The categorical transform's catalog</param>
        /// <param name="columns">The pairs of input and output columns.</param>
        public static KeyToValueEstimator ToValue(this TransformsCatalog.CategoricalTransforms catalog, params (string input, string output)[] columns)
             => new KeyToValueEstimator(CatalogUtils.GetEnvironment(catalog), columns);
    }

    /// <summary>
    /// Extensions for KeyToVectorMappingEstimator.
    /// </summary>
    public static class ToVectorCatalog
    {
        /// <summary>
        /// Convert the key types back to their original vectors.
        /// </summary>
        /// <param name="catalog">The categorical transform's catalog.</param>
        /// <param name="columns">The input column to map back to vectors.</param>
        public static KeyToVectorMappingEstimator ToVector(this TransformsCatalog.CategoricalTransforms catalog,
            params KeyToVectorTransform.ColumnInfo[] columns)
            => new KeyToVectorMappingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Convert the key types back to their original vectors.
        /// </summary>
        /// <param name="catalog">The categorical transform's catalog.</param>
        /// <param name="inputColumn">The name of the input column.</param>
        /// <param name="outputColumn">The name of the output column.</param>
        /// <param name="bag">Whether bagging is used for the conversion. </param>
        public static KeyToVectorMappingEstimator ToVector(this TransformsCatalog.CategoricalTransforms catalog,
            string inputColumn, string outputColumn = null, bool bag = KeyToVectorMappingEstimator.Defaults.Bag)
            => new KeyToVectorMappingEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, bag);
    }
}
