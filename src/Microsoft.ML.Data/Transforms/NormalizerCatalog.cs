// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML
{
    /// <summary>
    /// Extensions for normalizer operations.
    /// </summary>
    public static class NormalizerCatalog
    {
        /// <summary>
        /// Normalize (rescale) the column according to the specified <paramref name="mode"/>.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="inputName">The column name</param>
        /// <param name="outputName">The column name</param>
        /// <param name="mode">The <see cref="NormalizerEstimator.NormalizerMode"/> used to map the old values in the new scale. </param>
        public static NormalizerEstimator Normalizer(this TransformsCatalog catalog, string inputName, string outputName = null, NormalizerEstimator.NormalizerMode mode = Runtime.Data.NormalizerEstimator.NormalizerMode.MinMax)
            => new NormalizerEstimator(CatalogUtils.GetEnvironment(catalog), inputName, outputName, mode);

        /// <summary>
        /// Normalize (rescale) several columns according to the specified <paramref name="mode"/>.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="mode">The <see cref="NormalizerEstimator.NormalizerMode"/> used to map the old values to the new ones. </param>
        /// <param name="columns">The pairs of input and output columns.</param>
        public static NormalizerEstimator Normalizer(this TransformsCatalog catalog, NormalizerEstimator.NormalizerMode mode, params (string input, string output)[] columns)
            => new NormalizerEstimator(CatalogUtils.GetEnvironment(catalog), mode, columns);

        /// <summary>
        /// Normalize (rescale) columns according to specified custom parameters.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="columns">The normalization settings for all the columns</param>
        public static NormalizerEstimator Normalizer(this TransformsCatalog catalog, params NormalizerEstimator.ColumnBase[] columns)
            => new NormalizerEstimator(CatalogUtils.GetEnvironment(catalog), columns);
    }
}
