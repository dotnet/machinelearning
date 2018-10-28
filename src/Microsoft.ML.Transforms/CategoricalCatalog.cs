// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;

namespace Microsoft.ML
{
    /// <summary>
    /// Static extensions for categorical transforms.
    /// </summary>
    public static class CategoricalCatalog
    {
        /// <summary>
        /// Convert a text column into one-hot encoded vector.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="inputColumn">The input column</param>
        /// <param name="outputColumn">The output column. If <c>null</c>, <paramref name="inputColumn"/> is used.</param>
        /// <param name="outputKind">The conversion mode.</param>
        /// <returns></returns>
        public static CategoricalEstimator OneHotEncoding(this TransformsCatalog.CategoricalTransforms catalog,
                string inputColumn, string outputColumn = null, CategoricalTransform.OutputKind outputKind = CategoricalTransform.OutputKind.Ind)
            => new CategoricalEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, outputKind);

        /// <summary>
        /// Convert several text column into one-hot encoded vectors.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="columns">The column settings.</param>
        /// <returns></returns>
        public static CategoricalEstimator OneHotEncoding(this TransformsCatalog.CategoricalTransforms catalog,
                params CategoricalEstimator.ColumnInfo[] columns)
            => new CategoricalEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Convert a text column into hash-based one-hot encoded vector.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="inputColumn">The input column</param>
        /// <param name="outputColumn">The output column. If <c>null</c>, <paramref name="inputColumn"/> is used.</param>
        /// <param name="outputKind">The conversion mode.</param>
        /// <returns></returns>
        public static CategoricalHashEstimator OneHotHashEncoding(this TransformsCatalog.CategoricalTransforms catalog,
                string inputColumn, string outputColumn = null, CategoricalTransform.OutputKind outputKind = CategoricalTransform.OutputKind.Ind)
            => new CategoricalHashEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, outputKind);

        /// <summary>
        /// Convert several text column into hash-based one-hot encoded vectors.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="columns">The column settings.</param>
        /// <returns></returns>
        public static CategoricalHashEstimator OneHotHashEncoding(this TransformsCatalog.CategoricalTransforms catalog,
                params CategoricalHashEstimator.ColumnInfo[] columns)
            => new CategoricalHashEstimator(CatalogUtils.GetEnvironment(catalog), columns);
    }
}
