// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    /// <summary>
    /// Extensions for Column Copying Estimator.
    /// </summary>
    public static class ColumnCopyingCatalog
    {
        /// <summary>
        /// Copies the input column to another column named as specified in <paramref name="outputColumn"/>.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="inputColumn">Name of the input column.</param>
        /// <param name="outputColumn">Name of the new column, resulting from copying.</param>
        public static CopyColumnsEstimator CopyColumns(this TransformsCatalog catalog, string inputColumn, string outputColumn)
            => new CopyColumnsEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn);

        /// <summary>
        /// Copies the input column, name specified in the first item of the tuple,
        /// to another column, named as specified in the second item of the tuple.
        /// </summary>
        /// <param name="catalog">The transform's catalog</param>
        /// <param name="columns">The pairs of input and output columns.</param>
        public static CopyColumnsEstimator CopyColumns(this TransformsCatalog catalog, params (string source, string name)[] columns)
            => new CopyColumnsEstimator(CatalogUtils.GetEnvironment(catalog), columns);

    }

    /// <summary>
    /// Extension ColumnConcatenatingEstimator
    /// </summary>
    public static class ColumnConcatenatingEstimatorCatalog
    {
        /// <summary>
        /// Concatenates two columns together.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumn">The name of the output column.</param>
        /// <param name="inputColumns">The names of the columns to concatenate together.</param>
        public static ColumnConcatenatingEstimator Concatenate(this TransformsCatalog catalog, string outputColumn, params string[] inputColumns)
            => new ColumnConcatenatingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumn, inputColumns);

    }
}
