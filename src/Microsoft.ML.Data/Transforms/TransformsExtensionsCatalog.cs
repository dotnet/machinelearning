using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML
{
    /// <summary>
    /// Extensions for Column Copying Estiamtor.
    /// </summary>
    public static class ColumnCopyingCatalog
    {
        /// <summary>
        /// Copies the input column to another column named as specified in <paramref name="outputColumn"/>.
        /// </summary>
        /// <param name="catalog"></param>
        /// <param name="inputColumn">The source column.</param>
        /// <param name="outputColumn">The new column, resulting from copying.</param>
        public static CopyColumnsEstimator CopyColumns(this TransformsCatalog catalog, string inputColumn, string outputColumn)
            => new CopyColumnsEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn);

        /// <summary>
        /// Copies the input column, name specified in the first item of the tuple,
        /// to another column, named as specified in the second item of the tuple.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="columns">The pairs of input and output columns.</param>
        public static CopyColumnsEstimator CopyColumns(this TransformsCatalog catalog, params (string source, string name)[] columns)
            => new CopyColumnsEstimator(CatalogUtils.GetEnvironment(catalog), columns);

    }

    /// <summary>
    /// Extension ColumnConcatenatingEstimator
    /// </summary>
    public static class ColumnConcatenatingEstimator
    {
        /// <summary>
        /// Concatenates two columns together.
        /// </summary>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="inputColumns">The names of the columns to concatenate together.</param>
        /// <param name="outputColumn">The name of the output column.</param>
        public static ConcatEstimator Concatenate(this TransformsCatalog catalog, string outputColumn, params string[] inputColumns)
            => new ConcatEstimator(CatalogUtils.GetEnvironment(catalog), outputColumn, inputColumns);

    }
}
