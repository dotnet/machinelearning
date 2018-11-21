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
    /// Extensions for ColumnConcatenatingEstimator.
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

    /// <summary>
    /// Extensions for ColumnSelectingEstimator.
    /// </summary>
    public static class ColumnSelectingCatalog
    {
        /// <summary>
        /// KeepColumns is used to select a list of columns that the user wants to keep on a given an input. Any column not specified
        /// will be dropped from the output output schema.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columnsToKeep">The array of column names to keep.</param>
        public static ColumnSelectingEstimator KeepColumns(this TransformsCatalog catalog, params string[] columnsToKeep)
            => ColumnSelectingEstimator.KeepColumns(CatalogUtils.GetEnvironment(catalog), columnsToKeep);

        /// <summary>
        /// DropColumns is used to select a list of columns that user wants to drop from a given input. Any column not specified will
        /// be maintained in the output schema.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columnsToDrop">The array of column names to drop.</param>
        public static ColumnSelectingEstimator DropColumns(this TransformsCatalog catalog, params string[] columnsToDrop)
            => ColumnSelectingEstimator.DropColumns(CatalogUtils.GetEnvironment(catalog), columnsToDrop);

        /// <summary>
        /// ColumnSelectingEstimator is used to select a list of columns that user wants to drop from a given input.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="keepColumns">The array of column names to keep, cannot be set with <paramref name="dropColumns"/>.</param>
        /// <param name="dropColumns">The array of column names to drop, cannot be set with <paramref name="keepColumns"/>.</param>
        /// <param name="keepHidden">If true will keep hidden columns and false will remove hidden columns.</param>
        /// <param name="ignoreMissing">If false will check for any columns given in <paramref name="keepColumns"/>
        /// or <paramref name="dropColumns"/> that are missing from the input. If a missing colums exists a
        /// SchemaMistmatch exception is thrown. If true, the check is not made.</param>
        public static ColumnSelectingEstimator SelectColumns(this TransformsCatalog catalog,
            string[] keepColumns,
            string[] dropColumns,
            bool keepHidden = SelectColumnsTransform.Defaults.KeepHidden,
            bool ignoreMissing = SelectColumnsTransform.Defaults.IgnoreMissing)
            => new ColumnSelectingEstimator(CatalogUtils.GetEnvironment(catalog),
                keepColumns, dropColumns, keepHidden, ignoreMissing);
    }
}
