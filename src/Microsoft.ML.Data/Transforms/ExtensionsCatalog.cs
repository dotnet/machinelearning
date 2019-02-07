// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    /// <summary>
    /// Extension methods for the <see cref="TransformsCatalog"/>.
    /// </summary>
    public static class TransformExtensionsCatalog
    {
        /// <summary>
        /// Copies the input column to another column named as specified in <paramref name="outputColumnName"/>.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the columns to transform.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[CopyColumns](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/CopyColumns.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static ColumnCopyingEstimator CopyColumns(this TransformsCatalog catalog, string outputColumnName, string inputColumnName)
            => new ColumnCopyingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName);

        /// <summary>
        /// Copies the input column, name specified in the first item of the tuple,
        /// to another column, named as specified in the second item of the tuple.
        /// </summary>
        /// <param name="catalog">The transform's catalog</param>
        /// <param name="columns">The pairs of input and output columns.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[CopyColumns](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/CopyColumns.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static ColumnCopyingEstimator CopyColumns(this TransformsCatalog catalog, params (string outputColumnName, string inputColumnName)[] columns)
            => new ColumnCopyingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Concatenates columns together.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnNames"/>.</param>
        /// <param name="inputColumnNames">Name of the columns to transform.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[Concat](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Concatenate.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static ColumnConcatenatingEstimator Concatenate(this TransformsCatalog catalog, string outputColumnName, params string[] inputColumnNames)
            => new ColumnConcatenatingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnNames);

        /// <summary>
        /// DropColumns is used to select a list of columns that user wants to drop from a given input. Any column not specified will
        /// be maintained in the output schema.
        /// </summary>
        /// <remarks>
        /// <see cref="DropColumns"/> is commonly used to remove unwanted columns from the schema if the dataset is going to be serialized or
        /// written out to a file. It is not actually necessary to drop unused columns before training or
        /// performing transforms, as <see cref="IDataView"/>'s lazy evaluation won't actually materialize those columns.
        /// In the case of serialization, every column in the schema will be written out. If you have columns
        /// that you don't want to save, you can use <see cref="DropColumns"/> to remove them from the schema.
        /// </remarks>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columnsToDrop">The array of column names to drop.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[Concat](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/DropColumns.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static ColumnSelectingEstimator DropColumns(this TransformsCatalog catalog, params string[] columnsToDrop)
            => ColumnSelectingEstimator.DropColumns(CatalogUtils.GetEnvironment(catalog), columnsToDrop);

        /// <summary>
        /// Select a list of columns to keep in a given <see cref="IDataView"/>.
        /// </summary>
        /// <remarks>
        /// <format type="text/markdown">
        /// <see cref="SelectColumns(TransformsCatalog, string[], bool)"/> operates on the schema of an input <see cref="IDataView"/>,
        /// either dropping unselected columns from the schema or keeping them but marking them as hidden in the schema. Keeping columns hidden
        /// is recommended when it is necessary to understand how the inputs of a pipeline map to outputs of the pipeline. This feature
        /// is useful, for example, in debugging a pipeline of transforms by allowing you to print out results from the middle of the pipeline.
        /// For more information on hidden columns, please refer to [IDataView Design Principles](~/../docs/samples/docs/code/IDataViewDesignPrinciples.md).
        /// </format>
        /// </remarks>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="keepColumns">The array of column names to keep.</param>
        /// <param name="keepHidden">If <see langword="true"/> will keep hidden columns and <see langword="false"/> will remove hidden columns.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[SelectColumns](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/SelectColumns.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static ColumnSelectingEstimator SelectColumns(this TransformsCatalog catalog,
            string[] keepColumns,
            bool keepHidden)
            => new ColumnSelectingEstimator(CatalogUtils.GetEnvironment(catalog),
                keepColumns, null, keepHidden, ColumnSelectingEstimator.Defaults.IgnoreMissing);

        /// <summary>
        /// Select a list of columns to keep in a given <see cref="IDataView"/>.
        /// </summary>
        /// <remarks>
        /// <format type="text/markdown"><![CDATA[
        /// <xref:Microsoft.ML.SelectColumns(Microsoft.ML.TransformsCatalog, string[])> operates on the schema of an input <xref:Microsoft.Data.DataView.IDataView>,
        /// dropping unselected columns from the schema.
        /// ]]></format>
        /// </remarks>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="keepColumns">The array of column names to keep.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[SelectColumns](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/SelectColumns.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static ColumnSelectingEstimator SelectColumns(this TransformsCatalog catalog,
            params string[] keepColumns) => catalog.SelectColumns(keepColumns, false);
    }
}
