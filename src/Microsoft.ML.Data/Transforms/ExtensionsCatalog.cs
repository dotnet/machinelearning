// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    /// <summary>
    /// Specifies input and output column names for transformer components that operate on multiple columns.
    /// </summary>
    /// <remarks>
    /// It is often advantageous to transform several columns at once as all of the changes can be done in a
    /// single data pass.
    /// </remarks>
    public sealed class InputOutputColumnPair
    {
        /// <summary>
        /// Name of the column to transform. If set to <see langword="null"/>, the value of the <see cref="OutputColumnName"/> will be used as source.
        /// </summary>
        public string InputColumnName { get; }
        /// <summary>
        /// Name of the column resulting from the transformation of <see cref="InputColumnName"/>.
        /// </summary>
        public string OutputColumnName { get; }

        /// <summary>
        /// Specifies input and output column names for a transformation.
        /// </summary>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        public InputOutputColumnPair(string outputColumnName, string inputColumnName = null)
        {
            Contracts.CheckNonEmpty(outputColumnName, nameof(outputColumnName));
            InputColumnName = inputColumnName ?? outputColumnName;
            OutputColumnName = outputColumnName;
        }

        [BestFriend]
        internal static (string outputColumnName, string inputColumnName)[] ConvertToValueTuples(InputOutputColumnPair[] infos) =>
            infos.Select(info => (info.OutputColumnName, info.InputColumnName)).ToArray();

        [BestFriend]
        internal static IReadOnlyList<InputOutputColumnPair> ConvertFromValueTuples((string outputColumnName, string inputColumnName)[] infos) =>
            infos.Select(info => new InputOutputColumnPair(info.outputColumnName, info.inputColumnName)).ToList().AsReadOnly();
    }

    /// <summary>
    /// Collection of extension methods for <see cref="TransformsCatalog"/> to create instances of transform components
    /// that manipulate columns.
    /// </summary>
    public static class TransformExtensionsCatalog
    {
        /// <summary>
        /// Create a <see cref="ColumnCopyingEstimator"/>, which copies the data from the column specified in <paramref name="inputColumnName"/>
        /// to a new column: <paramref name="outputColumnName"/>.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// This column's data type will be the same as that of the input column.</param>
        /// <param name="inputColumnName">Name of the column to copy the data from.
        /// This estimator operates over any data type.</param>
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
        /// Create a <see cref="ColumnCopyingEstimator"/>, which copies the data from the column specified in <see cref="InputOutputColumnPair.InputColumnName" />
        /// to a new column: <see cref="InputOutputColumnPair.OutputColumnName" />.
        /// </summary>
        /// <remarks>This transform can operate over several columns.</remarks>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">The pairs of input and output columns. This estimator operates over any data type.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[CopyColumns](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/CopyColumns.cs)]
        /// ]]>
        /// </format>
        /// </example>
        [BestFriend]
        internal static ColumnCopyingEstimator CopyColumns(this TransformsCatalog catalog, params InputOutputColumnPair[] columns)
        {
            var env = CatalogUtils.GetEnvironment(catalog);
            env.CheckValue(columns, nameof(columns));
            return new ColumnCopyingEstimator(env, InputOutputColumnPair.ConvertToValueTuples(columns));
        }

        /// <summary>
        /// Create a <see cref="ColumnConcatenatingEstimator"/>, which concatenates one or more input columns into a new output column.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnNames"/>.
        /// This column's data type will be a vector of the input columns' data type.</param>
        /// <param name="inputColumnNames">Name of the columns to concatenate.
        /// This estimator operates over any data type except key type.
        /// If more that one column is provided, they must all have the same data type.</param>
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
        /// Create a <see cref="ColumnSelectingEstimator"/>, which drops a given list of columns from an <see cref="IDataView"/>. Any column not specified will
        /// be maintained in the output.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columnNames">The array of column names to drop.
        /// This estimator operates over columns of any data type.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[Concat](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/DropColumns.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static ColumnSelectingEstimator DropColumns(this TransformsCatalog catalog, params string[] columnNames)
            => ColumnSelectingEstimator.DropColumns(CatalogUtils.GetEnvironment(catalog), columnNames);

        /// <summary>
        /// Create a <see cref="ColumnSelectingEstimator"/>, which keeps a given list of columns in an <see cref="IDataView"/> and drops the others.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columnNames">The array of column names to keep.</param>
        /// <param name="keepHidden">If <see langword="true"/> will keep hidden columns and <see langword="false"/> will remove hidden columns.
        /// Keeping hidden columns, instead of dropping them, is recommended when it is necessary to understand how the inputs of a pipeline
        /// map to outputs of the pipeline, for debugging purposes.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[SelectColumns](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/SelectColumns.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static ColumnSelectingEstimator SelectColumns(this TransformsCatalog catalog,
            string[] columnNames,
            bool keepHidden)
            => new ColumnSelectingEstimator(CatalogUtils.GetEnvironment(catalog),
                columnNames, null, keepHidden, ColumnSelectingEstimator.Defaults.IgnoreMissing);

        /// <summary>
        /// Create a <see cref="ColumnSelectingEstimator"/>, which keeps a given list of columns in an <see cref="IDataView"/> and drops the others.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columnNames">The array of column names to keep.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[SelectColumns](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/SelectColumns.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static ColumnSelectingEstimator SelectColumns(this TransformsCatalog catalog,
            params string[] columnNames) => catalog.SelectColumns(columnNames, false);
    }
}
