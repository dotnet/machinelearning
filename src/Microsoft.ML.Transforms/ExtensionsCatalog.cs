// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    /// <summary>
    /// Collection of extension methods for <see cref="TransformsCatalog"/> to create instances of
    /// missing value transformer components.
    /// </summary>
    public static class ExtensionsCatalog
    {
        /// <summary>
        /// Create a <see cref="MissingValueIndicatorEstimator"/>, which scans the data from the column specified in <paramref name="inputColumnName"/>
        /// and fills new column specified in <paramref name="outputColumnName"/> with vector of bools where i-th bool has value of <see langword="true"/>
        /// if i-th element in column data has missing value and <see langword="false"/> otherwise.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// This column's data type will be a vector of <see cref="System.Boolean"/>.</param>
        /// <param name="inputColumnName">Name of the column to copy the data from.
        /// This estimator operates over scalar or vector of <see cref="System.Single"/> or <see cref="System.Double"/>.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[MissingValueIndicator](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/IndicateMissingValues.cs)]
        /// ]]></format>
        /// </example>
        public static MissingValueIndicatorEstimator IndicateMissingValues(this TransformsCatalog catalog,
            string outputColumnName,
            string inputColumnName = null)
            => new MissingValueIndicatorEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName);

        /// <summary>
        /// Create a <see cref="MissingValueIndicatorEstimator"/>, which copies the data from the column specified in <see cref="InputOutputColumnPair.InputColumnName" />
        /// to a new column: <see cref="InputOutputColumnPair.OutputColumnName" />.
        /// </summary>
        /// <remarks>This transform can operate over several columns.</remarks>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">The pairs of input and output columns. This estimator operates over data which is either scalar or vector of <see cref="System.Single"/> or <see cref="System.Double"/>.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[MissingValueIndicator](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/IndicateMissingValuesMultiColumn.cs)]
        /// ]]></format>
        /// </example>
        public static MissingValueIndicatorEstimator IndicateMissingValues(this TransformsCatalog catalog, InputOutputColumnPair[] columns)
        {
            var env = CatalogUtils.GetEnvironment(catalog);
            env.CheckValue(columns, nameof(columns));
            return new MissingValueIndicatorEstimator(env, columns.Select(x => (x.OutputColumnName, x.InputColumnName)).ToArray());
        }

        /// <summary>
        /// Create a <see cref="MissingValueReplacingEstimator"/>, which copies the data from the column specified in <paramref name="inputColumnName"/>
        /// to a new column: <paramref name="outputColumnName"/> and replaces missing values in it according to <paramref name="replacementMode"/>.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// This column's data type will be the same as that of the input column.</param>
        /// <param name="inputColumnName">Name of the column to copy the data from.
        /// This estimator operates over scalar or vector of <see cref="System.Single"/> or <see cref="System.Double"/>.</param>
        /// <param name="replacementMode">The type of replacement to use as specified in <see cref="MissingValueReplacingEstimator.ReplacementMode"/></param>
        /// <param name="imputeBySlot">If true, per-slot imputation of replacement is performed.
        /// Otherwise, replacement value is imputed for the entire vector column. This setting is ignored for scalars and variable vectors,
        /// where imputation is always for the entire column.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[MissingValuesReplace](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/ReplaceMissingValues.cs)]
        /// ]]></format>
        /// </example>
        public static MissingValueReplacingEstimator ReplaceMissingValues(this TransformsCatalog catalog,
            string outputColumnName,
            string inputColumnName = null,
            MissingValueReplacingEstimator.ReplacementMode replacementMode = MissingValueReplacingEstimator.Defaults.Mode,
            bool imputeBySlot = MissingValueReplacingEstimator.Defaults.ImputeBySlot)
        => new MissingValueReplacingEstimator(CatalogUtils.GetEnvironment(catalog), new[] { new MissingValueReplacingEstimator.ColumnOptions(outputColumnName, inputColumnName, replacementMode, imputeBySlot) });

        /// <summary>
        /// Create a <see cref="ColumnCopyingEstimator"/>, which copies the data from the column specified in <see cref="InputOutputColumnPair.InputColumnName" />
        /// to a new column: <see cref="InputOutputColumnPair.OutputColumnName" /> and replaces missing values in it according to <paramref name="replacementMode"/>.
        /// </summary>
        /// <remarks>This transform can operate over several columns.</remarks>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">The pairs of input and output columns. This estimator operates over scalar or vector of floats or doubles.</param>
        /// <param name="replacementMode">The type of replacement to use as specified in <see cref="MissingValueReplacingEstimator.ReplacementMode"/></param>
        /// <param name="imputeBySlot">If <see langword="true"/>, per-slot imputation of replacement is performed.
        /// Otherwise, replacement value is imputed for the entire vector column. This setting is ignored for scalars and variable vectors,
        /// where imputation is always for the entire column.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[MissingValuesReplace](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/ReplaceMissingValuesMultiColumn.cs)]
        /// ]]></format>
        /// </example>
        public static MissingValueReplacingEstimator ReplaceMissingValues(this TransformsCatalog catalog,
            InputOutputColumnPair[] columns,
            MissingValueReplacingEstimator.ReplacementMode replacementMode = MissingValueReplacingEstimator.Defaults.Mode,
            bool imputeBySlot = MissingValueReplacingEstimator.Defaults.ImputeBySlot)
        {
            var env = CatalogUtils.GetEnvironment(catalog);
            env.CheckValue(columns, nameof(columns));
            var columnOptions = columns.Select(x => new MissingValueReplacingEstimator.ColumnOptions(x.OutputColumnName, x.InputColumnName, replacementMode, imputeBySlot)).ToArray();
            return new MissingValueReplacingEstimator(env, columnOptions);
        }

        /// <summary>
        /// Creates a new output column, identical to the input column for everything but the missing values.
        /// The missing values of the input column, in this new column are replaced with <see cref="MissingValueReplacingEstimator.ReplacementMode.DefaultValue"/>.
        /// </summary>
        /// <param name="catalog">The transform extensions' catalog.</param>
        /// <param name="columns">The name of the columns to use, and per-column transformation configuraiton.</param>
        [BestFriend]
        internal static MissingValueReplacingEstimator ReplaceMissingValues(this TransformsCatalog catalog, params MissingValueReplacingEstimator.ColumnOptions[] columns)
            => new MissingValueReplacingEstimator(CatalogUtils.GetEnvironment(catalog), columns);
    }
}
