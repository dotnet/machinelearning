// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    public static class ExtensionsCatalog
    {
        /// <summary>
        /// Creates a new output column, of boolean type, with the same number of slots as the input column. The value in the output column
        /// is true if the value in the input column is missing.
        /// </summary>
        /// <param name="catalog">The transform extensions' catalog.</param>
        /// <param name="columns">The names of the input columns of the transformation and the corresponding names for the output columns.</param>
        public static MissingValueIndicatorEstimator IndicateMissingValues(this TransformsCatalog catalog,
            params (string inputColumn, string outputColumn)[] columns)
            => new MissingValueIndicatorEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Creates a new output column, or replaces the inputColumn with a new column
        /// (depending on whether the <paramref name="outputColumn"/> is given a value, or left to null)
        /// of boolean type, with the same number of slots as the input column. The value in the output column
        /// is true if the value in the input column is missing.
        /// </summary>
        /// <param name="catalog">The transform extensions' catalog.</param>
        /// <param name="inputColumn">The name of the input column of the transformation.</param>
        /// <param name="outputColumn">The name of the optional column produced by the transformation.
        /// If left to <value>null</value> the <paramref name="inputColumn"/> will get replaced.</param>
        public static MissingValueIndicatorEstimator IndicateMissingValues(this TransformsCatalog catalog,
            string inputColumn,
            string outputColumn = null)
            => new MissingValueIndicatorEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn);

        /// <summary>
        /// Creates a new output column, or replaces the inputColumn with a new column
        /// (depending on whether the <paramref name="outputColumn"/> is given a value, or left to null)
        /// identical to the input column for everything but the missing values. The missing values of the input column, in this new column are replaced with
        /// one of the values specifid in the <paramref name="replacementKind"/>. The default for the <paramref name="replacementKind"/> is
        /// <see cref="MissingValueReplacingTransformer.ColumnInfo.ReplacementMode.DefaultValue"/>.
        /// </summary>
        /// <param name="catalog">The transform extensions' catalog.</param>
        /// <param name="inputColumn">The name of the input column.</param>
        /// <param name="outputColumn">The optional name of the output column,
        /// If not provided, the <paramref name="inputColumn"/> will be replaced with the results of the transforms.</param>
        /// <param name="replacementKind">The type of replacement to use as specified in <see cref="MissingValueReplacingTransformer.ColumnInfo.ReplacementMode"/></param>
        public static MissingValueReplacingEstimator ReplaceMissingValues(this TransformsCatalog catalog,
            string inputColumn,
            string outputColumn = null,
            MissingValueReplacingTransformer.ColumnInfo.ReplacementMode replacementKind = MissingValueReplacingEstimator.Defaults.ReplacementMode)
        => new MissingValueReplacingEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, replacementKind);

        /// <summary>
        /// Creates a new output column, identical to the input column for everything but the missing values.
        /// The missing values of the input column, in this new column are replaced with <see cref="MissingValueReplacingTransformer.ColumnInfo.ReplacementMode.DefaultValue"/>.
        /// </summary>
        /// <param name="catalog">The transform extensions' catalog.</param>
        /// <param name="columns">The name of the columns to use, and per-column transformation configuraiton.</param>
        public static MissingValueReplacingEstimator ReplaceMissingValues(this TransformsCatalog catalog, params MissingValueReplacingTransformer.ColumnInfo[] columns)
            => new MissingValueReplacingEstimator(CatalogUtils.GetEnvironment(catalog), columns);
    }
}
