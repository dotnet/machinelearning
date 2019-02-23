// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
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
            params SimpleColumnInfo[] columns)
            => new MissingValueIndicatorEstimator(CatalogUtils.GetEnvironment(catalog), SimpleColumnInfo.ConvertToValueTuples(columns));

        /// <summary>
        /// Creates a new output column, or replaces the source with a new column
        /// (depending on whether the <paramref name="outputColumnName"/> is given a value, or left to null)
        /// of boolean type, with the same number of slots as the input column. The value in the output column
        /// is true if the value in the input column is missing.
        /// </summary>
        /// <param name="catalog">The transform extensions' catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.
        /// If left to <value>null</value> the <paramref name="inputColumnName"/> will get replaced.</param>
        public static MissingValueIndicatorEstimator IndicateMissingValues(this TransformsCatalog catalog,
            string outputColumnName,
            string inputColumnName = null)
            => new MissingValueIndicatorEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName);

        /// <summary>
        /// Creates a new output column, or replaces the source with a new column
        /// (depending on whether the <paramref name="outputColumnName"/> is given a value, or left to null)
        /// identical to the input column for everything but the missing values. The missing values of the input column, in this new column are replaced with
        /// one of the values specifid in the <paramref name="replacementKind"/>. The default for the <paramref name="replacementKind"/> is
        /// <see cref="MissingValueReplacingEstimator.ColumnInfo.ReplacementMode.DefaultValue"/>.
        /// </summary>
        /// <param name="catalog">The transform extensions' catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.
        /// If not provided, the <paramref name="inputColumnName"/> will be replaced with the results of the transforms.</param>
        /// <param name="replacementKind">The type of replacement to use as specified in <see cref="MissingValueReplacingEstimator.ColumnInfo.ReplacementMode"/></param>
        public static MissingValueReplacingEstimator ReplaceMissingValues(this TransformsCatalog catalog,
            string outputColumnName,
            string inputColumnName = null,
            MissingValueReplacingEstimator.ColumnInfo.ReplacementMode replacementKind = MissingValueReplacingEstimator.Defaults.ReplacementMode)
        => new MissingValueReplacingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName, replacementKind);

        /// <summary>
        /// Creates a new output column, identical to the input column for everything but the missing values.
        /// The missing values of the input column, in this new column are replaced with <see cref="MissingValueReplacingEstimator.ColumnInfo.ReplacementMode.DefaultValue"/>.
        /// </summary>
        /// <param name="catalog">The transform extensions' catalog.</param>
        /// <param name="columns">The name of the columns to use, and per-column transformation configuraiton.</param>
        public static MissingValueReplacingEstimator ReplaceMissingValues(this TransformsCatalog catalog, params MissingValueReplacingEstimator.ColumnInfo[] columns)
            => new MissingValueReplacingEstimator(CatalogUtils.GetEnvironment(catalog), columns);
    }
}
