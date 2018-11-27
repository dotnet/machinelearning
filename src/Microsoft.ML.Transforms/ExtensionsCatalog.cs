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

        /// <include file='doc.xml' path='doc/members/member[@name="CountFeatureSelection"]' />
        /// <param name="catalog">The categorical transform's catalog.</param>
        /// <param name="inputColumn">The input column to apply feature selection on.</param>
        /// <param name="outputColumn">The output column. Null means <paramref name="inputColumn"/> is used.</param>
        /// <param name="count">If the count of non-default values for a slot is greater than or equal to this threshold, the slot is preserved.</param>
        public static CountFeatureSelector CountFeatures(this TransformsCatalog catalog,
            string inputColumn,
            string outputColumn = null,
            long count = CountFeatureSelectingTransformer.Defaults.Count)
            => new CountFeatureSelector(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, count);

        /// <include file='doc.xml' path='doc/members/member[@name="CountFeatureSelection"]' />
        /// <param name="catalog">The transform extensions' catalog.</param>
        /// <param name="count">If the count of non-default values for a slot is greater than or equal to this threshold, the slot is preserved.</param>
        /// <param name="columns">Columns to use for feature selection.</param>
        public static CountFeatureSelector CountFeatures(this TransformsCatalog catalog,
            (string input, string output)[] columns,
            long count = CountFeatureSelectingTransformer.Defaults.Count)
            => new CountFeatureSelector(CatalogUtils.GetEnvironment(catalog), columns, count);

        /// <include file='doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
        /// <param name="catalog">The transform extensions' catalog.</param>
        /// <param name="inputColumn">The input column to apply feature selection on.</param>
        /// <param name="outputColumn">The output column. Null means <paramref name="inputColumn"/> is used.</param>
        /// <param name="labelColumn">Column to use for labels.</param>
        /// <param name="slotsInOutput">The maximum number of slots to preserve in output.</param>
        /// <param name="numBins">Max number of bins for float/double columns, power of 2 recommended.</param>
        public static MutualInformationFeatureSelector SelectFeaturesWithMutualInformation(this TransformsCatalog catalog,
            string inputColumn,
            string outputColumn = null,
            string labelColumn = DefaultColumnNames.Label,
            int slotsInOutput = MutualInformationFeatureSelectionTransform.Defaults.SlotsInOutput,
            int numBins = MutualInformationFeatureSelectionTransform.Defaults.NumBins)
            => new MutualInformationFeatureSelector(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, labelColumn, slotsInOutput, numBins);

        /// <include file='doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
        /// <param name="catalog">The transform extensions' catalog.</param>
        /// <param name="columns">Columns to use for feature selection.</param>
        /// <param name="labelColumn">Column to use for labels.</param>
        /// <param name="slotsInOutput">The maximum number of slots to preserve in output.</param>
        /// <param name="numBins">Max number of bins for float/double columns, power of 2 recommended.</param>
        public static MutualInformationFeatureSelector SelectFeaturesWithMutualInformation(this TransformsCatalog catalog,
            (string input, string output)[] columns,
            string labelColumn = DefaultColumnNames.Label,
            int slotsInOutput = MutualInformationFeatureSelectionTransform.Defaults.SlotsInOutput,
            int numBins = MutualInformationFeatureSelectionTransform.Defaults.NumBins)
            => new MutualInformationFeatureSelector(CatalogUtils.GetEnvironment(catalog), columns, labelColumn, slotsInOutput, numBins);
    }
}
