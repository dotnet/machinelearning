// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    using CountSelectDefaults = CountFeatureSelectingEstimator.Defaults;
    using MutualInfoSelectDefaults = MutualInformationFeatureSelectingEstimator.Defaults;

    /// <summary>
    /// Collection of extension methods for <see cref="TransformsCatalog"/> to create instances of feature
    /// selection transformer components.
    /// </summary>
    public static class FeatureSelectionCatalog
    {
        /// <summary>
        /// Create a <see cref="MutualInformationFeatureSelectingEstimator"/>, which selects the top k slots across all specified columns ordered by their mutual information with the label column.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="slotsInOutput">The maximum number of slots to preserve in the output. The number of slots to preserve is taken across all input columns.</param>
        /// <param name="numberOfBins">Max number of bins used to approximate mutual information between each input column and the label column. Power of 2 recommended.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[SelectFeaturesBasedOnMutualInformation](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/FeatureSelection/SelectFeaturesBasedOnMutualInformation.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static MutualInformationFeatureSelectingEstimator SelectFeaturesBasedOnMutualInformation(this TransformsCatalog.FeatureSelectionTransforms catalog,
            string outputColumnName, string inputColumnName = null,
            string labelColumnName = MutualInfoSelectDefaults.LabelColumnName,
            int slotsInOutput = MutualInfoSelectDefaults.SlotsInOutput,
            int numberOfBins = MutualInfoSelectDefaults.NumBins)
            => new MutualInformationFeatureSelectingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName, labelColumnName, slotsInOutput, numberOfBins);

        /// <summary>
        /// Create a <see cref="MutualInformationFeatureSelectingEstimator"/>, which selects the top k slots across all specified columns ordered by their mutual information with the label column.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">Specifies the names of the input columns for the transformation, and their respective output column names.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="slotsInOutput">The maximum number of slots to preserve in the output. The number of slots to preserve is taken across all input columns.</param>
        /// <param name="numberOfBins">Max number of bins used to approximate mutual information between each input column and the label column. Power of 2 recommended.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[SelectFeaturesBasedOnMutualInformation](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/FeatureSelection/SelectFeaturesBasedOnMutualInformationMultiColumn.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static MutualInformationFeatureSelectingEstimator SelectFeaturesBasedOnMutualInformation(this TransformsCatalog.FeatureSelectionTransforms catalog,
            InputOutputColumnPair[] columns,
            string labelColumnName = MutualInfoSelectDefaults.LabelColumnName,
            int slotsInOutput = MutualInfoSelectDefaults.SlotsInOutput,
            int numberOfBins = MutualInfoSelectDefaults.NumBins)
        {
            var env = CatalogUtils.GetEnvironment(catalog);
            env.CheckValue(columns, nameof(columns));
            return new MutualInformationFeatureSelectingEstimator(env, labelColumnName, slotsInOutput, numberOfBins,
                columns.Select(x => (x.OutputColumnName, x.InputColumnName)).ToArray());
        }

        /// <summary>
        /// Create a <see cref="CountFeatureSelectingEstimator"/>, which selects the slots for which the count of non-default values is greater than or equal to a threshold.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">Describes the parameters of the feature selection process for each column pair.</param>
        [BestFriend]
        internal static CountFeatureSelectingEstimator SelectFeaturesBasedOnCount(this TransformsCatalog.FeatureSelectionTransforms catalog,
            params CountFeatureSelectingEstimator.ColumnOptions[] columns)
            => new CountFeatureSelectingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Create a <see cref="CountFeatureSelectingEstimator"/>, which selects the slots for which the count of non-default values is greater than or equal to a threshold.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// This column's data type will be the same as the input column's data type.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.
        /// This estimator operates over vector or scalar of numeric, text or keys data types.</param>
        /// <param name="count">If the count of non-default values for a slot is greater than or equal to this threshold in the training data, the slot is preserved.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[SelectFeaturesBasedOnCount](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/FeatureSelection/SelectFeaturesBasedOnCount.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static CountFeatureSelectingEstimator SelectFeaturesBasedOnCount(this TransformsCatalog.FeatureSelectionTransforms catalog,
            string outputColumnName,
            string inputColumnName = null,
            long count = CountSelectDefaults.Count)
            => new CountFeatureSelectingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName, count);

        /// <summary>
        /// Create a <see cref="CountFeatureSelectingEstimator"/>, which selects the slots for which the count of non-default values is greater than or equal to a threshold.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">Specifies the names of the columns on which to apply the transformation.
        /// This estimator operates over vector or scalar of numeric, text or keys data types.
        /// The output columns' data types will be the same as the input columns' data types.</param>
        /// <param name="count">If the count of non-default values for a slot is greater than or equal to this threshold in the training data, the slot is preserved.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[SelectFeaturesBasedOnCount](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/FeatureSelection/SelectFeaturesBasedOnCountMultiColumn.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static CountFeatureSelectingEstimator SelectFeaturesBasedOnCount(this TransformsCatalog.FeatureSelectionTransforms catalog,
            InputOutputColumnPair[] columns,
            long count = CountSelectDefaults.Count)
        {
            var env = CatalogUtils.GetEnvironment(catalog);
            env.CheckValue(columns, nameof(columns));
            var columnOptions = columns.Select(x => new CountFeatureSelectingEstimator.ColumnOptions(x.OutputColumnName, x.InputColumnName, count)).ToArray();
            return new CountFeatureSelectingEstimator(env, columnOptions);
        }
    }
}
