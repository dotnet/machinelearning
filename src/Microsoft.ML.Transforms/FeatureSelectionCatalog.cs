// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    using CountSelectDefaults = CountFeatureSelectingEstimator.Defaults;
    using MutualInfoSelectDefaults = MutualInformationFeatureSelectingEstimator.Defaults;

    public static class FeatureSelectionCatalog
    {
        /// <include file='doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="slotsInOutput">The maximum number of slots to preserve in the output. The number of slots to preserve is taken across all input columns.</param>
        /// <param name="numberOfBins">Max number of bins used to approximate mutual information between each input column and the label column. Power of 2 recommended.</param>
        /// <param name="columns">Specifies the names of the input columns for the transformation, and their respective output column names.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[SelectFeaturesBasedOnMutualInformation](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/FeatureSelectionTransform.cs?range=1-4,10-121)]
        /// ]]>
        /// </format>
        /// </example>
        [BestFriend]
        internal static MutualInformationFeatureSelectingEstimator SelectFeaturesBasedOnMutualInformation(this TransformsCatalog.FeatureSelectionTransforms catalog,
            string labelColumnName = MutualInfoSelectDefaults.LabelColumn,
            int slotsInOutput = MutualInfoSelectDefaults.SlotsInOutput,
            int numberOfBins = MutualInfoSelectDefaults.NumBins,
            params ColumnOptions[] columns)
            => new MutualInformationFeatureSelectingEstimator(CatalogUtils.GetEnvironment(catalog), labelColumnName, slotsInOutput, numberOfBins,
                ColumnOptions.ConvertToValueTuples(columns));

        /// <include file='doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="slotsInOutput">The maximum number of slots to preserve in the output. The number of slots to preserve is taken across all input columns.</param>
        /// <param name="numberOfBins">Max number of bins used to approximate mutual information between each input column and the label column. Power of 2 recommended.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[SelectFeaturesBasedOnMutualInformation](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/FeatureSelectionTransform.cs?range=1-4,10-121)]
        /// ]]>
        /// </format>
        /// </example>
        public static MutualInformationFeatureSelectingEstimator SelectFeaturesBasedOnMutualInformation(this TransformsCatalog.FeatureSelectionTransforms catalog,
            string outputColumnName, string inputColumnName = null,
            string labelColumnName = MutualInfoSelectDefaults.LabelColumn,
            int slotsInOutput = MutualInfoSelectDefaults.SlotsInOutput,
            int numberOfBins = MutualInfoSelectDefaults.NumBins)
            => new MutualInformationFeatureSelectingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName, labelColumnName, slotsInOutput, numberOfBins);

        /// <include file='doc.xml' path='doc/members/member[@name="CountFeatureSelection"]' />
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">Describes the parameters of the feature selection process for each column pair.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[SelectFeaturesBasedOnCount](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/FeatureSelectionTransform.cs?range=1-4,10-121)]
        /// ]]>
        /// </format>
        /// </example>
        [BestFriend]
        internal static CountFeatureSelectingEstimator SelectFeaturesBasedOnCount(this TransformsCatalog.FeatureSelectionTransforms catalog,
            params CountFeatureSelectingEstimator.ColumnOptions[] columns)
            => new CountFeatureSelectingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <include file='doc.xml' path='doc/members/member[@name="CountFeatureSelection"]' />
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="count">If the count of non-default values for a slot is greater than or equal to this threshold in the training data, the slot is preserved.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[SelectFeaturesBasedOnCount](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/FeatureSelectionTransform.cs?range=1-4,10-121)]
        /// ]]>
        /// </format>
        /// </example>
        public static CountFeatureSelectingEstimator SelectFeaturesBasedOnCount(this TransformsCatalog.FeatureSelectionTransforms catalog,
            string outputColumnName,
            string inputColumnName = null,
            long count = CountSelectDefaults.Count)
            => new CountFeatureSelectingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName, count);
    }
}
