// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms.FeatureSelection;

namespace Microsoft.ML
{
    using MutualInfoSelectDefaults = MutualInformationFeatureSelectingEstimator.Defaults;
    using CountSelectDefaults = CountFeatureSelectingEstimator.Defaults;

    public static class FeatureSelectionCatalog
    {
        /// <include file='doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="labelColumn">Name of the column to use for labels.</param>
        /// <param name="slotsInOutput">The maximum number of slots to preserve in the output. The number of slots to preserve is taken across all input columns.</param>
        /// <param name="numBins">Max number of bins used to approximate mutual information between each input column and the label column. Power of 2 recommended.</param>
        /// <param name="columns">Specifies the names of the input columns for the transformation, and their respective output column names.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[SelectFeaturesBasedOnMutualInformation](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/FeatureSelectionTransform.cs?range=1-4,10-121)]
        /// ]]>
        /// </format>
        /// </example>
        public static MutualInformationFeatureSelectingEstimator SelectFeaturesBasedOnMutualInformation(this TransformsCatalog.FeatureSelectionTransforms catalog,
            string labelColumn = MutualInfoSelectDefaults.LabelColumn,
            int slotsInOutput = MutualInfoSelectDefaults.SlotsInOutput,
            int numBins = MutualInfoSelectDefaults.NumBins,
            params (string input, string output)[] columns)
            => new MutualInformationFeatureSelectingEstimator(CatalogUtils.GetEnvironment(catalog), labelColumn, slotsInOutput, numBins, columns);

        /// <include file='doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="inputColumn">Name of the input column.</param>
        /// <param name="outputColumn">Name of the column resulting from the transformation of <paramref name="inputColumn"/>. Null means <paramref name="inputColumn"/> is replaced. </param>
        /// <param name="labelColumn">Name of the column to use for labels.</param>
        /// <param name="slotsInOutput">The maximum number of slots to preserve in the output. The number of slots to preserve is taken across all input columns.</param>
        /// <param name="numBins">Max number of bins used to approximate mutual information between each input column and the label column. Power of 2 recommended.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[SelectFeaturesBasedOnMutualInformation](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/FeatureSelectionTransform.cs?range=1-4,10-121)]
        /// ]]>
        /// </format>
        /// </example>
        public static MutualInformationFeatureSelectingEstimator SelectFeaturesBasedOnMutualInformation(this TransformsCatalog.FeatureSelectionTransforms catalog,
            string inputColumn, string outputColumn = null,
            string labelColumn = MutualInfoSelectDefaults.LabelColumn,
            int slotsInOutput = MutualInfoSelectDefaults.SlotsInOutput,
            int numBins = MutualInfoSelectDefaults.NumBins)
            => new MutualInformationFeatureSelectingEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, labelColumn, slotsInOutput, numBins);

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
        public static CountFeatureSelectingEstimator SelectFeaturesBasedOnCount(this TransformsCatalog.FeatureSelectionTransforms catalog,
            params CountFeatureSelectingEstimator.ColumnInfo[] columns)
            => new CountFeatureSelectingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <include file='doc.xml' path='doc/members/member[@name="CountFeatureSelection"]' />
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="inputColumn">Name of the input column.</param>
        /// <param name="outputColumn">Name of the column resulting from the transformation of <paramref name="inputColumn"/>. Null means <paramref name="inputColumn"/> is replaced. </param>
        /// <param name="count">If the count of non-default values for a slot is greater than or equal to this threshold in the training data, the slot is preserved.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[SelectFeaturesBasedOnCount](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/FeatureSelectionTransform.cs?range=1-4,10-121)]
        /// ]]>
        /// </format>
        /// </example>
        public static CountFeatureSelectingEstimator SelectFeaturesBasedOnCount(this TransformsCatalog.FeatureSelectionTransforms catalog,
            string inputColumn,
            string outputColumn = null,
            long count = CountSelectDefaults.Count)
            => new CountFeatureSelectingEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, count);
    }
}
