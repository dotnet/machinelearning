// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.TensorFlow;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    /// <include file='doc.xml' path='doc/members/member[@name="TensorflowTransformer"]/*' />
    public static class TensorflowCatalog
    {
        /// <summary>
        /// Load TensorFlow model into memory. This is the convenience method that allows the model to be loaded once and subsequently use it for querying schema and creation of
        /// <see cref="TensorFlowEstimator"/> using <see cref="TensorFlowModel.ScoreTensorFlowModel(string, string, bool)"/>.
        /// usage of this API requires additional NuGet dependencies on TensorFlow redist, see linked document for more information.
        /// <see cref="TensorFlowModel"/> also holds references to unmanaged resources that need to be freed either with an explicit
        /// call to Dispose() or implicitly by declaring the variable with the "using" syntax/>
        ///
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!include[io](~/../docs/samples/docs/api-reference/tensorflow-usage.md)]
        /// ]]>
        /// </format>
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="modelLocation">Location of the TensorFlow model.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[LoadTensorFlowModel](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/TensorFlow/TextClassification.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static TensorFlowModel LoadTensorFlowModel(this ModelOperationsCatalog catalog, string modelLocation)
            => TensorFlowUtils.LoadTensorFlowModel(CatalogUtils.GetEnvironment(catalog), modelLocation);

        /// <summary>
        /// Load TensorFlow model into memory. This is the convenience method that allows the model to be loaded once and subsequently use it for querying schema and creation of
        /// <see cref="TensorFlowEstimator"/> using <see cref="TensorFlowModel.ScoreTensorFlowModel(string, string, bool)"/>.
        /// usage of this API requires additional NuGet dependencies on TensorFlow redist, see linked document for more information.
        /// <see cref="TensorFlowModel"/> also holds references to unmanaged resources that need to be freed either with an explicit
        /// call to Dispose() or implicitly by declaring the variable with the "using" syntax/>
        ///
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!include[io](~/../docs/samples/docs/api-reference/tensorflow-usage.md)]
        /// ]]>
        /// </format>
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="modelLocation">Location of the TensorFlow model.</param>
        /// <param name="treatOutputAsBatched">If the first dimension of the output is unknown, should it be treated as batched or not.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[LoadTensorFlowModel](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/TensorFlow/TextClassification.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static TensorFlowModel LoadTensorFlowModel(this ModelOperationsCatalog catalog, string modelLocation, bool treatOutputAsBatched)
            => TensorFlowUtils.LoadTensorFlowModel(CatalogUtils.GetEnvironment(catalog), modelLocation, treatOutputAsBatched);
    }
}
