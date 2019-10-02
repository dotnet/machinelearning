// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.TensorFlow;

namespace Microsoft.ML
{
    /// <include file='doc.xml' path='doc/members/member[@name="TensorflowTransformer"]/*' />
    public static class TensorflowCatalog
    {
        /// <summary>
        /// Load TensorFlow model into memory. This is the convenience method that allows the model to be loaded once and subsequently use it for querying schema and creation of
        /// <see cref="TensorFlowEstimator"/> using <see cref="TensorFlowModel.ScoreTensorFlowModel(string, string, bool)"/>.
        /// usage of this API requires additional NuGet dependencies on TensorFlow redist, see linked document for more information.
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
    }
}
