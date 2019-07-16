// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.TransferLearning;

namespace Microsoft.ML
{
    public static class TransferLearningCatalog
    {
        /// <summary>
        /// Load TransferLearning model into memory. This is the convenience method that allows the model to be loaded once and subsequently use it for querying schema and creation of
        /// <see cref="TransferLearningEstimator"/> using <see cref="TransferLearningModel.ScoreTransferLearningModel(string, string, bool)"/>.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[LoadTransferLearningModel](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/TransferLearning/TextClassification.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static TransferLearningModel LoadTransferLearningModel(this ModelOperationsCatalog catalog)
            => TransferLearning.LoadTransferLearningModel(CatalogUtils.GetEnvironment(catalog));
    }
}
