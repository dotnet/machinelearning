﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    /// <include file='doc.xml' path='doc/members/member[@name="TensorflowTransformer"]/*' />
    public static class TensorflowCatalog
    {
        /// <summary>
        /// Scores a dataset using a pre-traiend TensorFlow model located in <paramref name="modelLocation"/>.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="modelLocation">Location of the TensorFlow model.</param>
        /// <param name="inputs"> The names of the model inputs.</param>
        /// <param name="outputs">The names of the requested model outputs.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ScoreTensorFlowModel](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/TensorFlowTransform.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static TensorFlowEstimator ScoreTensorFlowModel(this TransformsCatalog catalog,
            string modelLocation,
            string[] inputs,
            string[] outputs)
            => new TensorFlowEstimator(CatalogUtils.GetEnvironment(catalog), modelLocation, inputs, outputs);

        /// <summary>
        /// Scores a dataset using a pre-traiend TensorFlow model specified via <paramref name="tensorFlowModel"/>.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="tensorFlowModel">The pre-trained TensorFlow model.</param>
        /// <param name="inputs"> The names of the model inputs.</param>
        /// <param name="outputs">The names of the requested model outputs.</param>
        public static TensorFlowEstimator ScoreTensorFlowModel(this TransformsCatalog catalog,
            TensorFlowModelInfo tensorFlowModel,
            string[] inputs,
            string[] outputs)
            => new TensorFlowEstimator(CatalogUtils.GetEnvironment(catalog), tensorFlowModel, inputs, outputs);

        /// <summary>
        /// Score or Retrain a tensorflow model (based on setting of the <see cref="TensorFlowTransformer.Arguments.ReTrain"/>) setting.
        /// The model is specified in the <see cref="TensorFlowTransformer.Arguments.ModelLocation"/>.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="args">The <see cref="TensorFlowTransformer.Arguments"/> specifying the inputs and the settings of the <see cref="TensorFlowEstimator"/>.</param>
        public static TensorFlowEstimator TensorFlow(this TransformsCatalog catalog,
            TensorFlowTransformer.Arguments args)
            => new TensorFlowEstimator(CatalogUtils.GetEnvironment(catalog), args);

        /// <summary>
        /// Scores or retrains (based on setting of the <see cref="TensorFlowTransformer.Arguments.ReTrain"/>) a pre-traiend TensorFlow model specified via <paramref name="tensorFlowModel"/>.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="args">The <see cref="TensorFlowTransformer.Arguments"/> specifying the inputs and the settings of the <see cref="TensorFlowEstimator"/>.</param>
        /// <param name="tensorFlowModel">The pre-trained TensorFlow model.</param>
        public static TensorFlowEstimator TensorFlow(this TransformsCatalog catalog,
            TensorFlowTransformer.Arguments args,
            TensorFlowModelInfo tensorFlowModel)
            => new TensorFlowEstimator(CatalogUtils.GetEnvironment(catalog), args, tensorFlowModel);
    }
}
