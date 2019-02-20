// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.TensorFlow;

namespace Microsoft.ML
{
    /// <include file='doc.xml' path='doc/members/member[@name="TensorflowTransformer"]/*' />
    public static class TensorflowCatalog
    {
        /// <summary>
        /// Scores a dataset using a pre-traiend <a href="https://www.tensorflow.org/">TensorFlow</a> model located in <paramref name="modelLocation"/>.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="modelLocation">Location of the TensorFlow model.</param>
        /// <param name="inputColumnName"> The name of the model input.</param>
        /// <param name="outputColumnName">The name of the requested model output.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ScoreTensorFlowModel](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/TensorFlowTransform.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static TensorFlowEstimator ScoreTensorFlowModel(this TransformsCatalog.TensorFlowTransforms catalog,
            string modelLocation,
            string outputColumnName,
            string inputColumnName)
            => new TensorFlowEstimator(CatalogUtils.GetEnvironment(catalog), new[] { outputColumnName }, new[] { inputColumnName }, modelLocation);

        /// <summary>
        /// Scores a dataset using a pre-traiend TensorFlow model located in <paramref name="modelLocation"/>.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="modelLocation">Location of the TensorFlow model.</param>
        /// <param name="inputColumnNames"> The names of the model inputs.</param>
        /// <param name="outputColumnNames">The names of the requested model outputs.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ScoreTensorFlowModel](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/TensorFlow/ImageClassification.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static TensorFlowEstimator ScoreTensorFlowModel(this TransformsCatalog.TensorFlowTransforms catalog,
            string modelLocation,
            string[] outputColumnNames,
            string[] inputColumnNames)
            => new TensorFlowEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnNames, inputColumnNames, modelLocation);

        /// <summary>
        /// Scores a dataset using a pre-traiend <a href="https://www.tensorflow.org/">TensorFlow</a> model specified via <paramref name="tensorFlowModel"/>.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="tensorFlowModel">The pre-loaded TensorFlow model.</param>
        /// <param name="inputColumnName"> The name of the model input.</param>
        /// <param name="outputColumnName">The name of the requested model output.</param>
        public static TensorFlowEstimator ScoreTensorFlowModel(this TransformsCatalog.TensorFlowTransforms catalog,
            TensorFlowModelInfo tensorFlowModel,
            string outputColumnName,
            string inputColumnName)
            => new TensorFlowEstimator(CatalogUtils.GetEnvironment(catalog), new[] { outputColumnName }, new[] { inputColumnName }, tensorFlowModel);

        /// <summary>
        /// Scores a dataset using a pre-traiend TensorFlow model specified via <paramref name="tensorFlowModel"/>.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="tensorFlowModel">The pre-loaded TensorFlow model.</param>
        /// <param name="inputColumnNames"> The names of the model inputs.</param>
        /// <param name="outputColumnNames">The names of the requested model outputs.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ScoreTensorFlowModel](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/TensorFlow/TextClassification.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static TensorFlowEstimator ScoreTensorFlowModel(this TransformsCatalog.TensorFlowTransforms catalog,
            TensorFlowModelInfo tensorFlowModel,
            string[] outputColumnNames,
            string[] inputColumnNames)
            => new TensorFlowEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnNames, inputColumnNames, tensorFlowModel);

        /// <summary>
        /// Score or Retrain a tensorflow model (based on setting of the <see cref="TensorFlowEstimator.Options.ReTrain"/>) setting.
        /// The model is specified in the <see cref="TensorFlowEstimator.Options.ModelLocation"/>.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="options">The <see cref="TensorFlowEstimator.Options"/> specifying the inputs and the settings of the <see cref="TensorFlowEstimator"/>.</param>
        public static TensorFlowEstimator TensorFlow(this TransformsCatalog.TensorFlowTransforms catalog,
            TensorFlowEstimator.Options options)
            => new TensorFlowEstimator(CatalogUtils.GetEnvironment(catalog), options);

        /// <summary>
        /// Scores or retrains (based on setting of the <see cref="TensorFlowEstimator.Options.ReTrain"/>) a pre-traiend TensorFlow model specified via <paramref name="tensorFlowModel"/>.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="options">The <see cref="TensorFlowEstimator.Options"/> specifying the inputs and the settings of the <see cref="TensorFlowEstimator"/>.</param>
        /// <param name="tensorFlowModel">The pre-loaded TensorFlow model.</param>
        public static TensorFlowEstimator TensorFlow(this TransformsCatalog.TensorFlowTransforms catalog,
            TensorFlowEstimator.Options options,
            TensorFlowModelInfo tensorFlowModel)
            => new TensorFlowEstimator(CatalogUtils.GetEnvironment(catalog), options, tensorFlowModel);

        /// <summary>
        /// This method retrieves the information about the graph nodes of a TensorFlow model as an <see cref="DataViewSchema"/>.
        /// For every node in the graph that has an output type that is compatible with the types supported by
        /// <see cref="TensorFlowEstimator"/>, the output schema contains a column with the name of that node, and the
        /// type of its output (including the item type and the shape, if it is known). Every column also contains metadata
        /// of kind <see cref="TensorFlowUtils.TensorflowOperatorTypeKind"/>, indicating the operation type of the node, and if that node has inputs in the graph,
        /// it contains metadata of kind <see cref="TensorFlowUtils.TensorflowUpstreamOperatorsKind"/>, indicating the names of the input nodes.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="modelLocation">Location of the TensorFlow model.</param>
        public static DataViewSchema GetModelSchema(this TransformsCatalog.TensorFlowTransforms catalog, string modelLocation)
            => TensorFlowUtils.GetModelSchema(CatalogUtils.GetEnvironment(catalog), modelLocation);

        /// <summary>
        /// This is a convenience method for iterating over the nodes of a TensorFlow model graph. It
        /// iterates over the columns of the <see cref="DataViewSchema"/> returned by <see cref="GetModelSchema(TransformsCatalog.TensorFlowTransforms, string)"/>,
        /// and for each one it returns a tuple containing the name, operation type, column type and an array of input node names.
        /// This method is convenient for filtering nodes based on certain criteria, for example, by the operation type.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="modelLocation">Location of the TensorFlow model.</param>
        public static IEnumerable<(string, string, DataViewType, string[])> GetModelNodes(this TransformsCatalog.TensorFlowTransforms catalog, string modelLocation)
            => TensorFlowUtils.GetModelNodes(CatalogUtils.GetEnvironment(catalog), modelLocation);

        /// <summary>
        /// Load TensorFlow model into memory. This is the convenience method that allows the model to be loaded once and subsequently use it for querying schema and creation of
        /// <see cref="TensorFlowEstimator"/> using <see cref="TensorFlow(TransformsCatalog.TensorFlowTransforms, TensorFlowEstimator.Options, TensorFlowModelInfo)"/>.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="modelLocation">Location of the TensorFlow model.</param>
        public static TensorFlowModelInfo LoadTensorFlowModel(this TransformsCatalog.TensorFlowTransforms catalog, string modelLocation)
            => TensorFlowUtils.LoadTensorFlowModel(CatalogUtils.GetEnvironment(catalog), modelLocation);
    }
}
