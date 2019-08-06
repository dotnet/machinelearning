// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Onnx;

namespace Microsoft.ML
{
    public static class OnnxCatalog
    {
        /// <summary>
        /// Create a <see cref="OnnxScoringEstimator"/>, which applies a pre-trained Onnx model to the input column.
        /// Input/output columns are determined based on the input/output columns of the provided ONNX model.
        /// </summary>
        /// <remarks>
        /// The name/type of input columns must exactly match name/type of the ONNX model inputs.
        /// The name/type of the produced output columns will match name/type of the ONNX model outputs.
        /// </remarks>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="modelFile">The path of the file containing the ONNX model.</param>
        /// <param name="gpuDeviceId">Optional GPU device ID to run execution on, <see langword="null" /> to run on CPU.</param>
        /// <param name="fallbackToCpu">If GPU error, raise exception or fallback to CPU.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ApplyOnnxModel](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/ApplyOnnxModel.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static OnnxScoringEstimator ApplyOnnxModel(this TransformsCatalog catalog,
            string modelFile,
            int? gpuDeviceId = null,
            bool fallbackToCpu = false)
        => new OnnxScoringEstimator(CatalogUtils.GetEnvironment(catalog), modelFile, gpuDeviceId, fallbackToCpu);

        /// <summary>
        /// Create a <see cref="OnnxScoringEstimator"/>, which applies a pre-trained Onnx model to the input column.
        /// Input/output columns are determined based on the input/output columns of the provided ONNX model.
        /// </summary>
        /// <remarks>
        /// The name/type of input columns must exactly match name/type of the ONNX model inputs.
        /// The name/type of the produced output columns will match name/type of the ONNX model outputs.
        /// </remarks>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="modelFile">The path of the file containing the ONNX model.</param>
        /// <param name="shapeDictionary">ONNX shape should be used to over those loaded from <paramref name="modelFile"/>.</param>
        /// <param name="gpuDeviceId">Optional GPU device ID to run execution on, <see langword="null" /> to run on CPU.</param>
        /// <param name="fallbackToCpu">If GPU error, raise exception or fallback to CPU.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ApplyOnnxModel](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/ApplyOnnxModel.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static OnnxScoringEstimator ApplyOnnxModel(this TransformsCatalog catalog,
            string modelFile,
            IDictionary<string, int[]> shapeDictionary,
            int? gpuDeviceId = null,
            bool fallbackToCpu = false)
        => new OnnxScoringEstimator(CatalogUtils.GetEnvironment(catalog), modelFile, gpuDeviceId, fallbackToCpu,
            shapeDictionary: shapeDictionary);

        /// <summary>
        /// Create a <see cref="OnnxScoringEstimator"/>, which applies a pre-trained Onnx model to the <paramref name="inputColumnName"/> column.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">The output column resulting from the transformation.</param>
        /// <param name="inputColumnName">The input column.</param>
        /// <param name="modelFile">The path of the file containing the ONNX model.</param>
        /// <param name="gpuDeviceId">Optional GPU device ID to run execution on, <see langword="null" /> to run on CPU.</param>
        /// <param name="fallbackToCpu">If GPU error, raise exception or fallback to CPU.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ApplyOnnxModel](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/ApplyONNXModelWithInMemoryImages.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static OnnxScoringEstimator ApplyOnnxModel(this TransformsCatalog catalog,
            string outputColumnName,
            string inputColumnName,
            string modelFile,
            int? gpuDeviceId = null,
            bool fallbackToCpu = false)
        => new OnnxScoringEstimator(CatalogUtils.GetEnvironment(catalog), new[] { outputColumnName }, new[] { inputColumnName },
            modelFile, gpuDeviceId, fallbackToCpu);

        /// <summary>
        /// Create a <see cref="OnnxScoringEstimator"/>, which applies a pre-trained Onnx model to the <paramref name="inputColumnName"/> column.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">The output column resulting from the transformation.</param>
        /// <param name="inputColumnName">The input column.</param>
        /// <param name="modelFile">The path of the file containing the ONNX model.</param>
        /// <param name="shapeDictionary">ONNX shape should be used to over those loaded from <paramref name="modelFile"/>.</param>
        /// <param name="gpuDeviceId">Optional GPU device ID to run execution on, <see langword="null" /> to run on CPU.</param>
        /// <param name="fallbackToCpu">If GPU error, raise exception or fallback to CPU.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ApplyOnnxModel](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/ApplyONNXModelWithInMemoryImages.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static OnnxScoringEstimator ApplyOnnxModel(this TransformsCatalog catalog,
            string outputColumnName,
            string inputColumnName,
            string modelFile,
            IDictionary<string, int[]> shapeDictionary,
            int? gpuDeviceId = null,
            bool fallbackToCpu = false)
        => new OnnxScoringEstimator(CatalogUtils.GetEnvironment(catalog), new[] { outputColumnName }, new[] { inputColumnName },
            modelFile, gpuDeviceId, fallbackToCpu, shapeDictionary: shapeDictionary);

        /// <summary>
        /// Create a <see cref="OnnxScoringEstimator"/>, which applies a pre-trained Onnx model to the <paramref name="inputColumnNames"/> columns.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnNames">The output columns resulting from the transformation.</param>
        /// <param name="inputColumnNames">The input columns.</param>
        /// <param name="modelFile">The path of the file containing the ONNX model.</param>
        /// <param name="gpuDeviceId">Optional GPU device ID to run execution on, <see langword="null" /> to run on CPU.</param>
        /// <param name="fallbackToCpu">If GPU error, raise exception or fallback to CPU.</param>
        public static OnnxScoringEstimator ApplyOnnxModel(this TransformsCatalog catalog,
            string[] outputColumnNames,
            string[] inputColumnNames,
            string modelFile,
            int? gpuDeviceId = null,
            bool fallbackToCpu = false)
        => new OnnxScoringEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnNames, inputColumnNames,
            modelFile, gpuDeviceId, fallbackToCpu);

        /// <summary>
        /// Create a <see cref="OnnxScoringEstimator"/>, which applies a pre-trained Onnx model to the <paramref name="inputColumnNames"/> columns.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnNames">The output columns resulting from the transformation.</param>
        /// <param name="inputColumnNames">The input columns.</param>
        /// <param name="modelFile">The path of the file containing the ONNX model.</param>
        /// <param name="shapeDictionary">ONNX shape should be used to over those loaded from <paramref name="modelFile"/>.</param>
        /// <param name="gpuDeviceId">Optional GPU device ID to run execution on, <see langword="null" /> to run on CPU.</param>
        /// <param name="fallbackToCpu">If GPU error, raise exception or fallback to CPU.</param>
        public static OnnxScoringEstimator ApplyOnnxModel(this TransformsCatalog catalog,
            string[] outputColumnNames,
            string[] inputColumnNames,
            string modelFile,
            IDictionary<string, int[]> shapeDictionary,
            int? gpuDeviceId = null,
            bool fallbackToCpu = false)
        => new OnnxScoringEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnNames, inputColumnNames,
            modelFile, gpuDeviceId, fallbackToCpu, shapeDictionary: shapeDictionary);

        /// <summary>
        /// Create <see cref="DnnImageFeaturizerEstimator"/>, which applies one of the pre-trained DNN models in
        /// <see cref="DnnImageModelSelector"/> to featurize an image.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">The name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="modelFactory">An extension method on the <see cref="DnnImageModelSelector"/> that creates a chain of two
        /// <see cref="OnnxScoringEstimator"/> (one for preprocessing and one with a pretrained image DNN) with specific models
        /// included in a package together with that extension method.</param>
        /// <param name="inputColumnName">Name of column to transform.
        /// If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[DnnFeaturizeImage](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/ImageAnalytics/DnnFeaturizeImage.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static DnnImageFeaturizerEstimator DnnFeaturizeImage(this TransformsCatalog catalog,
            string outputColumnName,
            Func<DnnImageFeaturizerInput, EstimatorChain<ColumnCopyingTransformer>> modelFactory,
            string inputColumnName = null)
        => new DnnImageFeaturizerEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, modelFactory, inputColumnName);
    }
}
