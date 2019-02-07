// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    public static class OnnxCatalog
    {
        /// <summary>
        /// Applies a pre-trained Onnx model.
        /// </summary>
        /// <remarks>
        /// All column names are provided, the input data column names/types must exactly match
        /// all model input names. All possible output columns are then generated, with names/types
        /// specified by the model.
        /// </remarks>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="modelFile">The path of the file containing the ONNX model.</param>
        /// <param name="gpuDeviceId">Optional GPU device ID to run execution on, <see langword="null" /> to run on CPU.</param>
        /// <param name="fallbackToCpu">If GPU error, raise exception or fallback to CPU.</param>
        public static OnnxScoringEstimator ApplyOnnxModel(this TransformsCatalog catalog,
            string modelFile,
            int? gpuDeviceId = null,
            bool fallbackToCpu = false)
        => new OnnxScoringEstimator(CatalogUtils.GetEnvironment(catalog), modelFile, gpuDeviceId, fallbackToCpu);

        /// <summary>
        /// Applies a pre-trained Onnx model.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="modelFile">The path of the file containing the ONNX model.</param>
        /// <param name="outputColumnName">The output column resulting from the transformation.</param>
        /// <param name="inputColumnName">The input column.</param>
        /// <param name="gpuDeviceId">Optional GPU device ID to run execution on, <see langword="null" /> to run on CPU.</param>
        /// <param name="fallbackToCpu">If GPU error, raise exception or fallback to CPU.</param>
        public static OnnxScoringEstimator ApplyOnnxModel(this TransformsCatalog catalog,
            string modelFile,
            string outputColumnName,
            string inputColumnName,
            int? gpuDeviceId = null,
            bool fallbackToCpu = false)
        => new OnnxScoringEstimator(CatalogUtils.GetEnvironment(catalog), new[] { outputColumnName }, new[] { inputColumnName }, modelFile, gpuDeviceId, fallbackToCpu);

        /// <summary>
        /// Applies a pre-trained Onnx model.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="modelFile">The path of the file containing the ONNX model.</param>
        /// <param name="outputColumnNames">The output columns resulting from the transformation.</param>
        /// <param name="inputColumnNames">The input columns.</param>
        /// <param name="gpuDeviceId">Optional GPU device ID to run execution on, <see langword="null" /> to run on CPU.</param>
        /// <param name="fallbackToCpu">If GPU error, raise exception or fallback to CPU.</param>
        public static OnnxScoringEstimator ApplyOnnxModel(this TransformsCatalog catalog,
            string modelFile,
            string[] outputColumnNames,
            string[] inputColumnNames,
            int? gpuDeviceId = null,
            bool fallbackToCpu = false)
        => new OnnxScoringEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnNames, inputColumnNames, modelFile, gpuDeviceId, fallbackToCpu);

    }
}
