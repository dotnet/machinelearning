// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    public static class OnnxCatalog
    {
        /// <summary>
        /// Initializes a new instance of <see cref="OnnxScoringEstimator"/>.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="modelFile">The path of the file containing the ONNX model.</param>
        /// <param name="inputColumns">The input columns.</param>
        /// <param name="outputColumns">The output columns resulting from the transformation.</param>
        public static OnnxScoringEstimator ApplyOnnxModel(this TransformsCatalog catalog,
            string modelFile,
            string[] inputColumns,
            string[] outputColumns)
        => new OnnxScoringEstimator(CatalogUtils.GetEnvironment(catalog), modelFile, inputColumns, outputColumns);

        /// <summary>
        /// Initializes a new instance of <see cref="OnnxScoringEstimator"/>.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="transformer">The ONNX transformer.</param>
        public static OnnxScoringEstimator ApplyOnnxModel(this TransformsCatalog catalog, OnnxTransform transformer)
            => new OnnxScoringEstimator(CatalogUtils.GetEnvironment(catalog), transformer);
    }
}
