// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.Transforms.Onnx
{
    /// <summary>
    /// The options for an <see cref="OnnxScoringEstimator"/>.
    /// </summary>
    public sealed class OnnxOptions
    {
        /// <summary>
        /// Path to the onnx model file.
        /// </summary>
        public string ModelFile;

        /// <summary>
        /// Name of the input column.
        /// </summary>
        public string[] InputColumns;

        /// <summary>
        /// Name of the output column.
        /// </summary>
        public string[] OutputColumns;

        /// <summary>
        /// GPU device id to run on (e.g. 0,1,..). Null for CPU. Requires CUDA 10.1.
        /// </summary>
        public int? GpuDeviceId = null;

        /// <summary>
        /// If true, resumes execution on CPU upon GPU error. If false, will raise the GPU exception.
        /// </summary>
        public bool FallbackToCpu = false;

        /// <summary>
        /// ONNX shapes to be used over those loaded from <see cref="ModelFile"/>.
        /// </summary>
        public IDictionary<string, int[]> ShapeDictionary;

        /// <summary>
        /// Protobuf CodedInputStream recursion limit.
        /// </summary>
        public int RecursionLimit = 100;

        /// <summary>
        /// Controls the number of threads used to parallelize the execution of the graph (across nodes).
        /// </summary>
        public int? InterOpNumThreads = null;

        /// <summary>
        /// Controls the number of threads to use to run the model.
        /// </summary>
        public int? IntraOpNumThreads = null;
    }
}
