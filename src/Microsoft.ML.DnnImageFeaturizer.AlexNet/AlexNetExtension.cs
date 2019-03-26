// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Onnx;

namespace Microsoft.ML
{
    /// <summary>
    /// This is an extension method to be used with the <see cref="DnnImageFeaturizerEstimator"/> in order to use a pretrained AlexNet model.
    /// The NuGet containing this extension is also guaranteed to include the binary model file.
    /// </summary>
    public static class AlexNetExtension
    {
        /// <summary>
        /// Returns an estimator chain with the two corresponding models (a preprocessing one and a main one) required for the AlexNet pipeline.
        /// Also includes the renaming ColumnsCopyingTransforms required to be able to use arbitrary input and output column names.
        /// This assumes both of the models are in the same location as the file containing this method, which they will be if used through the NuGet.
        /// This should be the default way to use AlexNet if importing the model from a NuGet.
        /// </summary>
        public static EstimatorChain<ColumnCopyingTransformer> AlexNet(this DnnImageModelSelector dnnModelContext, IHostEnvironment env, string outputColumnName, string inputColumnName)
        {
            return AlexNet(dnnModelContext, env, outputColumnName, inputColumnName, Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "DnnImageModels"));
        }

        /// <summary>
        /// This allows a custom model location to be specified. This is useful is a custom model is specified,
        /// or if the model is desired to be placed or shipped separately in a different folder from the main application. Note that because Onnx models
        /// must be in a directory all by themsleves for the OnnxTransformer to work, this method appends a AlexNetOnnx/AlexNetPrepOnnx subdirectory
        /// to the passed in directory to prevent having to make that directory manually each time.
        /// </summary>
        public static EstimatorChain<ColumnCopyingTransformer> AlexNet(this DnnImageModelSelector dnnModelContext, IHostEnvironment env, string outputColumnName, string inputColumnName, string modelDir)
        {
            var modelChain = new EstimatorChain<ColumnCopyingTransformer>();

            var inputRename = new ColumnCopyingEstimator(env, new[] { ("OriginalInput", inputColumnName) });
            var midRename = new ColumnCopyingEstimator(env, new[] { ("Input140", "PreprocessedInput") });
            var endRename = new ColumnCopyingEstimator(env, new[] { (outputColumnName, "Dropout234_Output_0") });

            // There are two estimators created below. The first one is for image preprocessing and the second one is the actual DNN model.
            var prepEstimator = new OnnxScoringEstimator(env, new[] { "PreprocessedInput" }, new[] { "OriginalInput" }, Path.Combine(modelDir, "AlexNetPrepOnnx", "AlexNetPreprocess.onnx"));
            var mainEstimator = new OnnxScoringEstimator(env, new[] { "Dropout234_Output_0" }, new[] { "Input140" }, Path.Combine(modelDir, "AlexNetOnnx", "AlexNet.onnx"));
            modelChain = modelChain.Append(inputRename);
            var modelChain2 = modelChain.Append(prepEstimator);
            modelChain = modelChain2.Append(midRename);
            modelChain2 = modelChain.Append(mainEstimator);
            modelChain = modelChain2.Append(endRename);
            return modelChain;
        }
    }
}