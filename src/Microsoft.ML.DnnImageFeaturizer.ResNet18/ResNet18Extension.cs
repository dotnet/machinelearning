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
    /// This is an extension method to be used with the <see cref="DnnImageFeaturizerEstimator"/> in order to use a pretrained ResNet18 model.
    /// The NuGet containing this extension is also guaranteed to include the binary model file.
    /// </summary>
    public static class ResNet18Extension
    {
        /// <summary>
        /// Returns an estimator chain with the two corresponding models (a preprocessing one and a main one) required for the ResNet pipeline.
        /// Also includes the renaming ColumnsCopyingTransforms required to be able to use arbitrary input and output column names.
        /// This assumes both of the models are in the same location as the file containing this method, which they will be if used through the NuGet.
        /// This should be the default way to use ResNet18 if importing the model from a NuGet.
        /// </summary>
        public static EstimatorChain<ColumnCopyingTransformer> ResNet18(this DnnImageModelSelector dnnModelContext, IHostEnvironment env, string outputColumnName, string inputColumnName)
        {
            return ResNet18(dnnModelContext, env, outputColumnName, inputColumnName, Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "DnnImageModels"));
        }

        /// <summary>
        /// This allows a custom model location to be specified. This is useful is a custom model is specified,
        /// or if the model is desired to be placed or shipped separately in a different folder from the main application. Note that because Onnx models
        /// must be in a directory all by themsleves for the OnnxTransformer to work, this method appends a ResNet18Onnx/ResNetPrepOnnx subdirectory
        /// to the passed in directory to prevent having to make that directory manually each time.
        /// </summary>
        public static EstimatorChain<ColumnCopyingTransformer> ResNet18(this DnnImageModelSelector dnnModelContext, IHostEnvironment env, string outputColumnName, string inputColumnName, string modelDir)
        {
            var modelChain = new EstimatorChain<ColumnCopyingTransformer>();

            var inputRename = new ColumnCopyingEstimator(env, new[] { ("OriginalInput", inputColumnName) });
            var midRename = new ColumnCopyingEstimator(env, new[] { ("Input247", "PreprocessedInput") });
            var endRename = new ColumnCopyingEstimator(env, new[] { (outputColumnName, "Pooling395_Output_0") });

            // There are two estimators created below. The first one is for image preprocessing and the second one is the actual DNN model.
            var prepEstimator = new OnnxScoringEstimator(env, new[] { "PreprocessedInput" }, new[] { "OriginalInput" }, Path.Combine(modelDir, "ResNetPrepOnnx", "ResNetPreprocess.onnx"));
            var mainEstimator = new OnnxScoringEstimator(env, new[] { "Pooling395_Output_0" }, new[] { "Input247" }, Path.Combine(modelDir, "ResNet18Onnx", "ResNet18.onnx"));
            modelChain = modelChain.Append(inputRename);
            var modelChain2 = modelChain.Append(prepEstimator);
            modelChain = modelChain2.Append(midRename);
            modelChain2 = modelChain.Append(mainEstimator);
            modelChain = modelChain2.Append(endRename);
            return modelChain;
        }
    }
}