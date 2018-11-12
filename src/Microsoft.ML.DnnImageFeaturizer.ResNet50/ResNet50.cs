// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using System.IO;
using System.Reflection;

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// This is an extension method to be used with the <see cref="DnnImageFeaturizerEstimator"/> in order to use a pretrained ResNet50 model.
    /// The NuGet containing this extension is also guaranteed to include the binary model file. Note that when building the project
    /// containing this extension method, the corresponding binary model will be downloaded from the CDN at
    /// https://express-tlcresources.azureedge.net/image/ResNetPrepOnnx/ResNetPreprocess.onnx and
    /// https://express-tlcresources.azureedge.net/image/ResNet50Onnx/ResNet50.onnx  and placed into the local app directory
    /// folder under mlnet-resources.
    /// </summary>
    public static class ResNet50Extension
    {
        /// <summary>
        /// If including this through a NuGet, the location of the model will be the same as of this file. This looks for the model there.
        /// This should be the default way to use ResNet50 if importing the model from a NuGet.
        /// </summary>
        public static EstimatorChain<OnnxTransform> ResNet50(this DnnImageModelSelector dnnModelContext, IHostEnvironment env, string inputColumn, string outputColumn)
        {
            return ResNet50(dnnModelContext, env, inputColumn, outputColumn, Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location));
        }

        /// <summary>
        /// This allows a custom model location to be specified. This is useful is a custom model is specified,
        /// or if the model is desired to be placed or shipped separately in a different folder from the main application. Note that because Onnx models
        /// must be in a directory all by themsleves for the OnnxTransform to work, this method appends a ResNet50Onnx/ResNetPrepOnnx subdirectory
        /// to the passed in directory to prevent having to make that directory manually each time.
        /// </summary>
        public static EstimatorChain<OnnxTransform> ResNet50(this DnnImageModelSelector dnnModelContext, IHostEnvironment env, string inputColumn, string outputColumn, string modelDir)
        {
            var modelChain = new EstimatorChain<OnnxTransform>();
            var tempCol = "onnxDnnPrep";

            // There are two estimators created below. The first one is for image preprocessing and the second one is the actual DNN model.
            var prepEstimator = new OnnxScoringEstimator(env, Path.Combine(modelDir, "ResNetPrepOnnx", "ResNetPreprocess.onnx"), inputColumn, tempCol);
            var mainEstimator = new OnnxScoringEstimator(env, Path.Combine(modelDir, "ResNet50Onnx", "ResNet50.onnx"), tempCol, outputColumn);
            modelChain = modelChain.Append(prepEstimator);
            modelChain = modelChain.Append(mainEstimator);
            return modelChain;
        }
    }
}