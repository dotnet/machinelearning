// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using System.IO;
using System.Reflection;

namespace Microsoft.ML.Transforms
{
    // This is an extension method to be used with the DnnImageFeaturizerTransform in order to use a pretrained ResNet18 model.
    // The NuGet containing this extension is also guaranteed to include the binary model file.
    public static class ResNet18Extension
    {
        // If including this through a NuGet, the location of the model will be the same as of this file. This looks for the model there.
        // This should be the default way to use ResNet18 if importing the model from a NuGet.
        public static EstimatorChain<OnnxTransform> ResNet18(this DnnImageModelSelector model)
        {
            var modelChain = new EstimatorChain<OnnxTransform>();
            var tempCol = "onnxDnnPrep";
            var execDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);

            var prepEstimator = new OnnxScoringEstimator(model.Env, Path.Combine(execDir, "ResNetPrepOnnx", "ResNetPreprocess.onnx"), model.Input, tempCol);
            var mainEstimator = new OnnxScoringEstimator(model.Env, Path.Combine(execDir, "ResNet18Onnx", "ResNet18.onnx"), tempCol, model.Output);
            modelChain = modelChain.Append(prepEstimator);
            modelChain = modelChain.Append(mainEstimator);
            return modelChain;
        }

        // This allows a custom model location to be specified. This is mainly useful for other code inside of ML.NET (such as tests),
        // but could also be used if one wishes to change the directory of the model for whatever reason. Note that because Onnx models
        // must be in a directory all by themsleves for the OnnxTransform to work, this method appends a ResNet18Onnx/ResNetPrepOnnx subdirectory
        // to the passed in directory to prevent having to make that directory manually each time.
        public static EstimatorChain<OnnxTransform> ResNet18(this DnnImageModelSelector model, string modelDir)
        {
            var modelChain = new EstimatorChain<OnnxTransform>();
            var tempCol = "onnxDnnPrep";

            var prepEstimator = new OnnxScoringEstimator(model.Env, Path.Combine(modelDir, "ResNetPrepOnnx", "ResNetPreprocess.onnx"), model.Input, tempCol);
            var mainEstimator = new OnnxScoringEstimator(model.Env, Path.Combine(modelDir, "ResNet18Onnx", "ResNet18.onnx"), tempCol, model.Output);
            modelChain = modelChain.Append(prepEstimator);
            modelChain = modelChain.Append(mainEstimator);
            return modelChain;
        }
    }
}