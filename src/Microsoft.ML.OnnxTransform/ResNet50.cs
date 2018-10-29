// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Transforms
{
    public partial class DnnImageModelSelector
    {
        public EstimatorChain<OnnxTransform> ResNet50()
        {
            var modelChain = new EstimatorChain<OnnxTransform>();
            var tempCol = "onnxDnnPrep";
            var prepEstimator = new OnnxEstimator(_env, "C:\\Models\\DnnImageFeat\\Results\\FinalOnnx\\Prep\\resnetPreprocess.onnx", _input, tempCol);
            var mainEstimator = new OnnxEstimator(_env, "C:\\Models\\DnnImageFeat\\Results\\FinalOnnx\\ResNet50\\resnet50.onnx", tempCol, _output);
            modelChain.Append(prepEstimator);
            modelChain.Append(mainEstimator);
            return modelChain;
        }
    }
}