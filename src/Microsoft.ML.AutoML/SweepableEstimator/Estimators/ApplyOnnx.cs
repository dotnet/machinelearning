// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.AutoML.CodeGen
{
    internal partial class ApplyOnnxModel
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, ApplyOnnxModelOption param)
        {
            return context.Transforms.ApplyOnnxModel(outputColumnName: param.OutputColumnName, inputColumnName: param.InputColumnName, modelFile: param.ModelFile, gpuDeviceId: param.GpuDeviceId, fallbackToCpu: param.FallbackToCpu);
        }
    }
}
