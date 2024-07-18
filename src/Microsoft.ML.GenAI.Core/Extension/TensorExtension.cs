// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using TorchSharp;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Core.Extension;

internal static class TensorExtension
{
    public static string Peek(this Tensor tensor, string id, int n = 10)
    {
        var device = tensor.device;
        var dType = tensor.dtype;
        // if type is fp16, convert to fp32
        if (tensor.dtype == ScalarType.Float16)
        {
            tensor = tensor.to_type(ScalarType.Float32);
        }
        tensor = tensor.cpu();
        var shapeString = string.Join(',', tensor.shape);
        var tensor1D = tensor.reshape(-1);
        var tensorIndex = torch.arange(tensor1D.shape[0], dtype: ScalarType.Float32).to(tensor1D.device).sqrt();
        var avg = (tensor1D * tensorIndex).sum();
        avg = avg / tensor1D.sum();
        // keep four decimal places
        avg = avg.round(4);
        var str = $"{id}: sum: {avg.ToSingle()}  dType: {dType} shape: [{shapeString}]";

        return str;
    }
}
