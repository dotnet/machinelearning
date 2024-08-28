// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using TorchSharp;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Core;
#pragma warning disable MSML_GeneralName // This name should be PascalCased
internal class NewGELUActivation : torch.nn.Module<Tensor, Tensor>
#pragma warning disable MSML_GeneralName // This name should be PascalCased
{
    public NewGELUActivation()
        : base(nameof(NewGELUActivation))
    {
    }

    public override Tensor forward(Tensor input)
    {
        using var result = 0.044715 * torch.pow(input, 3.0);
        using var result2 = result + input;
        using var result3 = Math.Sqrt(2.0 / Math.PI) * result2;
        using var result4 = torch.tanh(result3);
        using var result5 = 1.0 + result4;
        return 0.5 * input * result5;
    }
}
