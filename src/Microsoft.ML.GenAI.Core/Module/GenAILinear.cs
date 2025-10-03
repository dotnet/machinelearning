// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using TorchSharp;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Core;

internal class GenAILinear : nn.Module<Tensor, Tensor>
{
#pragma warning disable MSML_GeneralName // This name should be PascalCased
    protected Tensor? weight;
    protected Tensor? bias;
    protected readonly int _inFeatures;
    protected readonly int _outFeatures;
#pragma warning restore MSML_GeneralName // This name should be PascalCased

    public GenAILinear(int inFeatures, int outFeatures, bool hasBias = true, ScalarType dtype = ScalarType.Float32, string? device = null)
        : base(nameof(GenAILinear))
    {
        this._inFeatures = inFeatures;
        this._outFeatures = outFeatures;
        device ??= torch.get_default_device().ToString();
        this.weight = torch.zeros(outFeatures, inFeatures, dtype: dtype, device: device);

        if (hasBias)
        {
            this.bias = torch.zeros(outFeatures, dtype: dtype, device: device);
        }

        base.RegisterComponents();
    }

#pragma warning disable MSML_GeneralName // This name should be PascalCased
    public override Tensor forward(Tensor input)
#pragma warning restore MSML_GeneralName // This name should be PascalCased
    {
        using var dispose = torch.NewDisposeScope();

        // use float32
        var input2 = input.to_type(ScalarType.Float32);
        var weight2 = this.weight!.to_type(ScalarType.Float32);
        var result = torch.matmul(input2, weight2.t());

        if (this.bias is not null)
        {
            result = result + this.bias.to_type(ScalarType.Float32);
        }

        return result.to_type(input.dtype).MoveToOuterDisposeScope();
    }
}
