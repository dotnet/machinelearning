// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.GenAI.Core;
using TorchSharp;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Mistral.Module;
#pragma warning disable MSML_GeneralName // This name should be PascalCased
internal class MistralMLP : torch.nn.Module<Tensor, Tensor>
#pragma warning restore MSML_GeneralName // This name should be PascalCased
{
    private readonly int _intermediateSize;
    private readonly int _hiddenSize;
#pragma warning disable MSML_PrivateFieldName // Private field name not in: _camelCase format
    private readonly QuantizedLinear gate_proj;
    private readonly QuantizedLinear up_proj;
    private readonly QuantizedLinear down_proj;
    private readonly torch.nn.Module<Tensor, Tensor> act_fn;
#pragma warning restore MSML_PrivateFieldName // Private field name not in: _camelCase format

    public MistralMLP(MistralConfig config)
        : base(nameof(MistralMLP))
    {
        this._hiddenSize = config.HiddenSize;
        this._intermediateSize = config.IntermediateSize;
        var hiddenAct = config.HiddenAct;
        this.gate_proj = new QuantizedLinear(this._hiddenSize, this._intermediateSize, hasBias: false, dtype: config.DType);
        this.up_proj = new QuantizedLinear(this._hiddenSize, this._intermediateSize, hasBias: false, dtype: config.DType);
        this.down_proj = new QuantizedLinear(this._intermediateSize, this._hiddenSize, hasBias: false, dtype: config.DType);
        this.RegisterComponents();
        this.act_fn = Core.Utils.GetActivation(hiddenAct);
    }

#pragma warning disable MSML_GeneralName // This name should be PascalCased
    public override Tensor forward(Tensor input)
#pragma warning restore MSML_GeneralName // This name should be PascalCased
    {
        using var input1 = this.gate_proj.forward(input);
        using var input2 = this.act_fn.forward(input1);
        using var input3 = input2 * this.up_proj.forward(input);
        return this.down_proj.forward(input3);
    }
}
