// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.GenAI.Core;
using Microsoft.ML.GenAI.LLaMA;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.LLaMA.Module;
#pragma warning disable MSML_GeneralName // This name should be PascalCased
internal class LlamaMLP : torch.nn.Module<Tensor, Tensor>
#pragma warning restore MSML_GeneralName // This name should be PascalCased
{
    private readonly int _pretrainingTp;
    private readonly int _intermediateSize;
    private readonly int _hiddenSize;
    private readonly bool _hasBias;
#pragma warning disable MSML_PrivateFieldName // Private field name not in: _camelCase format
    private readonly QuantizedLinear gate_proj;
    private readonly QuantizedLinear up_proj;
    private readonly QuantizedLinear down_proj;
    private readonly torch.nn.Module<Tensor, Tensor> activation_fn;
#pragma warning restore MSML_PrivateFieldName // Private field name not in: _camelCase format

    public LlamaMLP(LlamaConfig config)
        : base(nameof(LlamaMLP))
    {
        this._hiddenSize = config.HiddenSize;
        this._intermediateSize = config.IntermediateSize;
        this._hasBias = config.MlpBias;
        this._pretrainingTp = config.PretrainingTp;
        var hiddenAct = config.HiddenAct;
        this.gate_proj = new QuantizedLinear(this._hiddenSize, this._intermediateSize, hasBias: this._hasBias, dtype: config.DType);
        this.up_proj = new QuantizedLinear(this._hiddenSize, this._intermediateSize, hasBias: this._hasBias, dtype: config.DType);
        this.down_proj = new QuantizedLinear(this._intermediateSize, this._hiddenSize, hasBias: this._hasBias, dtype: config.DType);
        this.RegisterComponents();
        this.activation_fn = Core.Utils.GetActivation(hiddenAct);
    }

#pragma warning disable MSML_GeneralName // This name should be PascalCased
    public override Tensor forward(Tensor input)
#pragma warning restore MSML_GeneralName // This name should be PascalCased
    {
        if (this._pretrainingTp > 1)
        {
            throw new NotImplementedException("PretrainingTp > 1 is not supported yet.");
        }

        using var input1 = this.gate_proj.forward(input);
        using var input2 = this.activation_fn.forward(input1);
        using var input3 = input2 * this.up_proj.forward(input);
        return this.down_proj.forward(input3);
    }
}
