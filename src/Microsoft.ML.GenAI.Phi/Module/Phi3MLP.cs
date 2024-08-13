// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.GenAI.Core;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Phi.Module;
#pragma warning disable MSML_GeneralName // This name should be PascalCased
internal class Phi3MLP : torch.nn.Module<Tensor, Tensor>
#pragma warning restore MSML_GeneralName // This name should be PascalCased
{
#pragma warning disable MSML_PrivateFieldName // Private field name not in: _camelCase format
    private readonly QuantizedLinear gate_up_proj;
    private readonly QuantizedLinear down_proj;
    private readonly torch.nn.Module<Tensor, Tensor> activation_fn;
#pragma warning restore MSML_PrivateFieldName // Private field name not in: _camelCase format

    public Phi3MLP(Phi3Config config)
        : this(config.HiddenSize, config.IntermediateSize, config.HiddenAct, config.DType)
    {
    }

    public Phi3MLP(int hiddenSize, int intermediateSize, string hiddenAct, ScalarType dtype)
        : base(nameof(Phi3MLP))
    {
        this.gate_up_proj = new QuantizedLinear(hiddenSize, 2 * intermediateSize, hasBias: false, dtype: dtype);
        this.down_proj = new QuantizedLinear(intermediateSize, hiddenSize, hasBias: false, dtype: dtype);
        this.RegisterComponents();
        this.activation_fn = Core.Utils.GetActivation(hiddenAct);
    }

#pragma warning disable MSML_GeneralName // This name should be PascalCased
    public override Tensor forward(Tensor input)
#pragma warning restore MSML_GeneralName // This name should be PascalCased
    {
        using var input1 = this.gate_up_proj.forward(input);
        var chunks = input1.chunk(2, dim: -1);
        var gate = chunks[0];
        var upStatus = chunks[1];
        upStatus = upStatus * this.activation_fn.forward(gate);
        return this.down_proj.forward(upStatus);
    }
}
