// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.GenAI.Core;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Phi.Module;

#pragma warning disable MSML_GeneralName // This name should be PascalCased
internal class Phi2MLP : torch.nn.Module<Tensor, Tensor>
#pragma warning restore MSML_GeneralName // This name should be PascalCased
{
#pragma warning disable MSML_PrivateFieldName // Private field name not in: _camelCase format
    private readonly GenAILinear fc1;
    private readonly GenAILinear fc2;
    private readonly torch.nn.Module<Tensor, Tensor> activation_fn;
#pragma warning restore MSML_PrivateFieldName // Private field name not in: _camelCase format

    public Phi2MLP(Phi2Config config)
        : base(nameof(Phi2MLP))
    {
        this.fc1 = new GenAILinear(config.HiddenSize, config.IntermediateSize, dtype: config.Dtype);
        this.fc2 = new GenAILinear(config.IntermediateSize, config.HiddenSize, dtype: config.Dtype);
        this.activation_fn = new NewGELUActivation();
    }

#pragma warning disable MSML_GeneralName // This name should be PascalCased
    public override Tensor forward(Tensor input)
#pragma warning restore MSML_GeneralName // This name should be PascalCased
    {
        using var input1 = this.fc1.forward(input);
        using var input2 = this.activation_fn.forward(input1);
        return this.fc2.forward(input2);
    }
}
