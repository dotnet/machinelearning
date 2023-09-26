// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using TorchSharp;
using TorchSharp.Modules;

namespace Microsoft.ML.TorchSharp.Roberta.Modules
{
    internal class AttentionOutput : torch.nn.Module<torch.Tensor, torch.Tensor, torch.Tensor>
    {
#pragma warning disable MSML_GeneralName // This name should be PascalCased
        public readonly Linear dense;
        public readonly Dropout dropout;
        public readonly LayerNorm LayerNorm;

        public AttentionOutput(long hiddenSize, double layerNormEps, double attentionDropoutRate, double outputDropoutRate)
            : base(nameof(AttentionOutput))
        {
            dense = torch.nn.Linear(hiddenSize, hiddenSize, true);
            dropout = torch.nn.Dropout(outputDropoutRate);
            LayerNorm = torch.nn.LayerNorm(new long[] { hiddenSize });
            RegisterComponents();
        }

        public override torch.Tensor forward(torch.Tensor hiddenStates, torch.Tensor inputTensor)
        {
            using var disposeScope = torch.NewDisposeScope();
            hiddenStates = dense.forward(hiddenStates);
            hiddenStates = dropout.forward(hiddenStates);
            hiddenStates = LayerNorm.forward(hiddenStates + inputTensor);
            return hiddenStates.MoveToOuterDisposeScope();
        }
    }
}
