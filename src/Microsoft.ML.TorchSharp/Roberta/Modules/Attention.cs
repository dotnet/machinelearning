// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using TorchSharp;

namespace Microsoft.ML.TorchSharp.Roberta.Modules
{
    internal class Attention : torch.nn.Module<torch.Tensor, torch.Tensor, torch.Tensor>
    {
#pragma warning disable MSML_GeneralName // This name should be PascalCased
        public readonly AttentionSelf self;
        public readonly AttentionOutput output;

        public Attention(int numAttentionHeads, long hiddenSize, double layerNormEps, double attentionDropoutRate, double outputDropoutRate)
            : base(nameof(Attention))
        {
            self = new AttentionSelf(numAttentionHeads, hiddenSize, layerNormEps, attentionDropoutRate);
            output = new AttentionOutput(hiddenSize, layerNormEps, attentionDropoutRate, outputDropoutRate);
            RegisterComponents();
        }

        public override torch.Tensor forward(torch.Tensor hiddenStates, torch.Tensor attentionMask)
        {
            using var disposeScope = torch.NewDisposeScope();
            var x = self.forward(hiddenStates, attentionMask);
            x = output.forward(x, hiddenStates);
            return x.MoveToOuterDisposeScope();
        }
    }
}
