// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.TorchSharp.NasBert.Modules.Layers;
using System;
using Microsoft.ML.TorchSharp.Roberta.Modules;
using TorchSharp;

namespace Microsoft.ML.TorchSharp.Roberta.Models
{
    internal class Layer : torch.nn.Module<torch.Tensor, torch.Tensor, torch.Tensor>
    {
#pragma warning disable MSML_GeneralName // This name should be PascalCased
        public readonly Attention attention;
        public readonly Intermediate intermediate;
        public readonly Output output;
        private bool _disposedValue;

        public Layer(int numAttentionHeads, long hiddenSize, long ffnHiddenSize, double layerNormEps,
            double dropoutRate, double attentionDropoutRate, double outputDropoutRate)
            : base(nameof(Layer))
        {
            attention = new Attention(numAttentionHeads, hiddenSize, layerNormEps, attentionDropoutRate, outputDropoutRate);
            intermediate = new Intermediate(hiddenSize, ffnHiddenSize);
            output = new Output(ffnHiddenSize, hiddenSize, dropoutRate);
            RegisterComponents();
        }

        public override torch.Tensor forward(torch.Tensor input, torch.Tensor attentionMask)
        {
            using var disposeScope = torch.NewDisposeScope();
            var attentionOutput = attention.forward(input, attentionMask);
            var intermediateOutput = intermediate.forward(attentionOutput);
            var layerOutput = output.forward(intermediateOutput, attentionOutput);
            return layerOutput.MoveToOuterDisposeScope();
        }

        protected override void Dispose(bool disposing)
        {
            if (!_disposedValue)
            {
                if (disposing)
                {
                    attention.Dispose();
                    intermediate.Dispose();
                    output.Dispose();
                    _disposedValue = true;
                }
            }

            base.Dispose(disposing);
        }
    }
}
