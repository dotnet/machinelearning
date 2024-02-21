// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.TorchSharp.Roberta.Modules;
using System.Text;
using TorchSharp;
using TorchSharp.Modules;

namespace Microsoft.ML.TorchSharp.Roberta.Models
{
    internal class Encoder : torch.nn.Module<torch.Tensor, torch.Tensor, torch.Tensor>
    {
        public readonly int NumLayers;
#pragma warning disable MSML_GeneralName // This name should be PascalCased
        public readonly ModuleList<Layer> layer;
        private bool _disposedValue;

        public Encoder(int numLayers, int numAttentionHeads, long embeddingSize, long hiddenSize, long outputSize, long ffnHiddenSize,
            double layerNormEps, double dropoutRate, double attentionDropoutRate, double outputDropoutRate)
            : base(nameof(Encoder))
        {
            NumLayers = numLayers;
#pragma warning disable MSML_ParameterLocalVarName // Parameter or local variable name not standard
            layer = new ModuleList<Layer>(Enumerable.Range(0, numLayers)
                .Select(_ => new Layer(numAttentionHeads, hiddenSize, ffnHiddenSize,
                    layerNormEps, dropoutRate, attentionDropoutRate, outputDropoutRate))
                .ToArray());
#pragma warning restore MSML_ParameterLocalVarName // Parameter or local variable name not standard
            RegisterComponents();
        }

        public override torch.Tensor forward(torch.Tensor x, torch.Tensor attentionMask)
        {
            using var disposeScope = torch.NewDisposeScope();
            foreach (var lyr in layer)
            {
                x = lyr.forward(x, attentionMask);
            }
            return x.MoveToOuterDisposeScope();
        }

        protected override void Dispose(bool disposing)
        {
            if (!_disposedValue)
            {
                if (disposing)
                {
                    layer.Dispose();
                    _disposedValue = true;
                }
            }

            base.Dispose(disposing);
        }
    }
}
