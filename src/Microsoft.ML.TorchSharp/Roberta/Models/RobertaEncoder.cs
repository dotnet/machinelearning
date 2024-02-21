// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.TorchSharp.Extensions;
using Microsoft.ML.TorchSharp.Roberta.Modules;
using TorchSharp;
using TorchSharp.Modules;
using TransformerEncoder = Microsoft.ML.TorchSharp.NasBert.Models.TransformerEncoder;

namespace Microsoft.ML.TorchSharp.Roberta.Models
{
    internal sealed class RobertaEncoder : TransformerEncoder, torch.nn.IModule<torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor>
    {
#pragma warning disable MSML_GeneralName // This name should be PascalCased
        public readonly Embeddings embeddings;
        public readonly Encoder encoder;
        private bool _disposedValue;

        public RobertaEncoder(int numLayers, int numAttentionHeads,
            long numEmbeddings, long embeddingSize, long hiddenSize, long outputSize, long ffnHiddenSize,
            long maxPositions, long maxTokenTypes, double layerNormEps,
            double embeddingDropoutRate, double attentionDropoutRate, double attentionOutputDropoutRate, double outputDropoutRate)
            : base(nameof(RobertaEncoder))
        {
            embeddings = new Embeddings(numEmbeddings, embeddingSize, maxPositions, maxTokenTypes,
                layerNormEps, embeddingDropoutRate);
            encoder = new Encoder(numLayers, numAttentionHeads, embeddingSize, hiddenSize, outputSize, ffnHiddenSize,
                layerNormEps, attentionDropoutRate, attentionOutputDropoutRate, outputDropoutRate);
            apply(InitWeights);
            RegisterComponents();
        }

        public torch.Tensor call(torch.Tensor tokens, torch.Tensor positions, torch.Tensor tokenTypes, torch.Tensor attentionMask)
        {
            using var disposeScope = torch.NewDisposeScope();
            var x = embeddings.forward(tokens, positions, tokenTypes);
            var sequenceOutput = encoder.forward(x, attentionMask);
            return sequenceOutput.MoveToOuterDisposeScope();
        }

        private void InitWeights(torch.nn.Module module)
        {
            using var disposeScope = torch.NewDisposeScope();
            if (module is Linear linearModule)
            {
                linearModule.weight.normal_(mean: 0.0, std: 0.02);
                if (linearModule.bias.IsNotNull())
                {
                    linearModule.bias.zero_();
                }
            }
            else if (module is Embedding embeddingModule)
            {
                embeddingModule.weight.normal_(mean: 0.0, std: 0.02);
                embeddingModule.weight[1].zero_();  // padding_idx
            }
            else if (module is LayerNorm layerNormModule)
            {
                layerNormModule.weight.fill_(1.0);
                layerNormModule.bias.zero_();
            }
        }

        protected override void Dispose(bool disposing)
        {
            if (!_disposedValue)
            {
                if (disposing)
                {
                    embeddings.Dispose();
                    encoder.Dispose();
                    _disposedValue = true;
                }
            }

            base.Dispose(disposing);
        }
    }
}
