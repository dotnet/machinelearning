// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using TorchSharp;
using TorchSharp.Modules;

namespace Microsoft.ML.TorchSharp.Roberta.Modules
{
    internal class Embeddings : torch.nn.Module<torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor>
    {
#pragma warning disable MSML_GeneralName // This name should be PascalCased
        public readonly Embedding word_embeddings;
        public readonly Embedding position_embeddings;
        public readonly Embedding token_type_embeddings;
        public readonly LayerNorm LayerNorm;
        public readonly Dropout dropout;
        private bool _disposedValue;

        public Embeddings(long numEmbeddings, long embeddingSize, long maxPositions, long maxTokenTypes,
            double layerNormEps, double dropoutRate)
            : base(nameof(Embeddings))
        {
            word_embeddings = torch.nn.Embedding(numEmbeddings, embeddingSize, padding_idx: 1);
            position_embeddings = torch.nn.Embedding(maxPositions, embeddingSize);
            token_type_embeddings = torch.nn.Embedding(maxTokenTypes, embeddingSize);
            LayerNorm = torch.nn.LayerNorm(new long[] { embeddingSize }, eps: layerNormEps);
            dropout = torch.nn.Dropout(dropoutRate);

            RegisterComponents();
        }

        public override torch.Tensor forward(torch.Tensor tokens, torch.Tensor positions, torch.Tensor segments)
        {
            using var disposeScope = torch.NewDisposeScope();
            var tokenEmbedding = word_embeddings.forward(tokens);
            var positionEmbedding = position_embeddings.forward(positions);
            var tokenTypeEmbedding = token_type_embeddings.forward(segments);
            tokenEmbedding.add_(positionEmbedding).add_(tokenTypeEmbedding);
            var output = LayerNorm.forward(tokenEmbedding);
            output = dropout.forward(output);
            return output.MoveToOuterDisposeScope();
        }

        protected override void Dispose(bool disposing)
        {
            if (!_disposedValue)
            {
                if (disposing)
                {
                    word_embeddings.Dispose();
                    position_embeddings.Dispose();
                    token_type_embeddings.Dispose();
                    LayerNorm.Dispose();
                    dropout.Dispose();
                    _disposedValue = true;
                }
            }

            base.Dispose(disposing);
        }
    }
}
