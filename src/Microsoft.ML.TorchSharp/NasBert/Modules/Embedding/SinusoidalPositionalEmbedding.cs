// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using TorchSharp;
using TorchSharp.Modules;

namespace Microsoft.ML.TorchSharp.NasBert.Modules
{
    internal sealed class SinusoidalPositionalEmbedding : PositionalEmbedding
    {
        private readonly torch.Tensor _floatTensor = torch.tensor(1.0f);
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_PrivateFieldName:Private field name not in: _camelCase format", Justification = "Has to match TorchSharp model.")]
        private Parameter Weight;

        public SinusoidalPositionalEmbedding(int numEmbeddings, int embeddingDim, int padTokenIndex)
            : base(embeddingDim, padTokenIndex, nameof(SinusoidalPositionalEmbedding))
        {
            Weight = GetEmbedding(numEmbeddings, embeddingDim);

            RegisterComponents();
        }

        /// <summary>
        /// Build sinusoidal embeddings.
        /// This matches the implementation in tensor2tensor, but differs slightly
        ///     from the description in Section 3.5 of "Attention Is All You Need".
        /// </summary>
        private static Parameter GetEmbedding(int numEmbeddings, int embeddingDim)
        {
            using var disposeScope = torch.NewDisposeScope();

            var halfDim = embeddingDim / 2;
            var embedDouble = Math.Log(10000) / (halfDim - 1);

            var embedBaseCol = torch.arange(halfDim, dtype: torch.float32).mul_(-embedDouble).exp_().unsqueeze_(0);
            var embedBaseRow = torch.arange(numEmbeddings, dtype: torch.float32).unsqueeze_(1);
            var embedBase = embedBaseRow.mul(embedBaseCol);
            var sinEmbed = torch.sin(embedBase);
            var cosEmbed = torch.cos(embedBase);
            var embedding = torch.cat(new List<torch.Tensor> { sinEmbed, cosEmbed }, 1);

            // zero pad
            if (embeddingDim % 2 == 1)
            {
                var zeroPad = torch.zeros(numEmbeddings, 1);
                embedding = torch.cat(new List<torch.Tensor> { embedding, zeroPad }, 1);
            }

            embedding[PadPositionIndex, torch.TensorIndex.Colon].fill_(0);

            // We must call parameter.MoveToOuterDisposeScope(), otherwise parameter will be disposed after return.
            // It is not OK to return new Parameter(embedding.MoveToOuterDisposeScope()).
            return (Parameter)new Parameter(embedding).MoveToOuterDisposeScope();
        }


        /// <summary>
        /// Input is expected to be of size [bsz x seqlen].
        /// </summary>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public override torch.Tensor forward(torch.Tensor input, Dictionary<string, object> param = null)
        {
            using var disposeScope = torch.NewDisposeScope();

            ParseArguments(param, out var incrementalState, out var timeStep);

            var bszInt = (int)input.shape[0];
            var seqLenInt = (int)input.shape[1];
            var maxPosition = (int)(PadPositionIndex + 1 + input.shape[1]);

            // recompute/expand embeddings if needed
            if (Weight is null || maxPosition > Weight.size(0))
            {
                Weight?.Dispose();
                Weight = GetEmbedding(maxPosition, EmbeddingDim);
                Weight = (Parameter)Weight.MoveToOuterDisposeScope();
            }

            // move Weight to the device where _float_tensor is
            foreach (var (bufferName, buffer) in named_buffers())
            {
                if (bufferName == nameof(_floatTensor))
                {
                    Weight = (Parameter)Weight.to(buffer);
                    Weight = (Parameter)Weight.MoveToOuterDisposeScope();
                    break;
                }
            }

            // positions is the same for every token when decoding a single step
            if (incrementalState)
            {
                var pos = timeStep is null
                    ? seqLenInt
                    : timeStep.item<int>() + 1;
                var slice = Weight[torch.TensorIndex.Single(PadPositionIndex + pos), torch.TensorIndex.Colon];
                return slice.expand(bszInt, 1, 1).MoveToOuterDisposeScope();
            }

            var positions = MakePositions(input, PadTokenIndex).view(-1);
            var weightsSelected = Weight.index_select(0, positions).view(bszInt, seqLenInt, -1);
            return weightsSelected.detach().MoveToOuterDisposeScope();
        }

        private static void ParseArguments(IReadOnlyDictionary<string, object> param, out bool incrementalState, out torch.Tensor timeStep)
        {
            incrementalState = false;
            timeStep = null;
            if (param == null) return;

            if (param.ContainsKey(IncrementalStateKey)) incrementalState = (bool)param[IncrementalStateKey];
            if (param.ContainsKey(TimeStepKey)) timeStep = (torch.Tensor)param[TimeStepKey];
        }
    }
}
