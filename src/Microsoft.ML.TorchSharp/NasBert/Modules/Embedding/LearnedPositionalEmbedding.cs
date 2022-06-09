// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.TorchSharp.Extensions;
using Microsoft.ML.TorchSharp.Utils;
using TorchSharp;
using TorchSharp.Modules;

namespace Microsoft.ML.TorchSharp.NasBert.Modules
{
    internal sealed class LearnedPositionalEmbedding : PositionalEmbedding
    {
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_PrivateFieldName:Private field name not in: _camelCase format", Justification = "Has to match TorchSharp model.")]
        private readonly Embedding Embedding;
        private readonly int _numEmbeddings;

        public LearnedPositionalEmbedding(int numEmbeddings, int embeddingDim, int padTokenIndex)
            : base(embeddingDim, padTokenIndex, nameof(LearnedPositionalEmbedding))
        {
            _numEmbeddings = numEmbeddings;
            Embedding = torch.nn.Embedding(numEmbeddings, embeddingDim, PadPositionIndex);

            ModelUtils.InitNormal(Embedding.weight, mean: 0, std: Math.Pow(EmbeddingDim, -0.5));
            ModelUtils.InitZeros(Embedding.weight[PadPositionIndex]);

            RegisterComponents();
        }


        /// <summary>
        /// Input is expected to be of size [bsz x seqlen].
        /// Positions should be 0-based and 0 is the padding position index.
        /// </summary>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public override torch.Tensor forward(torch.Tensor input, Dictionary<string, object> param = null)
        {
            using var disposeScope = torch.NewDisposeScope();

            ParseArguments(param, out var incrementalState, out var positions);

            if (positions.IsNull())
            {
                positions = incrementalState
                    ? torch.tensor(PadPositionIndex + input.size(1))
                    : MakePositions(input, PadTokenIndex);
            }

            var embedding = Embedding.forward(positions);
            return embedding.MoveToOuterDisposeScope();
        }

        private static void ParseArguments(IReadOnlyDictionary<string, object> param, out bool incrementalState,
            out torch.Tensor positions)
        {
            incrementalState = false;
            positions = null;
            if (param == null) return;

            if (param.ContainsKey(IncrementalStateKey)) incrementalState = (bool)param[IncrementalStateKey];
            if (param.ContainsKey(PositionKey)) positions = (torch.Tensor)param[PositionKey];
        }
    }
}
