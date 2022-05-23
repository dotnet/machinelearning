// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using TorchSharp;

namespace Microsoft.ML.TorchSharp.NasBert.Modules
{
    internal class PositionalEmbedding : BaseModule
    {
        public const string TimeStepKey = "timeStep";
        public const string IncrementalStateKey = "incrementalState";
        public const string PositionKey = "positions";

        protected int EmbeddingDim { get; }
        protected int PadTokenIndex { get; }
        protected const int PadPositionIndex = 0;

        protected PositionalEmbedding(int embeddingDim, int padTokenIndex, string name = null)
            : base(name ?? nameof(PositionalEmbedding))
        {
            EmbeddingDim = embeddingDim;
            PadTokenIndex = padTokenIndex;
        }

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public virtual torch.Tensor forward(torch.Tensor input, Dictionary<string, object> param = null)
        {
            return input.alias();
        }

        public static PositionalEmbedding GetPositionalEmbedding(int numEmbeddings, int embeddingDim,
            int padTokenIndex = -1, bool learned = false)
        {
            // If padding_idx is specified then offset the embedding ids by
            // this index and adjust num_embeddings appropriately.
            // TODO: The right place for this offset would be inside
            //   LearnedPositionalEmbedding. Move this there for a cleaner implementation.

            // leave space for padding positions (PadPositionIndex + 1) and prepended <s> token (1)
            numEmbeddings += (PadPositionIndex + 1) + 1;
            return learned
                ? (PositionalEmbedding)new LearnedPositionalEmbedding(numEmbeddings, embeddingDim, padTokenIndex)
                : new SinusoidalPositionalEmbedding(numEmbeddings, embeddingDim, padTokenIndex);
        }

        /// <summary>
        /// Replace non-padding symbols with their position numbers.
        /// Position numbers begin at padTokenIndex+1. Padding symbols are ignored.
        /// </summary>
        /// <param name="tensor">Cannot be null.</param>
        /// <param name="padTokenIndex"></param>
        protected static torch.Tensor MakePositions(torch.Tensor tensor, int padTokenIndex)
        {
            using var disposeScope = torch.NewDisposeScope();
            var mask = tensor.ne(padTokenIndex).@long();
            var positions = torch.cumsum(mask, dimension: 1).mul_(mask);
            positions.add_(PadPositionIndex);
            return positions.MoveToOuterDisposeScope();
        }
    }
}
