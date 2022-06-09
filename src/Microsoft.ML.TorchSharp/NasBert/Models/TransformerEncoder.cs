// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Runtime;
using Microsoft.ML.TorchSharp.Extensions;
using Microsoft.ML.TorchSharp.NasBert.Modules;
using Microsoft.ML.TorchSharp.Utils;
using TorchSharp;
using TorchSharp.Modules;

namespace Microsoft.ML.TorchSharp.NasBert.Models
{
    internal sealed class TransformerEncoder : BaseModule
    {
#pragma warning disable MSML_PrivateFieldName // Private field name not in: _camelCase format Have to match TorchSharp model

        private readonly int PaddingIdx;
        private readonly int? EmbedScale;
        private readonly int DistillBlocks;
        private readonly List<int> DiscreteArches;
        private readonly List<int> HiddenSizePerBlock;

        private readonly Embedding TokenEmbedding;

        /// <summary>
        /// Null if not using positional embedding.
        /// </summary>
        private readonly PositionalEmbedding PositionalEmbedding;

        /// <summary>
        /// Null if there is only one segment.
        /// </summary>
        private readonly Embedding SegmentEmbedding;

        /// <summary>
        /// Null if not using layer normalization in embedding.
        /// </summary>
        private readonly LayerNorm EmbeddingLayerNorm;
        private readonly EmbedTransfer EmbedTransfer;
        private readonly Dropout DropoutLayer;
        private readonly ModuleList Layers;
        private readonly ModuleList HiddenTransferList;
#pragma warning restore MSML_PrivateFieldName // Private field name not in: _camelCase format

        public Parameter TokenEmbeddingMatrix => TokenEmbedding.weight;

        public TransformerEncoder(
            int paddingIdx,
            int vocabSize,
            double dropout = 0.1f,
            double attentionDropout = 0.1f,
            double activationDropout = 0.1f,
            string activationFn = "relu",
            bool dynamicDropout = false,
            bool addBiasKv = false,
            bool addZeroAttention = false,
            int maxSeqLen = 256,
            bool learnedPositionEmbedding = true,
            int embedSize = -1,
            int? embedScale = null,
            IList<int> arches = null,
            bool usePositionEmbedding = true,
            bool offsetPositionsByPadding = true,
            int numSegments = 2,
            bool encoderNormalizeBefore = false,
            int numEncoderLayers = 6,
            bool applyBertInit = false,
            bool freezeEmbeddings = false,
            bool freezeLayers = false,
            bool freezeTransfer = false,
            int nTransLayersToFreeze = 0)
            : base(nameof(TransformerEncoder))
        {
            Contracts.AssertValue(arches);
            Contracts.AssertNonEmpty(arches);

            PaddingIdx = paddingIdx;
            DiscreteArches = arches.ToList();
            DistillBlocks = 4;

            // Embedding modules
            EmbedScale = embedScale;
            TokenEmbedding = torch.nn.Embedding(vocabSize, embedSize, paddingIdx);
            PositionalEmbedding = usePositionEmbedding
                ? PositionalEmbedding.GetPositionalEmbedding(maxSeqLen, embedSize,
                    paddingIdx, learnedPositionEmbedding)
                : null;
            SegmentEmbedding = numSegments > 0
                ? torch.nn.Embedding(numSegments, embedSize)
                : null;
            EmbeddingLayerNorm = encoderNormalizeBefore
                ? torch.nn.LayerNorm(new long[] { embedSize })
                : null;
            DropoutLayer = torch.nn.Dropout(dropout);

            ModelUtils.InitNormal(TokenEmbedding.weight, mean: 0.0, std: 0.02);
            ModelUtils.InitZeros(TokenEmbedding.weight[paddingIdx]);
            if (SegmentEmbedding != null)
            {
                ModelUtils.InitNormal(SegmentEmbedding.weight, mean: 0.0, std: 0.02);
            }

            // Encoder layers
            var layers = Enumerable.Range(0, numEncoderLayers)
                .Select(i => new TransformerCellDiscrete(
                    arches[i],
                    dropout,
                    attentionDropout,
                    activationDropout,
                    activationFn,
                    addBiasKv,
                    addZeroAttention,
                    dynamicDropout) as torch.nn.Module)
                .ToArray();
            Layers = new ModuleList(layers);

            var blockPerLayer = numEncoderLayers / DistillBlocks;
            HiddenSizePerBlock = CheckBlockHiddenSize(blockPerLayer);

            EmbedTransfer = new EmbedTransferDiscrete(embedSize, HiddenSizePerBlock[0]);
            var hiddenSizePerBlockExtend = HiddenSizePerBlock.Append(HiddenSizePerBlock[HiddenSizePerBlock.Count - 1]).ToList();
            var hiddenTransferList = Enumerable.Range(0, HiddenSizePerBlock.Count)
                .Select(i => new HiddenTransferDiscrete(hiddenSizePerBlockExtend[i],
                        hiddenSizePerBlockExtend[i + 1]) as torch.nn.Module)
                .ToArray();
            HiddenTransferList = new ModuleList(hiddenTransferList);

            if (freezeEmbeddings)
            {
                ModelUtils.FreezeModuleParams(TokenEmbedding);
                ModelUtils.FreezeModuleParams(PositionalEmbedding);
                ModelUtils.FreezeModuleParams(SegmentEmbedding);
                ModelUtils.FreezeModuleParams(EmbeddingLayerNorm);
            }

            if (freezeLayers)
            {
                ModelUtils.FreezeModuleParams(Layers);
                ModelUtils.FreezeModuleParams(HiddenTransferList);
            }

            if (freezeTransfer)
            {
                ModelUtils.FreezeModuleParams(HiddenTransferList);
            }

            for (var i = 0; i < nTransLayersToFreeze; ++i)
            {
                ModelUtils.FreezeModuleParams(Layers[i]);
            }

            RegisterComponents();
        }

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public torch.Tensor forward(
            torch.Tensor tokens,
            torch.Tensor segmentLabels = null,
            torch.Tensor positions = null)
        {
            using var disposeScope = torch.NewDisposeScope();

            var x = ForwardEmbedding(tokens, segmentLabels, positions);

            // Compute padding mask. This is needed for multi-head attention
            var paddingMask = tokens.eq(PaddingIdx);
            var usePaddingMask = paddingMask.any().ToBoolean();

            // Account for padding while computing the representation
            if (usePaddingMask)
            {
                var xValidPart = paddingMask.logical_not().unsqueeze(-1).type_as(x);
                x.mul_(xValidPart);
            }

            // B x T x C -> T x B x C
            x.transpose_(0, 1);

            // forward Layers
            var blockPerLayer = Layers.Count / DistillBlocks;
            var blockIndex = 0;
            for (var i = 0; i < Layers.Count; ++i)
            {
                x = ForwardOneLayer(x, usePaddingMask ? paddingMask : null, i, blockPerLayer, ref blockIndex);
            }

            // T x B x C -> B x T x C
            x.transpose_(0, 1);

            // var sentenceRepresentation = x[torch.TensorIndex.Colon, torch.TensorIndex.Single(0), torch.TensorIndex.Colon];
            return x.MoveToOuterDisposeScope();
        }

        private torch.Tensor ForwardEmbedding(torch.Tensor tokens, torch.Tensor segmentLabels, torch.Tensor positions)
        {
            using var disposeScope = torch.NewDisposeScope();

            var x = TokenEmbedding.forward(tokens);
            if (EmbedScale != null)
            {
                x.mul_(EmbedScale);
            }
            if (PositionalEmbedding != null)
            {
                var positionalEmbedding = PositionalEmbedding.forward(tokens,
                    new Dictionary<string, object> { { PositionalEmbedding.PositionKey, positions } });
                x.add_(positionalEmbedding);
            }
            if (SegmentEmbedding != null && segmentLabels.IsNotNull())
            {
                var segmentEmbedding = SegmentEmbedding.forward(segmentLabels);
                x.add_(segmentEmbedding);
            }
            if (EmbeddingLayerNorm != null)
            {
                x = EmbeddingLayerNorm.forward(x);
            }
            x = EmbedTransfer.forward(x, (int)x.size()[x.size().Length - 1]);
            x = DropoutLayer.forward(x);

            return x.MoveToOuterDisposeScope();
        }

        private torch.Tensor ForwardOneLayer(torch.Tensor input, torch.Tensor paddingMask,
            int i, int blockPerLayer, ref int blockIndex)
        {
            using var disposeScope = torch.NewDisposeScope();

            var x = input.alias();  // avoid scope mess
            var layer = Layers[i];
            if (i % blockPerLayer == 0)
            {
                x = (HiddenTransferList[blockIndex] as HiddenTransfer).forward(x, HiddenSizePerBlock[blockIndex], true);
            }

            x = (layer as TransformerCell).forward(x, null, paddingMask);

            if ((i + 1) % blockPerLayer == 0)
            {
                x = (HiddenTransferList[blockIndex] as HiddenTransfer).forward(x, HiddenSizePerBlock[blockIndex], false);
                ++blockIndex;
            }

            return x.MoveToOuterDisposeScope();
        }

        /// <summary>
        /// For each block, check whether all hidden dimensions in hiddenList are the same (except for 0).
        /// If all hidden dimensions in one block are 0, it will be set to the last hidden dimension
        /// (if exists) or the maximum hidden dimension (if not exist).
        /// </summary>
        /// <returns>The list of hidden dimensions in blocks.</returns>
        private List<int> CheckBlockHiddenSize(int blockPerLayer)
        {
            var hiddenSizePerBlock = new List<int>();
            for (var i = 0; i < DistillBlocks; ++i)
            {
                var hiddenSizesPerBlock = Enumerable.Range(i * blockPerLayer, blockPerLayer)
                    .Select(j => SearchSpace.ArchHiddenSize[DiscreteArches[j]]).ToArray();
                var nextHiddenSize = SearchSpace.CheckHiddenDimensionsAndReturnMax(hiddenSizesPerBlock);
                if (nextHiddenSize == 0)
                {
                    if (hiddenSizePerBlock.Count == 0)
                    {
                        nextHiddenSize = SearchSpace.ArchHiddenSize[SearchSpace.ArchHiddenSize.Length - 1];
                    }
                    else
                    {
                        nextHiddenSize = hiddenSizePerBlock[hiddenSizePerBlock.Count - 1];
                    }
                }
                hiddenSizePerBlock.Add(nextHiddenSize);
            }

            return hiddenSizePerBlock;
        }

        public void CloseLayerNormTraining()
        {
            EmbeddingLayerNorm?.eval();
            foreach (var layer in Layers)
            {
                (layer as TransformerCell)!.CloseLayerNormTraining();
            }
        }
    }

}
