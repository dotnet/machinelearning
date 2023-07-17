// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.TorchSharp.Extensions;
using TorchSharp;
using TorchSharp.Modules;

namespace Microsoft.ML.TorchSharp.Roberta.Modules
{
    internal class AttentionSelf : torch.nn.Module<torch.Tensor, torch.Tensor, torch.Tensor>
    {
        public readonly int NumAttentionHeads;
        public readonly int AttentionHeadSize;

#pragma warning disable MSML_GeneralName // This name should be PascalCased
        public readonly Linear query;
        public readonly Linear key;
        public readonly Linear value;
        public readonly Dropout attention_dropout;

        public AttentionSelf(int numAttentionHeads, long hiddenSize, double layerNormEps, double attentionDropoutRate)
            : base(nameof(AttentionSelf))
        {
            NumAttentionHeads = numAttentionHeads;
            AttentionHeadSize = (int)hiddenSize / numAttentionHeads;
            if (NumAttentionHeads * AttentionHeadSize != hiddenSize)
            {
                throw new ArgumentException($"NumAttentionHeads must be a factor of hiddenSize, got {numAttentionHeads} and {hiddenSize}.");
            }

            query = torch.nn.Linear(hiddenSize, hiddenSize, true);
            key = torch.nn.Linear(hiddenSize, hiddenSize, true);
            value = torch.nn.Linear(hiddenSize, hiddenSize, true);
            attention_dropout = torch.nn.Dropout(attentionDropoutRate);

            RegisterComponents();
        }

        public override torch.Tensor forward(torch.Tensor hiddenStates, torch.Tensor attentionMask)
        {
            using var disposeScope = torch.NewDisposeScope();
            var mixedQueryLayer = query.forward(hiddenStates);
            var mixedKeyLayer = key.forward(hiddenStates);
            var mixedValueLayer = value.forward(hiddenStates);

            var queryLayer = TransposeForScores(mixedQueryLayer);
            var keyLayer = TransposeForScores(mixedKeyLayer);
            var valueLayer = TransposeForScores(mixedValueLayer);

            // Attention
            queryLayer.div_(Math.Sqrt(AttentionHeadSize));
            var attentionScores = torch.matmul(queryLayer, keyLayer.transpose_(-1, -2));
            if (attentionMask.IsNotNull())
            {
                attentionScores.add_(attentionMask);
            }

            var attentionProbs = torch.nn.functional.softmax(attentionScores, dim: -1);
            attentionProbs = attention_dropout.forward(attentionProbs);

            var contextLayer = torch.matmul(attentionProbs, valueLayer);
            contextLayer = contextLayer.permute(0, 2, 1, 3).contiguous();
            var contextShape = Range.GetSubArray(contextLayer.shape, ..^2).Append(NumAttentionHeads * AttentionHeadSize).ToArray();
            contextLayer = contextLayer.view(contextShape);
            return contextLayer.MoveToOuterDisposeScope();
        }

        /// <summary>
        /// [B x T x C] -> [B x Head x T x C_Head]
        /// </summary>
        private torch.Tensor TransposeForScores(torch.Tensor x)
        {
            using var disposeScope = torch.NewDisposeScope();
            var newShape = Range.GetSubArray(x.shape, ..^1).Append(NumAttentionHeads).Append(AttentionHeadSize).ToArray();
            x = x.view(newShape);
            x = x.permute(0, 2, 1, 3).contiguous();
            return x.MoveToOuterDisposeScope();
        }
    }
}
