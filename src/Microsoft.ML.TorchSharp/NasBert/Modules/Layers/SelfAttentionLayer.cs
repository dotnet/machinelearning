// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using TorchSharp;
using TorchSharp.Modules;

namespace Microsoft.ML.TorchSharp.NasBert.Modules.Layers
{
    internal sealed class SelfAttentionLayer : Layer
    {
#pragma warning disable MSML_PrivateFieldName // Private field name not in: _camelCase format
        private readonly MultiHeadAttention SelfAttention;
        private readonly LayerNorm LayerNorm;
        private readonly Dropout DropoutLayer;
#pragma warning restore MSML_PrivateFieldName // Private field name not in: _camelCase format


        public SelfAttentionLayer(
            int embeddingDim = 768,
            int numAttentionHeads = 8,
            double dropoutRate = 0.1f,
            double attentionDropoutRate = 0.1f,
            bool addBiasKv = false,
            bool addZeroAttention = false)
            : base(nameof(SelfAttentionLayer))
        {
            SelfAttention = new MultiHeadAttention(
                embeddingDim,
                numAttentionHeads,
                dropout: attentionDropoutRate,
                addBiasKv: addBiasKv,
                addZeroAttention: addZeroAttention,
                selfAttention: true);
            DropoutLayer = torch.nn.Dropout(dropoutRate);

            // Layer norm associated with the self attention layer
            LayerNorm = torch.nn.LayerNorm(new long[] { embeddingDim });

            RegisterComponents();
        }

        /// <summary>
        /// LayerNorm is applied either before or after the self-attention/ffn modules
        ///     similar to the original Transformer implementation.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="param"></param>
        /// <returns></returns>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public override torch.Tensor forward(torch.Tensor x, Dictionary<string, object> param)
        {
            using var disposeScope = torch.NewDisposeScope();

            if (!ParseArguments(param, out var selfAttentionMask, out var selfAttentionPaddingMask))
            {
                throw new ArgumentException("Invalid arguments.");
            }

            var attention = SelfAttention.forward(query: x, key: x, value: x,
                out _,
                keyPaddingMask: selfAttentionPaddingMask,
                needWeights: false,
                attentionMask: selfAttentionMask);
            var dropout = DropoutLayer.forward(attention);
            dropout.add_(x);
            var norm = LayerNorm.forward(dropout);
            return norm.MoveToOuterDisposeScope();
        }

        public override void CloseLayerNormTraining() => LayerNorm.eval();

        private static bool ParseArguments(IReadOnlyDictionary<string, object> param,
            out torch.Tensor selfAttentionMask, out torch.Tensor selfAttentionPaddingMask)
        {
            selfAttentionMask = selfAttentionPaddingMask = null;
            if (!(param.ContainsKey(AttentionMaskKey) && param.ContainsKey(PaddingMaskKey))) return false;

            selfAttentionMask = (torch.Tensor)param[AttentionMaskKey];
            selfAttentionPaddingMask = (torch.Tensor)param[PaddingMaskKey];
            return true;
        }
    }
}
