// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.TorchSharp.NasBert.Modules.Layers;

namespace Microsoft.ML.TorchSharp.NasBert.Modules
{
    internal static class SearchSpace
    {
        public static readonly int[] HiddenSizeChoices = { 128, 192, 384, 512, 768 };
        public static readonly int[] EmbSizeChoices = { 64, 128, 256, 384, 512 };
        public static readonly int[] ArchHiddenSize =
        {
            0, 128, 128, 128, 128, 128,
            0, 192, 192, 192, 192, 192,
            0, 384, 384, 384, 384, 384,
            0, 512, 512, 512, 512, 512,
            0, 768, 768, 768, 768, 768,
        };

        /// <summary>
        /// Check whether all hidden dimensions in hiddenList are the same (except for 0),
        ///     and return the maximum among them.
        /// </summary>
        public static int CheckHiddenDimensionsAndReturnMax(int[] hiddenList)
        {
            var maxHidden = hiddenList.Max();
            if (!hiddenList.All(hidden => hidden == 0 || hidden == maxHidden))
            {
                throw new ArgumentException("all non-zero hidden dimensions should be the same.");
            }
            return maxHidden;
        }

        public const int NumLayerChoices = 30;

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Reliability", "CA2000:Dispose objects before losing scope",
            Justification = "The torch.nn.Module created in this method are meant to be alive out of the scope.")]
        public static Layer GetLayer(
            int layerIndex,
            double dropout,
            double attentionDropout,
            double activationDropout,
            string activationFn,
            bool addBiasKv,
            bool addZeroAttention,
            bool dynamicDropout)
        {
            return layerIndex switch
            {
                0 => new IdentityLayer(),
                1 => new SelfAttentionLayer(
                     embeddingDim: 128,
                     numAttentionHeads: 2,
                     dropoutRate: dropout,
                     attentionDropoutRate: attentionDropout,
                     addBiasKv: addBiasKv,
                     addZeroAttention: addZeroAttention),
                2 => new FeedForwardLayer(
                    embeddingDim: 128,
                    ffnEmbeddingDim: 128 * 4,
                    dropoutRate: dropout,
                    activationDropoutRate: activationDropout,
                    activationFn: activationFn,
                    dynamicDropout: dynamicDropout),
                3 => new EncConvLayer(
                    channel: 128,
                    kernelSize: 3,
                    dropoutRate: dropout,
                    activationDropoutRate: activationDropout,
                    activationFn: activationFn),
                4 => new EncConvLayer(
                    channel: 128,
                    kernelSize: 5,
                    dropoutRate: dropout,
                    activationDropoutRate: activationDropout,
                    activationFn: activationFn),
                5 => new EncConvLayer(
                    channel: 128,
                    kernelSize: 7,
                    dropoutRate: dropout,
                    activationDropoutRate: activationDropout,
                    activationFn: activationFn),
                6 => new IdentityLayer(),
                7 => new SelfAttentionLayer(
                    embeddingDim: 192,
                    numAttentionHeads: 3,
                    dropoutRate: dropout,
                    attentionDropoutRate: attentionDropout,
                    addBiasKv: addBiasKv,
                    addZeroAttention: addZeroAttention),
                8 => new FeedForwardLayer(
                    embeddingDim: 192,
                    ffnEmbeddingDim: 192 * 4,
                    dropoutRate: dropout,
                    activationDropoutRate: activationDropout,
                    activationFn: activationFn,
                    dynamicDropout: dynamicDropout),
                9 => new EncConvLayer(
                    channel: 192,
                    kernelSize: 3,
                    dropoutRate: dropout,
                    activationDropoutRate: activationDropout,
                    activationFn: activationFn),
                10 => new EncConvLayer(
                    channel: 192,
                    kernelSize: 5,
                    dropoutRate: dropout,
                    activationDropoutRate: activationDropout,
                    activationFn: activationFn),
                11 => new EncConvLayer(
                    channel: 192,
                    kernelSize: 7,
                    dropoutRate: dropout,
                    activationDropoutRate: activationDropout,
                    activationFn: activationFn),
                12 => new IdentityLayer(),
                13 => new SelfAttentionLayer(
                    embeddingDim: 384,
                    numAttentionHeads: 6,
                    dropoutRate: dropout,
                    attentionDropoutRate: attentionDropout,
                    addBiasKv: addBiasKv,
                    addZeroAttention: addZeroAttention),
                14 => new FeedForwardLayer(
                    embeddingDim: 384,
                    ffnEmbeddingDim: 384 * 4,
                    dropoutRate: dropout,
                    activationDropoutRate: activationDropout,
                    activationFn: activationFn,
                    dynamicDropout: dynamicDropout),
                15 => new EncConvLayer(
                    channel: 384,
                    kernelSize: 3,
                    dropoutRate: dropout,
                    activationDropoutRate: activationDropout,
                    activationFn: activationFn),
                16 => new EncConvLayer(
                    channel: 384,
                    kernelSize: 5,
                    dropoutRate: dropout,
                    activationDropoutRate: activationDropout,
                    activationFn: activationFn),
                17 => new EncConvLayer(
                    channel: 384,
                    kernelSize: 7,
                    dropoutRate: dropout,
                    activationDropoutRate: activationDropout,
                    activationFn: activationFn),
                18 => new IdentityLayer(),
                19 => new SelfAttentionLayer(
                    embeddingDim: 512,
                    numAttentionHeads: 8,
                    dropoutRate: dropout,
                    attentionDropoutRate: attentionDropout,
                    addBiasKv: addBiasKv,
                    addZeroAttention: addZeroAttention),
                20 => new FeedForwardLayer(
                    embeddingDim: 512,
                    ffnEmbeddingDim: 512 * 4,
                    dropoutRate: dropout,
                    activationDropoutRate: activationDropout,
                    activationFn: activationFn,
                    dynamicDropout: dynamicDropout),
                21 => new EncConvLayer(
                    channel: 512,
                    kernelSize: 3,
                    dropoutRate: dropout,
                    activationDropoutRate: activationDropout,
                    activationFn: activationFn),
                22 => new EncConvLayer(
                    channel: 512,
                    kernelSize: 5,
                    dropoutRate: dropout,
                    activationDropoutRate: activationDropout,
                    activationFn: activationFn),
                23 => new EncConvLayer(
                    channel: 512,
                    kernelSize: 7,
                    dropoutRate: dropout,
                    activationDropoutRate: activationDropout,
                    activationFn: activationFn),
                24 => new IdentityLayer(),
                25 => new SelfAttentionLayer(
                    embeddingDim: 768,
                    numAttentionHeads: 12,
                    dropoutRate: dropout,
                    attentionDropoutRate: attentionDropout,
                    addBiasKv: addBiasKv,
                    addZeroAttention: addZeroAttention),
                26 => new FeedForwardLayer(
                    embeddingDim: 768,
                    ffnEmbeddingDim: 768 * 4,
                    dropoutRate: dropout,
                    activationDropoutRate: activationDropout,
                    activationFn: activationFn,
                    dynamicDropout: dynamicDropout),
                27 => new EncConvLayer(
                    channel: 768,
                    kernelSize: 3,
                    dropoutRate: dropout,
                    activationDropoutRate: activationDropout,
                    activationFn: activationFn),
                28 => new EncConvLayer(
                    channel: 768,
                    kernelSize: 5,
                    dropoutRate: dropout,
                    activationDropoutRate: activationDropout,
                    activationFn: activationFn),
                29 => new EncConvLayer(
                    channel: 768,
                    kernelSize: 7,
                    dropoutRate: dropout,
                    activationDropoutRate: activationDropout,
                    activationFn: activationFn),
                _ => throw new NotSupportedException(
                    $"Unsupported layer index {layerIndex}. Expected to be within [0, {NumLayerChoices})."),
            };
        }
    }
}
