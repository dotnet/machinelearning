// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Microsoft.ML.TorchSharp.AutoFormerV2
{
    /// <summary>
    /// The backbone of AutoFormerV2 object detection network.
    /// </summary>
    public class AutoFormerV2Backbone : Module<Tensor, List<Tensor>>
    {
        private readonly List<int> outIndices;
        private readonly List<int> numFeatures;
        private readonly PatchEmbed patch_embed;
        private readonly ModuleList<Module<Tensor, int, int, (Tensor, int, int, Tensor, int, int)>> layers;
        private readonly LayerNorm norm1;
        private readonly LayerNorm norm2;
        private readonly LayerNorm norm3;

        /// <summary>
        /// Initializes a new instance of the <see cref="AutoFormerV2Backbone"/> class.
        /// </summary>
        /// <param name="inChannels">The input channels.</param>
        /// <param name="embedChannels">The embedding channels.</param>
        /// <param name="depths">The number of blocks in each layer.</param>
        /// <param name="numHeads">The number of heads in BasicLayer.</param>
        /// <param name="windowSizes">The sizes of window.</param>
        /// <param name="mlpRatio">The ratio of MLP.</param>
        /// <param name="dropRate">The ratio of drop.</param>
        /// <param name="mbconvExpandRatio">The expand ratio of MBConv.</param>
        /// <param name="outIndices">The indices of output.</param>
        /// <param name="useShiftWindow">Whether use shift window.</param>
        /// <param name="useInterpolate">Whether use interpolation.</param>
        /// <param name="outChannels">The channels of each outputs.</param>
        public AutoFormerV2Backbone(
                int inChannels = 3,
                List<int> embedChannels = null,
                List<int> depths = null,
                List<int> numHeads = null,
                List<int> windowSizes = null,
                double mlpRatio = 4.0,
                double dropRate = 0.0,
                double mbconvExpandRatio = 4.0,
                List<int> outIndices = null,
                bool useShiftWindow = true,
                bool useInterpolate = false,
                List<int> outChannels = null)
            : base(nameof(AutoFormerV2Backbone))
        {
            embedChannels ??= new List<int>() { 96, 192, 384, 576 };
            depths ??= new List<int>() { 2, 2, 6, 2 };
            numHeads ??= new List<int>() { 3, 6, 12, 18 };
            windowSizes ??= new List<int>() { 7, 7, 14, 7 };
            outIndices ??= new List<int>() { 1, 2, 3 };
            outChannels ??= embedChannels;

            this.outIndices = outIndices;
            this.numFeatures = outChannels;

            this.patch_embed = new PatchEmbed(inChannels: inChannels, embedChannels: embedChannels[0]);

            var dpr = new List<double>();
            int depthSum = 0;
            foreach (int depth in depths)
            {
                depthSum += depth;
            }

            for (int i = 0; i < depthSum; i++)
            {
                dpr.Add(0.0); // different from original AutoFormer, but ok with current model
            }

            this.layers = new ModuleList<Module<Tensor, int, int, (Tensor, int, int, Tensor, int, int)>>();
            this.layers.Add(new ConvLayer(
                inChannels: embedChannels[0],
                outChannels: embedChannels[1],
                depth: depths[0],
                convExpandRatio: mbconvExpandRatio));
            for (int iLayer = 1; iLayer < depths.Count; iLayer++)
            {
                this.layers.Add(new BasicLayer(
                    inChannels: embedChannels[iLayer],
                    outChannels: embedChannels[Math.Min(iLayer + 1, embedChannels.Count - 1)],
                    depth: depths[iLayer],
                    numHeads: numHeads[iLayer],
                    windowSize: windowSizes[iLayer],
                    mlpRatio: mlpRatio,
                    dropRatio: dropRate,
                    localConvSize: 3,
                    useShiftWindow: useShiftWindow,
                    useInterpolate: useInterpolate));
            }

            this.norm1 = nn.LayerNorm(new long[] { outChannels[1] });
            this.norm2 = nn.LayerNorm(new long[] { outChannels[2] });
            this.norm3 = nn.LayerNorm(new long[] { outChannels[3] });
        }

        /// <inheritdoc/>
        public override List<Tensor> forward(Tensor img_batch)
        {
            using (var scope = torch.NewDisposeScope())
            {
                var x = this.patch_embed.forward(img_batch);
                var B = (int)x.shape[0];
                var C = (int)x.shape[1];
                var Wh = (int)x.shape[2];
                var Ww = (int)x.shape[3];
                var outs = new List<Tensor>();
                Tensor xOut;
                int H, W;
                (xOut, H, W, x, Wh, Ww) = this.layers[0].forward(x, Wh, Ww);

                for (int iLayer = 1; iLayer < this.layers.Count; iLayer++)
                {

                    (xOut, H, W, x, Wh, Ww) = this.layers[iLayer].forward(x, Wh, Ww);

                    if (this.outIndices.Contains(iLayer))
                    {
                        switch (iLayer)
                        {
                            case 1:
                                xOut = this.norm1.forward(xOut);
                                break;
                            case 2:
                                xOut = this.norm2.forward(xOut);
                                break;
                            case 3:
                                xOut = this.norm3.forward(xOut);
                                break;
                            default:
                                break;
                        }

                        long N = xOut.shape[0];
                        var @out = xOut.view(N, H, W, this.numFeatures[iLayer]).permute(0, 3, 1, 2).contiguous();
                        @out = @out.MoveToOuterDisposeScope();
                        outs.Add(@out);
                    }
                }

                return outs;
            }
        }
    }
}
