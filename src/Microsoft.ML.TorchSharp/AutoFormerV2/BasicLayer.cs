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
    /// The basic layer.
    /// </summary>
    public class BasicLayer : Module<Tensor, int, int, (Tensor, int, int, Tensor, int, int)>
    {
        private readonly bool useShiftWindow;
        private readonly int windowSize;
        private readonly int shiftSize;
        private readonly ModuleList<AutoFormerV2Block> blocks;
        private readonly PatchMerging downsample;

        /// <summary>
        /// Initializes a new instance of the <see cref="BasicLayer"/> class.
        /// </summary>
        /// <param name="inChannels">The input channels.</param>
        /// <param name="outChannels">The output channels.</param>
        /// <param name="depth">The number of blocks.</param>
        /// <param name="numHeads">The number of heads.</param>
        /// <param name="windowSize">The size of window.</param>
        /// <param name="mlpRatio">The ratio of MLP.</param>
        /// <param name="dropRatio">The ratio of drop.</param>
        /// <param name="localConvSize">The size of local convolution.</param>
        /// <param name="useShiftWindow">Whether use shift window.</param>
        /// <param name="useInterpolate">Whether use interpolation.</param>
        public BasicLayer(int inChannels, int outChannels, int depth, int numHeads, int windowSize, double mlpRatio = 4.0, double dropRatio = 0, int localConvSize = 3, bool useShiftWindow = false, bool useInterpolate = false)
            : base(nameof(BasicLayer))
        {
            this.useShiftWindow = useShiftWindow;
            this.windowSize = windowSize;
            this.shiftSize = windowSize / 2;

            this.blocks = new ModuleList<AutoFormerV2Block>();
            for (int i = 0; i < depth; i++)
            {
                this.blocks.Add(new AutoFormerV2Block(inChannels: inChannels, numHeads: numHeads, windowSize: windowSize, shiftSize: (i % 2 == 0) ? 0 : (windowSize / 2), mlpRatio: mlpRatio, dropRatio: dropRatio, localConvSize: localConvSize, useShiftWindow: useShiftWindow, useInterpolate: useInterpolate));
            }

            this.downsample = new PatchMerging(inChannels: inChannels, outChannels: outChannels);
        }

        /// <inheritdoc/>
        public override (Tensor, int, int, Tensor, int, int) forward(Tensor x, int H, int W)
        {
            using (var scope = torch.NewDisposeScope())
            {
                Tensor attnMask;
                if (this.useShiftWindow)
                {
                    int Hp = (int)Math.Ceiling((double)H / this.windowSize) * this.windowSize;
                    int Wp = (int)Math.Ceiling((double)W / this.windowSize) * this.windowSize;
                    Tensor imgMask = torch.zeros(new long[] { 1, Hp, Wp, 1 }, device: x.device);
                    List<int> hSlicesStartAndEnd = new List<int>() { 0, Hp - this.windowSize, Hp - this.windowSize, Hp - this.shiftSize, Hp - this.shiftSize, Hp };
                    List<int> wSlicesStartAndEnd = new List<int>() { 0, Wp - this.windowSize, Wp - this.windowSize, Wp - this.shiftSize, Wp - this.shiftSize, Wp };
                    int cnt = 0;
                    for (int i = 0; i < hSlicesStartAndEnd.Count; i += 2)
                    {
                        for (int j = 0; j < wSlicesStartAndEnd.Count; j += 2)
                        {
                            int hStart = hSlicesStartAndEnd[i];
                            int hEnd = hSlicesStartAndEnd[i + 1];
                            int wStart = wSlicesStartAndEnd[j];
                            int wEnd = wSlicesStartAndEnd[j + 1];
                            for (int h = hStart; h < hEnd; h++)
                            {
                                for (int w = wStart; w < wEnd; w++)
                                {
                                    imgMask[0, h, w, 0] = cnt;
                                }
                            }

                            cnt += 1;
                        }
                    }

                    var maskWindows = WindowPartition(imgMask, this.windowSize);
                    maskWindows = maskWindows.view(-1, this.windowSize * this.windowSize);

                    attnMask = maskWindows.unsqueeze(1) - maskWindows.unsqueeze(2);
                    attnMask = attnMask.masked_fill(attnMask != 0, -100.0).masked_fill(attnMask == 0, 0.0);
                }
                else
                {
                    attnMask = null;
                }

                for (int i = 0; i < this.blocks.Count; i++)
                {
                    x = this.blocks[i].forward(x, H, W, attnMask);
                }

                var xOut = x;
                int nH = H;
                int nW = W;
                (x, nH, nW) = this.downsample.forward(x, H, W);

                return (xOut.MoveToOuterDisposeScope(), H, W, x.MoveToOuterDisposeScope(), nH, nW);
            }
        }

        /// <summary>
        /// Partition input to window size.
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <param name="window_size">The window size.</param>
        /// <returns>The output tensor.</returns>
        private static Tensor WindowPartition(Tensor x, int window_size)
        {
            using (var scope = torch.NewDisposeScope())
            {
                long B = x.shape[0];
                long H = x.shape[1];
                long W = x.shape[2];
                long C = x.shape[3];
                x = x.view(B, H / window_size, window_size, W / window_size, window_size, C);
                var windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C);

                return windows.MoveToOuterDisposeScope();
            }
        }
    }
}
