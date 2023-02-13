// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Microsoft.ML.TorchSharp.AutoFormerV2
{
    /// <summary>
    /// The block module of AutoFormer network.
    /// </summary>
    public class AutoFormerV2Block : Module<Tensor, int, int, Tensor, Tensor>
    {
        private readonly int windowSize;
        private readonly int shiftSize;
        private readonly bool useShiftWindow;
        private readonly bool useInterpolate;
        private readonly Attention attn;
        private readonly MLP mlp;
        private readonly Conv2dBN local_conv;

        /// <summary>
        /// Initializes a new instance of the <see cref="AutoFormerV2Block"/> class.
        /// </summary>
        /// <param name="inChannels">The input channels.</param>
        /// <param name="numHeads">The number of blocks.</param>
        /// <param name="windowSize">The size of window.</param>
        /// <param name="shiftSize">The size of shift.</param>
        /// <param name="mlpRatio">The ratio of MLP.</param>
        /// <param name="dropRatio">The ratio of drop.</param>
        /// <param name="localConvSize">The size of local convolution.</param>
        /// <param name="useShiftWindow">Whether use shift window.</param>
        /// <param name="useInterpolate">Whether use interpolation.</param>
        public AutoFormerV2Block(int inChannels, int numHeads, int windowSize = 7, int shiftSize = 0, double mlpRatio = 4.0, double dropRatio = 0, int localConvSize = 3, bool useShiftWindow = false, bool useInterpolate = false)
            : base(nameof(AutoFormerV2Block))
        {
            this.windowSize = windowSize;
            if (useShiftWindow)
            {
                this.shiftSize = shiftSize;
            }
            else
            {
                this.shiftSize = 0;
            }

            this.useShiftWindow = useShiftWindow;
            this.useInterpolate = useInterpolate;

            int headChannels = inChannels / numHeads;
            List<int> window_resolution = new List<int>() { windowSize, windowSize };
            this.attn = new Attention(inChannels, headChannels, numHeads, attnRatio: 1, windowResolution: window_resolution);

            int mlpHiddenChannels = (int)(inChannels * mlpRatio);
            this.mlp = new MLP(inFeatures: inChannels, hiddenFeatures: mlpHiddenChannels, dropRatio: dropRatio);

            int padding = localConvSize / 2;
            this.local_conv = new Conv2dBN(inChannels, inChannels, kernalSize: localConvSize, stride: 1, padding: padding, groups: inChannels);
        }

        /// <inheritdoc/>
        public override Tensor forward(Tensor x, int H, int W, Tensor mask_matrix)
        {
            using (var scope = torch.NewDisposeScope())
            {
                long B = x.shape[0];
                long L = x.shape[1];
                long C = x.shape[2];
                var resX = x;
                x = x.view(B, H, W, C);
                int padB = (this.windowSize - (H % this.windowSize)) % this.windowSize;
                int padR = (this.windowSize - (W % this.windowSize)) % this.windowSize;
                bool padding = false;
                if (padB > 0 || padR > 0)
                {
                    padding = true;
                }

                int pH = H + padB;
                int pW = W + padR;
                if (padding)
                {
                    x = nn.functional.pad(x, new long[] { 0, 0, 0, padR, 0, padB });
                }

                Tensor shiftedX, attnMask;
                if (this.useShiftWindow && this.shiftSize > 0)
                {
                    shiftedX = torch.roll(x, shifts: new long[] { -this.shiftSize, -this.shiftSize }, dims: new long[] { 1, 2 });
                    attnMask = mask_matrix;
                }
                else
                {
                    shiftedX = x;
                    attnMask = null;
                }

                var xWindows = WindowPartition(shiftedX, this.windowSize);
                xWindows = xWindows.view(-1, this.windowSize * this.windowSize, C);
                var attnWindows = this.attn.forward(xWindows, mask: attnMask);

                attnWindows = attnWindows.view(-1, this.windowSize, this.windowSize, C);
                shiftedX = WindowsReverse(attnWindows, this.windowSize, pH, pW);

                if (this.useShiftWindow && this.shiftSize > 0)
                {
                    x = torch.roll(shiftedX, shifts: new long[] { this.shiftSize, this.shiftSize }, dims: new long[] { 1, 2 });
                }
                else
                {
                    x = shiftedX;
                }

                if (padding)
                {
                    if (this.useInterpolate)
                    {
                        x = nn.functional.interpolate(x.permute(0, 3, 1, 2), size: new long[] { H, W }, mode: torch.InterpolationMode.Bilinear, align_corners: true).permute(0, 2, 3, 1);
                    }
                    else
                    {
                        x = x[.., ..H, ..W].contiguous();
                    }
                }

                x = x.view(B, L, C);

                x = resX + x;
                x = x.transpose(1, 2).reshape(B, C, H, W);
                x = this.local_conv.forward(x);
                x = x.view(B, C, L).transpose(1, 2);
                x = x + this.mlp.forward(x);

                return x.MoveToOuterDisposeScope();
            }
        }

        /// <summary>
        /// Reverse input in window size to original shape.
        /// </summary>
        /// <param name="windows">The input window tensor.</param>
        /// <param name="window_size">The size of window.</param>
        /// <param name="H">The height.</param>
        /// <param name="W">The width.</param>
        /// <returns>The reversed window tensor.</returns>
        private static Tensor WindowsReverse(Tensor windows, int window_size, int H, int W)
        {
            using (var scope = torch.NewDisposeScope())
            {
                int B = (int)windows.shape[0] / (H * W / window_size / window_size);
                var x = windows.view(B, H / window_size, W / window_size, window_size, window_size, -1);
                x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1);

                return x.MoveToOuterDisposeScope();
            }
        }

        /// <summary>
        /// Partition input to window size.
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <param name="window_size">The size of window.</param>
        /// <returns>The partition window.</returns>
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
