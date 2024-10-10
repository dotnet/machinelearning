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
#pragma warning disable MSML_PrivateFieldName // Need to match TorchSharp model names.
        private readonly int windowSize;
        private readonly int shiftSize;
        private readonly bool useShiftWindow;
        private readonly bool useInterpolate;
        private readonly Attention attn;
        private readonly MLP mlp;
        private readonly Conv2dBN local_conv;
        private bool _disposedValue;
#pragma warning restore MSML_PrivateFieldName

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
            List<int> windowResolution = new List<int>() { windowSize, windowSize };
            this.attn = new Attention(inChannels, headChannels, numHeads, attnRatio: 1, windowResolution: windowResolution);

            int mlpHiddenChannels = (int)(inChannels * mlpRatio);
            this.mlp = new MLP(inFeatures: inChannels, hiddenFeatures: mlpHiddenChannels, dropRatio: dropRatio);

            int padding = localConvSize / 2;
            this.local_conv = new Conv2dBN(inChannels, inChannels, kernalSize: localConvSize, stride: 1, padding: padding, groups: inChannels);
        }

        /// <inheritdoc/>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public override Tensor forward(Tensor x, int h, int w, Tensor maskMatrix)
        {
            using (var scope = torch.NewDisposeScope())
            {
                long b = x.shape[0];
                long l = x.shape[1];
                long c = x.shape[2];
                var resX = x;
                x = x.view(b, h, w, c);
                int padB = (this.windowSize - (h % this.windowSize)) % this.windowSize;
                int padR = (this.windowSize - (w % this.windowSize)) % this.windowSize;
                bool padding = false;
                if (padB > 0 || padR > 0)
                {
                    padding = true;
                }

                int pH = h + padB;
                int pW = w + padR;
                if (padding)
                {
                    x = nn.functional.pad(x, new long[] { 0, 0, 0, padR, 0, padB });
                }

                Tensor shiftedX;
                Tensor attnMask;
                if (this.useShiftWindow && this.shiftSize > 0)
                {
                    shiftedX = torch.roll(x, shifts: new long[] { -this.shiftSize, -this.shiftSize }, dims: new long[] { 1, 2 });
                    attnMask = maskMatrix;
                }
                else
                {
                    shiftedX = x;
                    attnMask = null;
                }

                var xWindows = WindowPartition(shiftedX, this.windowSize);
                xWindows = xWindows.view(-1, this.windowSize * this.windowSize, c);
                var attnWindows = this.attn.forward(xWindows, mask: attnMask);

                attnWindows = attnWindows.view(-1, this.windowSize, this.windowSize, c);
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
                        x = nn.functional.interpolate(x.permute(0, 3, 1, 2), size: new long[] { h, w }, mode: torch.InterpolationMode.Bilinear, align_corners: true).permute(0, 2, 3, 1);
                    }
                    else
                    {
                        x = x[RangeUtil.ToTensorIndex(..), RangeUtil.ToTensorIndex(..h), RangeUtil.ToTensorIndex(..w)].contiguous();
                    }
                }

                x = x.view(b, l, c);

                x = resX + x;
                x = x.transpose(1, 2).reshape(b, c, h, w);
                x = this.local_conv.forward(x);
                x = x.view(b, c, l).transpose(1, 2);
                x = x + this.mlp.forward(x);

                return x.MoveToOuterDisposeScope();
            }
        }

        /// <summary>
        /// Reverse input in window size to original shape.
        /// </summary>
        /// <param name="windows">The input window tensor.</param>
        /// <param name="windowSize">The size of window.</param>
        /// <param name="h">The height.</param>
        /// <param name="w">The width.</param>
        /// <returns>The reversed window tensor.</returns>
        private static Tensor WindowsReverse(Tensor windows, int windowSize, int h, int w)
        {
            using (var scope = torch.NewDisposeScope())
            {
                int b = (int)windows.shape[0] / (h * w / windowSize / windowSize);
                var x = windows.view(b, h / windowSize, w / windowSize, windowSize, windowSize, -1);
                x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1);

                return x.MoveToOuterDisposeScope();
            }
        }

        /// <summary>
        /// Partition input to window size.
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <param name="windowSize">The size of window.</param>
        /// <returns>The partition window.</returns>
        private static Tensor WindowPartition(Tensor x, int windowSize)
        {
            using (var scope = torch.NewDisposeScope())
            {
                long b = x.shape[0];
                long h = x.shape[1];
                long w = x.shape[2];
                long c = x.shape[3];
                x = x.view(b, h / windowSize, windowSize, w / windowSize, windowSize, c);
                var windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, windowSize, windowSize, c);

                return windows.MoveToOuterDisposeScope();
            }
        }

        protected override void Dispose(bool disposing)
        {
            if (!_disposedValue)
            {
                if (disposing)
                {
                    attn.Dispose();
                    mlp.Dispose();
                    local_conv.Dispose();
                    _disposedValue = true;
                }
            }

            base.Dispose(disposing);
        }
    }
}
