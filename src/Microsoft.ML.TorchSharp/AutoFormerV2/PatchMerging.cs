// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Microsoft.ML.TorchSharp.AutoFormerV2
{
    /// <summary>
    /// The PatchMerging layer.
    /// </summary>
    public class PatchMerging : Module<Tensor, int, int, (Tensor, int, int)>
    {
        private readonly GELU act;
        private readonly Conv2dBN conv1;
        private readonly Conv2dBN conv2;
        private readonly Conv2dBN conv3;

        /// <summary>
        /// Initializes a new instance of the <see cref="PatchMerging"/> class.
        /// </summary>
        /// <param name="inChannels">The input channels.</param>
        /// <param name="outChannels">The output channels.</param>
        public PatchMerging(int inChannels, int outChannels)
            : base(nameof(PatchMerging))
        {
            this.act = nn.GELU();
            this.conv1 = new Conv2dBN(inChannels, outChannels, 1, 1, 0);
            this.conv2 = new Conv2dBN(outChannels, outChannels, 3, 2, 1, groups: outChannels);
            this.conv3 = new Conv2dBN(outChannels, outChannels, 1, 1, 0);
        }

        /// <inheritdoc/>
        public override (Tensor, int, int) forward(Tensor x, int H, int W)
        {
            using (var scope = torch.NewDisposeScope())
            {
                if (x.shape.Length == 3)
                {
                    long B = x.shape[0];
                    x = x.view(B, H, W, -1).permute(0, 3, 1, 2);
                }

                x = this.conv1.forward(x);
                x = this.act.forward(x);
                x = this.conv2.forward(x);
                x = this.act.forward(x);
                x = this.conv3.forward(x);

                x = x.flatten(2).transpose(1, 2);
                H = (H + 1) / 2;
                W = (W + 1) / 2;

                return (x.MoveToOuterDisposeScope(), H, W);
            }
        }
    }
}
