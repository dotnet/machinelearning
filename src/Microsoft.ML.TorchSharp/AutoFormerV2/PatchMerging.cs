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
#pragma warning disable MSML_PrivateFieldName // Need to match TorchSharp model names.
        private readonly GELU act;
        private readonly Conv2dBN conv1;
        private readonly Conv2dBN conv2;
        private readonly Conv2dBN conv3;
#pragma warning restore MSML_PrivateFieldName

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
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public override (Tensor, int, int) forward(Tensor x, int h, int w)
        {
            using (var scope = torch.NewDisposeScope())
            {
                if (x.shape.Length == 3)
                {
                    long b = x.shape[0];
                    x = x.view(b, h, w, -1).permute(0, 3, 1, 2);
                }

                x = this.conv1.forward(x);
                x = this.act.forward(x);
                x = this.conv2.forward(x);
                x = this.act.forward(x);
                x = this.conv3.forward(x);

                x = x.flatten(2).transpose(1, 2);
                h = (h + 1) / 2;
                w = (w + 1) / 2;

                return (x.MoveToOuterDisposeScope(), h, w);
            }
        }
    }
}
