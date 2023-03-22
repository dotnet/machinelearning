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
    /// The MBConv layer.
    /// </summary>
    public class MBConv : Module<Tensor, Tensor>
    {
#pragma warning disable MSML_PrivateFieldName // Need to match TorchSharp model names.
        private readonly Conv2dBN conv1;
        private readonly GELU act1;
        private readonly Conv2dBN conv2;
        private readonly GELU act2;
        private readonly Conv2dBN conv3;
        private readonly GELU act3;
#pragma warning restore MSML_PrivateFieldName

        /// <summary>
        /// Initializes a new instance of the <see cref="MBConv"/> class.
        /// </summary>
        /// <param name="inChannels">The input channels.</param>
        /// <param name="outChannels">The output channels.</param>
        /// <param name="expandRatio">The expand ratio.</param>
        public MBConv(int inChannels, int outChannels, double expandRatio)
            : base(nameof(MBConv))
        {
            int hiddenChans = (int)(inChannels * expandRatio);
            this.conv1 = new Conv2dBN(inChannels, hiddenChans, kernalSize: 1);
            this.act1 = nn.GELU();
            this.conv2 = new Conv2dBN(hiddenChans, hiddenChans, kernalSize: 3, stride: 1, padding: 1, groups: hiddenChans);
            this.act2 = nn.GELU();
            this.conv3 = new Conv2dBN(hiddenChans, outChannels, kernalSize: 1);
            this.act3 = nn.GELU();
        }

        /// <inheritdoc/>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public override Tensor forward(Tensor x0)
        {
            using (var scope = torch.NewDisposeScope())
            {
                var shortcut = x0;
                var x = this.conv1.forward(x0);
                x = this.act1.forward(x);
                x = this.conv2.forward(x);
                x = this.act2.forward(x);
                x = this.conv3.forward(x);
                x += shortcut;
                x = this.act3.forward(x);

                return x.MoveToOuterDisposeScope();
            }
        }
    }
}
