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
    /// The Convolution and BN module.
    /// </summary>
    public class Conv2dBN : Module<Tensor, Tensor>
    {
#pragma warning disable MSML_PrivateFieldName // Need to match TorchSharp model names.
        private readonly Conv2d c;
        private readonly BatchNorm2d bn;
#pragma warning restore MSML_PrivateFieldName

        /// <summary>
        /// Initializes a new instance of the <see cref="Conv2dBN"/> class.
        /// </summary>
        /// <param name="inChannels">The input channels.</param>
        /// <param name="outChannels">The output channels.</param>
        /// <param name="kernalSize">The kernel size of convolution layer.</param>
        /// <param name="stride">The stride of convolution layer.</param>
        /// <param name="padding">The padding of convolution layer.</param>
        /// <param name="dilation">The dilation of convolution layer.</param>
        /// <param name="groups">The groups of convolution layer.</param>
        public Conv2dBN(int inChannels, int outChannels, int kernalSize = 1, int stride = 1, int padding = 0, int dilation = 1, int groups = 1)
            : base(nameof(Conv2dBN))
        {
            this.c = nn.Conv2d(inChannels, outChannels, kernalSize, stride, padding, dilation, groups: groups, bias: false);
            this.bn = nn.BatchNorm2d(outChannels);
        }

        /// <inheritdoc/>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public override Tensor forward(Tensor x)
        {
            using (var scope = torch.NewDisposeScope())
            {
                var x1 = this.c.forward(x);
                var x2 = this.bn.forward(x1);

                return x2.MoveToOuterDisposeScope();
            }
        }
    }
}
