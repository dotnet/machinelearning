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
    /// The convolution and activation module.
    /// </summary>
    public class ConvModule : Module<Tensor, Tensor>
    {
#pragma warning disable MSML_PrivateFieldName // Need to match TorchSharp model names.
        private readonly Conv2d conv;
        private readonly ReLU activation;
        private readonly bool useRelu;
        private bool _disposedValue;
#pragma warning restore MSML_PrivateFieldName

        /// <summary>
        /// Initializes a new instance of the <see cref="ConvModule"/> class.
        /// </summary>
        /// <param name="inChannel">The input channels of convolution layer.</param>
        /// <param name="outChannel">The output channels of convolution layer.</param>
        /// <param name="kernelSize">The kernel size of convolution layer.</param>
        /// <param name="stride">The stride of convolution layer.</param>
        /// <param name="padding">The padding of convolution layer.</param>
        /// <param name="dilation">The dilation of convolution layer.</param>
        /// <param name="bias">The bias of convolution layer.</param>
        /// <param name="useRelu">Whether use Relu activation function.</param>
        public ConvModule(int inChannel, int outChannel, int kernelSize, int stride = 1, int padding = 0, int dilation = 1, bool bias = true, bool useRelu = true)
            : base(nameof(ConvModule))
        {
            this.conv = nn.Conv2d(in_channels: inChannel, out_channels: outChannel, kernel_size: kernelSize, stride: stride, padding: padding, dilation: dilation, bias: bias);
            this.useRelu = useRelu;
            if (this.useRelu)
            {
                this.activation = nn.ReLU();
            }
        }

        /// <inheritdoc/>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public override Tensor forward(Tensor x)
        {
            using (var scope = torch.NewDisposeScope())
            {
                x = this.conv.forward(x);
                if (this.useRelu)
                {
                    x = this.activation.forward(x);
                }

                return x.MoveToOuterDisposeScope();
            }
        }

        protected override void Dispose(bool disposing)
        {
            if (!_disposedValue)
            {
                if (disposing)
                {
                    conv.Dispose();
                    activation?.Dispose();
                    _disposedValue = true;
                }
            }

            base.Dispose(disposing);
        }
    }
}
