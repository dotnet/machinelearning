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
    /// The PatchEmbed layer.
    /// </summary>
    public class PatchEmbed : Module<Tensor, Tensor>
    {
#pragma warning disable MSML_PrivateFieldName // Need to match TorchSharp model names.
        private readonly ModuleList<Module<Tensor, Tensor>> seq;
#pragma warning restore MSML_PrivateFieldName

        /// <summary>
        /// Initializes a new instance of the <see cref="PatchEmbed"/> class.
        /// </summary>
        /// <param name="inChannels">The input channels.</param>
        /// <param name="embedChannels">The output channels.</param>
        public PatchEmbed(int inChannels, int embedChannels)
            : base(nameof(PatchEmbed))
        {
            this.seq = ModuleList<Module<Tensor, Tensor>>();
            this.seq.Add(new Conv2dBN(inChannels, embedChannels / 2, 3, 2, 1));
            this.seq.Add(nn.GELU());
            this.seq.Add(new Conv2dBN(embedChannels / 2, embedChannels, 3, 2, 1));
        }

        /// <inheritdoc/>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public override Tensor forward(Tensor x)
        {
            using (var scope = torch.NewDisposeScope())
            {
                foreach (var submodule in this.seq)
                {
                    x = submodule.forward(x);
                }

                return x.MoveToOuterDisposeScope();
            }
        }
    }
}
