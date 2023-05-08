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
    /// The MLP (Multilayer Perceptron) layer.
    /// </summary>
    [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
    public class MLP : Module<Tensor, Tensor>
    {
#pragma warning disable MSML_PrivateFieldName // Need to match TorchSharp model names.
        private readonly LayerNorm norm;
        private readonly Linear fc1;
        private readonly Linear fc2;
        private readonly GELU act;
        private readonly Dropout drop;
#pragma warning restore MSML_PrivateFieldName

        /// <summary>
        /// Initializes a new instance of the <see cref="MLP"/> class.
        /// </summary>
        /// <param name="inFeatures">The input channels of features.</param>
        /// <param name="hiddenFeatures">The hidden layer channels of features.</param>
        /// <param name="outFeatures">The output channels of features.</param>
        /// <param name="dropRatio">The drop ratio.</param>
        public MLP(int inFeatures, int? hiddenFeatures = null, int? outFeatures = null, double dropRatio = 0)
            : base(nameof(MLP))
        {
            outFeatures ??= inFeatures;
            hiddenFeatures ??= inFeatures;
            this.norm = nn.LayerNorm(new long[] { inFeatures });
            this.fc1 = nn.Linear(inFeatures, (long)hiddenFeatures);
            this.fc2 = nn.Linear((long)hiddenFeatures, (long)outFeatures);
            this.act = nn.GELU();
            this.drop = nn.Dropout(dropRatio);
        }

        /// <inheritdoc/>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public override Tensor forward(Tensor x)
        {
            using (var scope = torch.NewDisposeScope())
            {
                x = this.norm.forward(x);
                x = this.fc1.forward(x);
                x = this.act.forward(x);
                x = this.drop.forward(x);
                x = this.fc2.forward(x);
                x = this.drop.forward(x);

                return x.MoveToOuterDisposeScope();
            }
        }
    }
}
