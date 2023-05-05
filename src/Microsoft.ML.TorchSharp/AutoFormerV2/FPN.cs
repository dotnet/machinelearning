// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Microsoft.ML.TorchSharp.AutoFormerV2
{
    /// <summary>
    /// The FPN (Feature Pyramid Networks) layer.
    /// </summary>
    [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
    public class FPN : Module<List<Tensor>, List<Tensor>>
    {
#pragma warning disable MSML_PrivateFieldName // Need to match TorchSharp model names.
        private readonly ModuleList<Module<Tensor, Tensor>> lateral_convs;
        private readonly ModuleList<Module<Tensor, Tensor>> fpn_convs;
        private readonly int numOuts;
#pragma warning restore MSML_PrivateFieldName

        /// <summary>
        /// Initializes a new instance of the <see cref="FPN"/> class.
        /// </summary>
        /// <param name="inChannels">The input channels.</param>
        /// <param name="outChannel">The output channels.</param>
        /// <param name="numOuts">The number of output tensors.</param>
        public FPN(List<int> inChannels = null, int outChannel = 256, int numOuts = 5)
            : base(nameof(FPN))
        {
            inChannels ??= new List<int>() { 192, 384, 576 };
            this.numOuts = numOuts;
            int startLevel = 0;
            int backboneEndLevel = 3;
            this.lateral_convs = new ModuleList<Module<Tensor, Tensor>>();
            this.fpn_convs = new ModuleList<Module<Tensor, Tensor>>();
            for (int i = startLevel; i < backboneEndLevel; i++)
            {
                this.lateral_convs.Add(new ConvModule(inChannels[i], outChannel, 1, useRelu: false));
                this.fpn_convs.Add(new ConvModule(outChannel, outChannel, 3, padding: 1, useRelu: false));
            }

            int extraLevel = 2;
            for (int i = 0; i < extraLevel; i++)
            {
                int inChannel;
                if (i == 0)
                {
                    inChannel = inChannels[backboneEndLevel - 1];
                }
                else
                {
                    inChannel = outChannel;
                }

                this.fpn_convs.Add(new ConvModule(inChannel, outChannel, 3, stride: 2, padding: 1, useRelu: false));
            }
        }

        /// <inheritdoc/>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public override List<Tensor> forward(List<Tensor> inputs)
        {
            using (var scope = torch.NewDisposeScope())
            {
                int usedBackboneLevels = this.lateral_convs.Count;
                var laterals = new List<Tensor>();
                for (int i = 0; i < usedBackboneLevels; i++)
                {
                    laterals.Add(this.lateral_convs[i].forward(inputs[i]));
                }

                for (int i = usedBackboneLevels - 1; i > 0; i--)
                {
                    var prevShape = new long[] { laterals[i - 1].shape[2], laterals[i - 1].shape[3] };
                    laterals[i - 1] = laterals[i - 1] + nn.functional.interpolate(laterals[i], prevShape);
                }

                var outs = new List<Tensor>();
                for (int i = 0; i < usedBackboneLevels; i++)
                {
                    outs.Add(this.fpn_convs[i].forward(laterals[i]));
                }

                var extraSource = inputs[usedBackboneLevels - 1];
                outs.Add(this.fpn_convs[usedBackboneLevels].forward(extraSource));
                for (int i = usedBackboneLevels + 1; i < this.numOuts; i++)
                {
                    outs.Add(this.fpn_convs[i].forward(outs[outs.Count - 1]));
                }

                for (int i = 0; i < outs.Count; i++)
                {
                    outs[i] = outs[i].MoveToOuterDisposeScope();
                }

                return outs;
            }
        }
    }
}
