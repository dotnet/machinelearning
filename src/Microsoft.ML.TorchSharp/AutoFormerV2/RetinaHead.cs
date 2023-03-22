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
    /// The head of RetinaNet.
    /// </summary>
    public class RetinaHead : Module<List<Tensor>, (List<Tensor>, List<Tensor>)>
    {
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_PrivateFieldName:private field names not in _camelCase format", Justification = "Need to match TorchSharp.")]
        private readonly ModuleList<Module<Tensor, Tensor>> cls_convs;

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_PrivateFieldName:private field names not in _camelCase format", Justification = "Need to match TorchSharp.")]
        private readonly ModuleList<Module<Tensor, Tensor>> reg_convs;

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_PrivateFieldName:private field names not in _camelCase format", Justification = "Need to match TorchSharp.")]
        private readonly Conv2d retina_cls;

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_PrivateFieldName:private field names not in _camelCase format", Justification = "Need to match TorchSharp.")]
        private readonly Conv2d retina_reg;

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_PrivateFieldName:private field names not in _camelCase format", Justification = "Need to match TorchSharp.")]
        private readonly Sigmoid output_act;

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_PrivateFieldName:private field names not in _camelCase format", Justification = "Need to match TorchSharp.")]
        private readonly int numClasses;

        /// <summary>
        /// Initializes a new instance of the <see cref="RetinaHead"/> class.
        /// </summary>
        /// <param name="numClasses">The number of classes.</param>
        /// <param name="inChannels">The input channels.</param>
        /// <param name="stackedConvs">The number of stacked convolution layers.</param>
        /// <param name="featChannels">The feature channels.</param>
        /// <param name="numBasePriors">The number of base priors.</param>
        public RetinaHead(int numClasses, int inChannels = 256, int stackedConvs = 4, int featChannels = 256, int numBasePriors = 9)
            : base(nameof(RetinaHead))
        {
            this.numClasses = numClasses;
            this.cls_convs = new ModuleList<Module<Tensor, Tensor>>();
            this.reg_convs = new ModuleList<Module<Tensor, Tensor>>();
            for (int i = 0; i < stackedConvs; i++)
            {
                int chn = i == 0 ? inChannels : featChannels;
                this.cls_convs.Add(new ConvModule(chn, featChannels, 3, stride: 1, padding: 1, useRelu: true));
                this.reg_convs.Add(new ConvModule(chn, featChannels, 3, stride: 1, padding: 1, useRelu: true));
            }

            this.retina_cls = Conv2d(featChannels, numBasePriors * numClasses, 3, padding: 1);
            this.retina_reg = Conv2d(featChannels, numBasePriors * 4, 3, padding: 1);
            this.output_act = nn.Sigmoid();
        }

        /// <inheritdoc/>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public override (List<Tensor>, List<Tensor>) forward(List<Tensor> inputs)
        {
            using (var scope = torch.NewDisposeScope())
            {
                var clsOutputs = new List<Tensor>();
                var regOutputs = new List<Tensor>();
                for (int i = 0; i < inputs.Count; i++)
                {
                    var clsOutput = inputs[i];
                    for (int j = 0; j < this.cls_convs.Count; j++)
                    {
                        clsOutput = this.cls_convs[j].forward(clsOutput);
                    }

                    clsOutput = this.retina_cls.forward(clsOutput);
                    clsOutput = this.output_act.forward(clsOutput);

                    // out is B x C x W x H, with C = num_classes * num_anchors
                    clsOutput = clsOutput.permute(0, 2, 3, 1);
                    clsOutput = clsOutput.contiguous().view(clsOutput.shape[0], -1, this.numClasses);
                    clsOutputs.Add(clsOutput.MoveToOuterDisposeScope());

                    var regOutput = inputs[i];
                    for (int j = 0; j < this.reg_convs.Count; j++)
                    {
                        regOutput = this.reg_convs[j].forward(regOutput);
                    }

                    regOutput = this.retina_reg.forward(regOutput);

                    // out is B x C x W x H, with C = 4*num_anchors
                    regOutput = regOutput.permute(0, 2, 3, 1);
                    regOutput = regOutput.contiguous().view(regOutput.shape[0], -1, 4);
                    regOutputs.Add(regOutput.MoveToOuterDisposeScope());
                }

                return (clsOutputs, regOutputs);
            }
        }
    }
}
