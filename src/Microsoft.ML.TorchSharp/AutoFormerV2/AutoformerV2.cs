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
    /// The object detection network based on AutoFormerV2 backbone which is pretrained only in MS COCO dataset.
    /// The network contains 3 scales with different parameters: small for 11MB, medium for 21MB and large for 41MB.
    /// </summary>
    public class AutoFormerV2 : Module<Tensor, (Tensor, Tensor, Tensor)>
    {

#pragma warning disable MSML_PrivateFieldName // Need to match TorchSharp model names.
        private readonly Device device;
        private readonly AutoFormerV2Backbone backbone;
        private readonly FPN neck;
        private readonly RetinaHead bbox_head;
        private readonly Anchors anchors;
#pragma warning restore MSML_PrivateFieldName

        /// <summary>
        /// Initializes a new instance of the <see cref="AutoFormerV2"/> class.
        /// </summary>
        /// <param name="numClasses">The number of object categories.</param>
        /// <param name="embedChannels">The embedding channels, which control the scale of model.</param>
        /// <param name="depths">The number of blocks, which control the scale of model.</param>
        /// <param name="numHeads">The number of heads, which control the scale of model.</param>
        /// <param name="device">The device where the model inference.</param>
        public AutoFormerV2(int numClasses, List<int> embedChannels, List<int> depths, List<int> numHeads, Device device = null)
            : base(nameof(AutoFormerV2))
        {
            this.device = device;

            this.backbone = new AutoFormerV2Backbone(embedChannels: embedChannels, depths: depths, numHeads: numHeads);
            this.neck = new FPN(inChannels: new List<int>() { embedChannels[1], embedChannels[2], embedChannels[3] });
            this.bbox_head = new RetinaHead(numClasses);
            this.anchors = new Anchors();

            this.RegisterComponents();
            this.InitializeWeight();
            this.FreezeBN();

            if (device != null)
            {
                this.to(device);
            }

            this.eval();
        }

        /// <summary>
        /// Freeze the weight of BatchNorm2d in network.
        /// </summary>
        public void FreezeBN()
        {
            foreach (var (name, layer) in this.named_modules())
            {
                if (layer.GetType() == typeof(BatchNorm2d))
                {
                    layer.eval();
                }
            }
        }

        /// <inheritdoc/>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public override (Tensor, Tensor, Tensor) forward(Tensor imgBatch)
        {
            using (var scope = torch.NewDisposeScope())
            {
                imgBatch = imgBatch.to(this.device);
                var xList = this.backbone.forward(imgBatch);
                var fpnOutput = this.neck.forward(xList);
                var (classificationInput, regressionInput) = this.bbox_head.forward(fpnOutput);
                var regression = torch.cat(regressionInput, dim: 1);
                var classification = torch.cat(classificationInput, dim: 1);
                var anchor = this.anchors.forward(imgBatch);

                return (classification.MoveToOuterDisposeScope(),
                        regression.MoveToOuterDisposeScope(),
                        anchor.MoveToOuterDisposeScope());
            }
        }

        /// <summary>
        /// Initialize weight of layers in network.
        /// </summary>
        private void InitializeWeight()
        {
            foreach (var (name, layer) in this.named_modules())
            {
                if (layer.GetType() == typeof(Linear))
                {
                    var module = layer as Linear;
                    var weightRequiresGrad = module.weight.requires_grad;
                    module.weight.requires_grad = false;
                    module.weight.normal_(0, 0.02);
                    module.weight.requires_grad = weightRequiresGrad;
                    var biasRequiresGrad = module.bias.requires_grad;
                    module.bias.requires_grad = false;
                    module.bias.zero_();
                    module.bias.requires_grad = biasRequiresGrad;
                }
                else if (layer.GetType() == typeof(LayerNorm))
                {
                    var module = layer as LayerNorm;
                    var weightRequiresGrad = module.weight.requires_grad;
                    module.weight.requires_grad = false;
                    module.weight.fill_(1);
                    module.weight.requires_grad = weightRequiresGrad;
                    var biasRequiresGrad = module.bias.requires_grad;
                    module.bias.requires_grad = false;
                    module.bias.zero_();
                    module.bias.requires_grad = biasRequiresGrad;
                }
                else if (layer.GetType() == typeof(BatchNorm2d))
                {
                    var module = layer as BatchNorm2d;
                    var weightRequiresGrad = module.weight.requires_grad;
                    module.weight.requires_grad = false;
                    module.weight.fill_(1.0);
                    module.weight.requires_grad = weightRequiresGrad;
                    var biasRequiresGrad = module.bias.requires_grad;
                    module.bias.requires_grad = false;
                    module.bias.zero_();
                    module.bias.requires_grad = biasRequiresGrad;
                }
            }
        }
    }
}
