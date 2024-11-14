// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Microsoft.ML.TorchSharp.AutoFormerV2
{
    /// <summary>
    /// Anchor boxes are a set of predefined bounding boxes of a certain height and width, whose location and size can be adjusted by the regression head of model.
    /// </summary>
    public class Anchors : Module<Tensor, Tensor>
    {
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_PrivateFieldName:private field names not in _camelCase format", Justification = "Need to match TorchSharp.")]
        private readonly int[] pyramidLevels;

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_PrivateFieldName:private field names not in _camelCase format", Justification = "Need to match TorchSharp.")]
        private readonly int[] strides;

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_PrivateFieldName:private field names not in _camelCase format", Justification = "Need to match TorchSharp.")]
        private readonly int[] sizes;

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_PrivateFieldName:private field names not in _camelCase format", Justification = "Need to match TorchSharp.")]
        private readonly double[] ratios;

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_PrivateFieldName:private field names not in _camelCase format", Justification = "Need to match TorchSharp.")]
        private readonly double[] scales;

        /// <summary>
        /// Initializes a new instance of the <see cref="Anchors"/> class.
        /// </summary>
        /// <param name="pyramidLevels">Pyramid levels.</param>
        /// <param name="strides">Strides between adjacent bboxes.</param>
        /// <param name="sizes">Different sizes for bboxes.</param>
        /// <param name="ratios">Different ratios for height/width.</param>
        /// <param name="scales">Scale size of bboxes.</param>
        public Anchors(int[] pyramidLevels = null, int[] strides = null, int[] sizes = null, double[] ratios = null, double[] scales = null)
            : base(nameof(Anchors))
        {
            this.pyramidLevels = pyramidLevels != null ? pyramidLevels : new int[] { 3, 4, 5, 6, 7 };
            this.strides = strides != null ? strides : this.pyramidLevels.Select(x => (int)Math.Pow(2, x)).ToArray();
            this.sizes = sizes != null ? sizes : this.pyramidLevels.Select(x => (int)Math.Pow(2, x + 2)).ToArray();
            this.ratios = ratios != null ? ratios : new double[] { 0.5, 1, 2 };
            this.scales = scales != null ? scales : new double[] { Math.Pow(2, 0), Math.Pow(2, 1.0 / 3.0), Math.Pow(2, 2.0 / 3.0) };
        }

        /// <summary>
        /// Generate anchors for an image.
        /// </summary>
        /// <param name="image">Image in Tensor format.</param>
        /// <returns>All anchors.</returns>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public override Tensor forward(Tensor image)
        {
            using (var scope = torch.NewDisposeScope())
            {
                var imageShape = torch.tensor(image.shape.AsSpan().Slice(2).ToArray());

                // compute anchors over all pyramid levels
                var allAnchors = torch.zeros(new long[] { 0, 4 }, dtype: torch.float32);

                for (int idx = 0; idx < this.pyramidLevels.Length; ++idx)
                {
                    var x = this.pyramidLevels[idx];
                    var shape = ((imageShape + Math.Pow(2, x) - 1) / Math.Pow(2, x)).to_type(torch.int32);
                    var anchors = GenerateAnchors(
                        baseSize: this.sizes[idx],
                        ratios: this.ratios,
                        scales: this.scales);
                    var shiftedAnchors = Shift(shape, this.strides[idx], anchors);
                    allAnchors = torch.cat(new List<Tensor>() { allAnchors, shiftedAnchors }, dim: 0);
                }

                var output = allAnchors.unsqueeze(dim: 0);
                output = output.to(image.device);

                return output.MoveToOuterDisposeScope();
            }
        }

        /// <summary>
        /// Generate a set of anchors given size, ratios and scales.
        /// </summary>
        /// <param name="baseSize">Base size for width and height.</param>
        /// <param name="ratios">Ratios for height/width.</param>
        /// <param name="scales">Scales to resize base size.</param>
        /// <returns>A set of anchors.</returns>
        private static Tensor GenerateAnchors(int baseSize = 16, double[] ratios = null, double[] scales = null)
        {
            using (var anchorsScope = torch.NewDisposeScope())
            {
                ratios ??= new double[] { 0.5, 1, 2 };
                scales ??= new double[] { Math.Pow(2, 0), Math.Pow(2, 1.0 / 3.0), Math.Pow(2, 2.0 / 3.0) };

                var numAnchors = ratios.Length * scales.Length;

                // initialize output anchors
                var anchors = torch.zeros(new long[] { numAnchors, 4 }, dtype: torch.float32);

                // scale base_size
                anchors[RangeUtil.ToTensorIndex(..), RangeUtil.ToTensorIndex(2..)] = baseSize * torch.tile(scales, new long[] { 2, ratios.Length }).transpose(1, 0);

                // compute areas of anchors
                var areas = torch.mul(anchors[RangeUtil.ToTensorIndex(..), 2], anchors[RangeUtil.ToTensorIndex(..), 3]);

                // correct for ratios
                anchors[RangeUtil.ToTensorIndex(..), 2] = torch.sqrt(areas / torch.repeat_interleave(ratios, new long[] { scales.Length }));
                anchors[RangeUtil.ToTensorIndex(..), 3] = torch.mul(anchors[RangeUtil.ToTensorIndex(..), 2], torch.repeat_interleave(ratios, new long[] { scales.Length }));

                // transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
                anchors[RangeUtil.ToTensorIndex(..), torch.TensorIndex.Tensor(torch.tensor(new long[] { 0, 2 }, dtype: torch.int64))] -= torch.tile(anchors[RangeUtil.ToTensorIndex(..), 2] * 0.5, new long[] { 2, 1 }).T;
                anchors[RangeUtil.ToTensorIndex(..), torch.TensorIndex.Tensor(torch.tensor(new long[] { 1, 3 }, dtype: torch.int64))] -= torch.tile(anchors[RangeUtil.ToTensorIndex(..), 3] * 0.5, new long[] { 2, 1 }).T;

                return anchors.MoveToOuterDisposeScope();
            }
        }

        /// <summary>
        /// Duplicate and distribute anchors to different positions give border of positions and stride between positions.
        /// </summary>
        /// <param name="shape">Border to distribute anchors.</param>
        /// <param name="stride">Stride between adjacent anchors.</param>
        /// <param name="anchors">Anchors to distribute.</param>
        /// <returns>The shifted anchors.</returns>
        private static Tensor Shift(Tensor shape, int stride, Tensor anchors)
        {
            using (var anchorsScope = torch.NewDisposeScope())
            {
                Tensor shiftX = (torch.arange(start: 0, stop: (int)shape[1]) + 0.5) * stride;
                Tensor shiftY = (torch.arange(start: 0, stop: (int)shape[0]) + 0.5) * stride;

                var shiftXExpand = torch.repeat_interleave(shiftX.reshape(new long[] { shiftX.shape[0], 1 }), shiftY.shape[0], dim: 1);
                shiftXExpand = shiftXExpand.transpose(0, 1).reshape(-1);
                var shiftYExpand = torch.repeat_interleave(shiftY, shiftX.shape[0]);

                List<Tensor> tensors = new List<Tensor> { shiftXExpand, shiftYExpand, shiftXExpand, shiftYExpand };
                var shifts = torch.vstack(tensors).transpose(0, 1);

                var a = anchors.shape[0];
                var k = shifts.shape[0];
                var allAnchors = anchors.reshape(new long[] { 1, a, 4 }) + shifts.reshape(new long[] { 1, k, 4 }).transpose(0, 1);
                allAnchors = allAnchors.reshape(new long[] { k * a, 4 });

                return allAnchors.MoveToOuterDisposeScope();
            }
        }
    }
}
