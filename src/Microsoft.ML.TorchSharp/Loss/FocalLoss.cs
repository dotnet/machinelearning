// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Microsoft.ML.TorchSharp.Loss
{
    /// <summary>
    /// A kind of loss function to balance easy and hard samples.
    /// </summary>
    public class FocalLoss : Module<Tensor, Tensor, Tensor, Tensor, Tensor>
    {
        private readonly double alpha;
        private readonly double gamma;

        /// <summary>
        /// Initializes a new instance of the <see cref="FocalLoss"/> class.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="gamma">The gamma.</param>
        public FocalLoss(double alpha = 0.25, double gamma = 2.0)
            : base(nameof(FocalLoss))
        {
            this.alpha = alpha;
            this.gamma = gamma;
        }

        /// <inheritdoc/>
        public override Tensor forward(Tensor classifications, Tensor regressions, Tensor anchors, Tensor annotations)
        {
            var batch_size = classifications.shape[0];
            var classification_losses = new List<Tensor>();
            var regression_losses = new List<Tensor>();

            var anchor = anchors[0, .., ..];

            var anchor_widths = anchor[.., 2] - anchor[.., 0];
            var anchor_heights = anchor[.., 3] - anchor[.., 1];
            var anchor_ctr_x = anchor[.., 0] + (0.5 * anchor_widths);
            var anchor_ctr_y = anchor[.., 1] + (0.5 * anchor_heights);

            for (int j = 0; j < batch_size; ++j)
            {
                var classification = classifications[j, .., ..];
                var regression = regressions[j, .., ..];

                var bbox_annotation = annotations[j, .., ..];
                bbox_annotation = bbox_annotation[bbox_annotation[.., 4] != -1];

                classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4);

                if (bbox_annotation.shape[0] == 0)
                {
                    var alpha_factor = this.alpha * torch.ones(classification.shape, dtype: ScalarType.Float32, device: classifications.device);
                    alpha_factor = 1.0f - alpha_factor;

                    var focal_weight = classification;
                    focal_weight = alpha_factor * torch.pow(focal_weight, this.gamma);

                    var bce = -torch.log(1.0f - classification);

                    var cls_loss = focal_weight * bce;
                    classification_losses.Add(cls_loss.sum());
                    regression_losses.Add(torch.tensor(0, dtype: ScalarType.Float32, device: classifications.device));
                }
                else
                {
                    var iou = CalcIOU(anchors[0, .., ..], bbox_annotation[.., ..4]); // num_anchors x num_annotations

                    var (iou_max, iou_argmax) = torch.max(iou, dim: 1); // num_anchors x 1

                    // compute the loss for classification
                    var targets = (-1) * torch.ones(classification.shape, dtype: ScalarType.Float32, device: classifications.device);
                    targets[torch.lt(iou_max, 0.4)] = 0;

                    Tensor positive_indices = torch.ge(iou_max, 0.5);

                    var num_positive_anchors = positive_indices.sum();

                    var assigned_annotations = bbox_annotation[iou_argmax];

                    targets[positive_indices] = 0;

                    var assigned_positive_indeces = positive_indices.nonzero().squeeze(-1);
                    for (int i = 0; i < assigned_positive_indeces.shape[0]; i++)
                    {
                        var t = assigned_positive_indeces[i];
                        targets[t, assigned_annotations[t, 4]] = 1;
                    }

                    var alpha_factor = torch.ones(targets.shape, dtype: ScalarType.Float32, device: classifications.device) * alpha;
                    alpha_factor = torch.where(targets.eq(1.0), alpha_factor, 1.0 - alpha_factor);

                    var focal_weight = torch.where(targets.eq(1.0), 1.0 - classification, classification);
                    focal_weight = alpha_factor * torch.pow(focal_weight, this.gamma);

                    var bce = -((targets * torch.log(classification)) +
                               ((1.0 - targets) * torch.log(1.0 - classification)));

                    var cls_loss = focal_weight * bce;
                    cls_loss = torch.where(targets.ne(-1.0), cls_loss,
                        torch.zeros(
                            cls_loss.shape,
                            dtype: ScalarType.Float32,
                            device: classifications.device));

                    var classification_loss = cls_loss.sum() / torch.clamp(num_positive_anchors.to_type(ScalarType.Float32), min: 1.0);
                    classification_losses.Add(classification_loss);

                    // compute the loss for regression
                    if (positive_indices.sum().ToSingle() > 0)
                    {
                        assigned_annotations = assigned_annotations[positive_indices];

                        var anchor_widths_pi = anchor_widths[positive_indices];
                        var anchor_heights_pi = anchor_heights[positive_indices];
                        var anchor_ctr_x_pi = anchor_ctr_x[positive_indices];
                        var anchor_ctr_y_pi = anchor_ctr_y[positive_indices];

                        var gt_widths = assigned_annotations[.., 2] - assigned_annotations[.., 0];
                        var gt_heights = assigned_annotations[.., 3] - assigned_annotations[.., 1];
                        var gt_ctr_x = assigned_annotations[.., 0] + (0.5 * gt_widths);
                        var gt_ctr_y = assigned_annotations[.., 1] + (0.5 * gt_heights);

                        // clip widths to 1
                        gt_widths = torch.clamp(gt_widths, min: 1);
                        gt_heights = torch.clamp(gt_heights, min: 1);

                        var targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi;
                        var targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi;

                        var targets_dw = torch.log(gt_widths / anchor_widths_pi);
                        var targets_dh = torch.log(gt_heights / anchor_heights_pi);

                        targets = torch.stack(new List<Tensor> { targets_dx, targets_dy, targets_dw, targets_dh });
                        targets = targets.t();
                        var factor = torch.from_array(new double[]
                        {
                            0.1, 0.1, 0.2, 0.2
                        }).unsqueeze(0).to(classifications.device);
                        targets = targets / factor;

                        var negative_indices = 1 + (~positive_indices);

                        var regression_diff = torch.abs(targets - regression[positive_indices]);

                        var regression_loss = torch.where(
                            regression_diff.le(1.0 / 9.0),
                            0.5 * 9.0 * torch.pow(regression_diff, 2),
                            regression_diff - (0.5 / 9.0));
                        regression_losses.Add(regression_loss.mean());
                    }
                    else
                    {
                        regression_losses.Add(torch.tensor(0, dtype: ScalarType.Float32, device: classifications.device));
                    }
                }
            }

            var final_classification_loss = torch.stack(classification_losses).mean(dimensions: new long[] { 0 }, keepdim: true);
            var final_regression_loss = torch.stack(regression_losses).mean(dimensions: new long[] { 0 }, keepdim: true);
            var loss = final_classification_loss.mean() + final_regression_loss.mean();
            return loss;
        }

        private static Tensor CalcIOU(Tensor a, Tensor b)
        {
            var area = (b[.., 2] - b[.., 0]) * (b[.., 3] - b[.., 1]);

            var iw = torch.minimum(input: torch.unsqueeze(a[.., 2], dim: 1), b[.., 2]) -
                     torch.maximum(input: torch.unsqueeze(a[.., 0], 1), b[.., 0]);
            var ih = torch.minimum(input: torch.unsqueeze(a[.., 3], dim: 1), b[.., 3]) -
                     torch.maximum(input: torch.unsqueeze(a[.., 1], 1), b[.., 1]);

            iw = torch.clamp(iw, min: 0);
            ih = torch.clamp(ih, min: 0);

            var ua = torch.unsqueeze((a[.., 2] - a[.., 0]) * (a[.., 3] - a[.., 1]), dim: 1) + area - (iw * ih);
            ua = torch.clamp(ua, min: 1e-8);

            var intersection = iw * ih;
            var iou = intersection / ua;

            return iou;
        }
    }
}
