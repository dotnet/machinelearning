// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
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
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_PrivateFieldName:private field names not in _camelCase format", Justification = "Need to match TorchSharp.")]
        private readonly double alpha;
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_PrivateFieldName:private field names not in _camelCase format", Justification = "Need to match TorchSharp.")]
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
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public override Tensor forward(Tensor classifications, Tensor regressions, Tensor anchors, Tensor annotations)
        {
            var batchSize = classifications.shape[0];
            var classificationLosses = new List<Tensor>();
            var regressionLosses = new List<Tensor>();

            var anchor = anchors[0, RangeUtil.ToTensorIndex(..), RangeUtil.ToTensorIndex(..)];

            var anchorWidths = anchor[RangeUtil.ToTensorIndex(..), 2] - anchor[RangeUtil.ToTensorIndex(..), 0];
            var anchorHeights = anchor[RangeUtil.ToTensorIndex(..), 3] - anchor[RangeUtil.ToTensorIndex(..), 1];
            var anchorCtrX = anchor[RangeUtil.ToTensorIndex(..), 0] + (0.5 * anchorWidths);
            var anchorCtrY = anchor[RangeUtil.ToTensorIndex(..), 1] + (0.5 * anchorHeights);

            for (int j = 0; j < batchSize; ++j)
            {
                var classification = classifications[j, RangeUtil.ToTensorIndex(..), RangeUtil.ToTensorIndex(..)];
                var regression = regressions[j, RangeUtil.ToTensorIndex(..), RangeUtil.ToTensorIndex(..)];

                var bboxAnnotation = annotations[j, RangeUtil.ToTensorIndex(..), RangeUtil.ToTensorIndex(..)];
                bboxAnnotation = bboxAnnotation[bboxAnnotation[RangeUtil.ToTensorIndex(..), 4] != -1];

                classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4);

                if (bboxAnnotation.shape[0] == 0)
                {
                    var alphaFactor = this.alpha * torch.ones(classification.shape, dtype: ScalarType.Float32, device: classifications.device);
                    alphaFactor = 1.0f - alphaFactor;

                    var focalWeight = classification;
                    focalWeight = alphaFactor * torch.pow(focalWeight, this.gamma);

                    var bce = -torch.log(1.0f - classification);

                    var clsLoss = focalWeight * bce;
                    classificationLosses.Add(clsLoss.sum());
                    regressionLosses.Add(torch.tensor(0, dtype: ScalarType.Float32, device: classifications.device));
                }
                else
                {
                    var iou = CalcIou(anchors[0, RangeUtil.ToTensorIndex(..), RangeUtil.ToTensorIndex(..)], bboxAnnotation[RangeUtil.ToTensorIndex(..), RangeUtil.ToTensorIndex(..4)]); // num_anchors x num_annotations

                    var (iou_max, iou_argmax) = torch.max(iou, dim: 1); // num_anchors x 1

                    // compute the loss for classification
                    var targets = (-1) * torch.ones(classification.shape, dtype: ScalarType.Float32, device: classifications.device);
                    targets[torch.lt(iou_max, 0.4)] = 0;

                    Tensor positiveIndices = torch.ge(iou_max, 0.5);

                    var numPositiveAnchors = positiveIndices.sum();

                    var assignedAnnotations = bboxAnnotation[iou_argmax];

                    targets[positiveIndices] = 0;

                    var assignedPositiveIndeces = positiveIndices.nonzero().squeeze(-1);
                    for (int i = 0; i < assignedPositiveIndeces.shape[0]; i++)
                    {
                        var t = assignedPositiveIndeces[i];
                        targets[t, assignedAnnotations[t, 4]] = 1;
                    }

                    var alphaFactor = torch.ones(targets.shape, dtype: ScalarType.Float32, device: classifications.device) * alpha;
                    alphaFactor = torch.where(targets.eq(1.0), alphaFactor, 1.0 - alphaFactor);

                    var focalWeight = torch.where(targets.eq(1.0), 1.0 - classification, classification);
                    focalWeight = alphaFactor * torch.pow(focalWeight, this.gamma);

                    var bce = -((targets * torch.log(classification)) +
                               ((1.0 - targets) * torch.log(1.0 - classification)));

                    var clsLoss = focalWeight * bce;
                    clsLoss = torch.where(targets.ne(-1.0), clsLoss,
                        torch.zeros(
                            clsLoss.shape,
                            dtype: ScalarType.Float32,
                            device: classifications.device));

                    var classificationLoss = clsLoss.sum() / torch.clamp(numPositiveAnchors.to_type(ScalarType.Float32), min: 1.0);
                    classificationLosses.Add(classificationLoss);

                    // compute the loss for regression
                    if (positiveIndices.sum().ToSingle() > 0)
                    {
                        assignedAnnotations = assignedAnnotations[positiveIndices];

                        var anchorWidthsPi = anchorWidths[positiveIndices];
                        var anchorHeightsPi = anchorHeights[positiveIndices];
                        var anchorCtrXPi = anchorCtrX[positiveIndices];
                        var anchorCtrYPi = anchorCtrY[positiveIndices];

                        var gtWidths = assignedAnnotations[RangeUtil.ToTensorIndex(..), 2] - assignedAnnotations[RangeUtil.ToTensorIndex(..), 0];
                        var gtHeights = assignedAnnotations[RangeUtil.ToTensorIndex(..), 3] - assignedAnnotations[RangeUtil.ToTensorIndex(..), 1];
                        var gtCtrX = assignedAnnotations[RangeUtil.ToTensorIndex(..), 0] + (0.5 * gtWidths);
                        var gtCtrY = assignedAnnotations[RangeUtil.ToTensorIndex(..), 1] + (0.5 * gtHeights);

                        // clip widths to 1
                        gtWidths = torch.clamp(gtWidths, min: 1);
                        gtHeights = torch.clamp(gtHeights, min: 1);

                        var targetsDx = (gtCtrX - anchorCtrXPi) / anchorWidthsPi;
                        var targetsDy = (gtCtrY - anchorCtrYPi) / anchorHeightsPi;

                        var targetsDw = torch.log(gtWidths / anchorWidthsPi);
                        var targetsDh = torch.log(gtHeights / anchorHeightsPi);

                        targets = torch.stack(new List<Tensor> { targetsDx, targetsDy, targetsDw, targetsDh });
                        targets = targets.t();
                        var factor = torch.from_array(new double[]
                        {
                            0.1, 0.1, 0.2, 0.2
                        }).unsqueeze(0).to(classifications.device);
                        targets = targets / factor;

                        var negativeIndices = 1 + (~positiveIndices);

                        var regressionDiff = torch.abs(targets - regression[positiveIndices]);

                        var regressionLoss = torch.where(
                            regressionDiff.le(1.0 / 9.0),
                            0.5 * 9.0 * torch.pow(regressionDiff, 2),
                            regressionDiff - (0.5 / 9.0));
                        regressionLosses.Add(regressionLoss.mean());
                    }
                    else
                    {
                        regressionLosses.Add(torch.tensor(0, dtype: ScalarType.Float32, device: classifications.device));
                    }
                }
            }

            var finalClassificationLoss = torch.stack(classificationLosses).mean(dimensions: new long[] { 0 }, keepdim: true);
            var finalRegressionLoss = torch.stack(regressionLosses).mean(dimensions: new long[] { 0 }, keepdim: true);
            var loss = finalClassificationLoss.mean() + finalRegressionLoss.mean();
            return loss;
        }

        private object ToTensorIndex()
        {
            throw new NotImplementedException();
        }

        private static Tensor CalcIou(Tensor a, Tensor b)
        {
            var area = (b[RangeUtil.ToTensorIndex(..), 2] - b[RangeUtil.ToTensorIndex(..), 0]) * (b[RangeUtil.ToTensorIndex(..), 3] - b[RangeUtil.ToTensorIndex(..), 1]);

            var iw = torch.minimum(input: torch.unsqueeze(a[RangeUtil.ToTensorIndex(..), 2], dim: 1), b[RangeUtil.ToTensorIndex(..), 2]) -
                     torch.maximum(input: torch.unsqueeze(a[RangeUtil.ToTensorIndex(..), 0], 1), b[RangeUtil.ToTensorIndex(..), 0]);
            var ih = torch.minimum(input: torch.unsqueeze(a[RangeUtil.ToTensorIndex(..), 3], dim: 1), b[RangeUtil.ToTensorIndex(..), 3]) -
                     torch.maximum(input: torch.unsqueeze(a[RangeUtil.ToTensorIndex(..), 1], 1), b[RangeUtil.ToTensorIndex(..), 1]);

            iw = torch.clamp(iw, min: 0);
            ih = torch.clamp(ih, min: 0);

            var ua = torch.unsqueeze((a[RangeUtil.ToTensorIndex(..), 2] - a[RangeUtil.ToTensorIndex(..), 0]) * (a[RangeUtil.ToTensorIndex(..), 3] - a[RangeUtil.ToTensorIndex(..), 1]), dim: 1) + area - (iw * ih);
            ua = torch.clamp(ua, min: 1e-8);

            var intersection = iw * ih;
            var iou = intersection / ua;

            return iou;
        }
    }
}
