// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using static TorchSharp.torch.utils.data;
using static TorchSharp.torch;
using TorchSharp;
using Microsoft.ML.Data;

namespace Microsoft.ML.TorchSharp.Utils
{
    internal class ImageUtils
    {

        public static void Postprocess(Tensor imgBatch, Tensor classification, Tensor regression, Tensor anchors, out UInt32[] predictedLabels, out Single[] score, out Single[] boxes, double scoreThreshold = 0.05, double overlapThreshold = 0.5)
        {
            predictedLabels = null;
            score = null;
            boxes = null;

            if (imgBatch is null)
            {
                throw new ArgumentNullException(nameof(imgBatch));
            }

            if (classification is null)
            {
                throw new ArgumentNullException(nameof(classification));
            }

            if (regression is null)
            {
                throw new ArgumentNullException(nameof(regression));
            }

            if (anchors is null)
            {
                throw new ArgumentNullException(nameof(anchors));
            }

            using (var postprocessScope = torch.NewDisposeScope())
            {
                var transformedAnchors1 = TransformBbox(anchors, regression);
                var transformedAnchors = ClipBoxes(transformedAnchors1, imgBatch);

                var finalResult = new[] { new List<Tensor>(), new List<Tensor>(), new List<Tensor>() };

                for (int i = 0; i < classification.shape[2]; ++i)
                {
                    var scores1 = torch.squeeze(classification[RangeUtil.ToTensorIndex(..), RangeUtil.ToTensorIndex(..), i], null);
                    var scoresOverThresh = scores1 > 0.05;
                    if (scoresOverThresh.sum().ToSingle() == 0)
                    {
                        // no boxes to NMS, just continue
                        continue;
                    }

                    var scores = scores1[scoresOverThresh];
                    var anchorBoxes1 = torch.squeeze(transformedAnchors, null);
                    var anchorBoxes = anchorBoxes1[scoresOverThresh];
                    var anchorsNmsIdx = Nms(anchorBoxes, scores, overlapThreshold);
                    var finalAnchorBoxesIndexesValue = torch.ones(anchorsNmsIdx.shape[0], dtype: ScalarType.Int64, device: imgBatch.device).multiply(i);

                    finalResult[0].Add(scores[anchorsNmsIdx]);
                    finalResult[1].Add(finalAnchorBoxesIndexesValue);
                    finalResult[2].Add(anchorBoxes[anchorsNmsIdx]);
                }

                int boxIndex = 0;

                if (finalResult[0].Count > 0)
                {
                    var finalScores = torch.cat(finalResult[0], dim: 0);
                    var finalAnchorBoxesIndexes = torch.cat(finalResult[1], dim: 0);
                    var finalAnchorBoxesCoordinates = torch.cat(finalResult[2], dim: 0);

                    var idxs = (finalScores >= scoreThreshold).nonzero();
                    predictedLabels = new uint[idxs.shape[0]];
                    score = new float[idxs.shape[0]];
                    boxes = new float[idxs.shape[0] * 4];
                    for (int i = 0; i < idxs.shape[0]; ++i)
                    {
                        var id = idxs[i, 0];
                        var bbox = finalAnchorBoxesCoordinates[id];
                        var index = finalAnchorBoxesIndexes[id].ToInt64();
                        predictedLabels[i] = (uint)index + 1;
                        score[i] = finalScores[id].ToSingle();
                        boxes[boxIndex++] = bbox[0].ToSingle();
                        boxes[boxIndex++] = bbox[1].ToSingle();
                        boxes[boxIndex++] = bbox[2].ToSingle();
                        boxes[boxIndex++] = bbox[3].ToSingle();
                    }
                }
                else
                {
                    predictedLabels = new uint[0];
                    score = new float[0];
                    boxes = new float[0];
                }
            }
        }

        private static Tensor Nms(Tensor boxes, Tensor scores, double iouThreshold = 0.5)
        {
            using (var nmsScope = torch.NewDisposeScope())
            {
                // boxes: Tensor [N,4]，scores: Tensor [N,]
                var x1 = boxes[RangeUtil.ToTensorIndex(..), 0];
                var y1 = boxes[RangeUtil.ToTensorIndex(..), 1];
                var x2 = boxes[RangeUtil.ToTensorIndex(..), 2];
                var y2 = boxes[RangeUtil.ToTensorIndex(..), 3];
                var areas = (x2 - x1) * (y2 - y1); // [N,]

                var (_, _order) = scores.sort(0, descending: true);

                var keep = new List<long>();
                var order = _order[RangeUtil.ToTensorIndex(..)];
                while (order.numel() > 0)
                {
                    long i;
                    if (order.numel() == 1)
                    {
                        i = order.cpu().item<long>();
                        keep.Add(i);
                        break;
                    }
                    else
                    {
                        i = order[0].cpu().item<long>();
                        keep.Add(i);
                    }

                    var xx1 = x1[order[RangeUtil.ToTensorIndex(1..)]].clamp(min: x1[i]); // [N - 1,]
                    var yy1 = y1[order[RangeUtil.ToTensorIndex(1..)]].clamp(min: y1[i]);
                    var xx2 = x2[order[RangeUtil.ToTensorIndex(1..)]].clamp(max: x2[i]);
                    var yy2 = y2[order[RangeUtil.ToTensorIndex(1..)]].clamp(max: y2[i]);
                    var inter = (xx2 - xx1).clamp(min: 0) * (yy2 - yy1).clamp(min: 0); // [N - 1,]

                    var iou = inter / (areas[i] + areas[order[RangeUtil.ToTensorIndex(1..)]] - inter); // [N-1, ]
                    var idx = (iou <= iouThreshold).nonzero().squeeze(); // idx: [N - 1,] and order:[N,]
                    if (idx.numel() == 0)
                    {
                        break;
                    }

                    order = order[idx + 1];
                }

                var ids = torch.from_array(keep.ToArray()).to_type(ScalarType.Int64).to(device: boxes.device);
                return ids.MoveToOuterDisposeScope();
            }
        }

        /// <summary>
        /// Transform bounding boxes with delta.
        /// </summary>
        /// <param name="boxes">The bounding boxes to be transformed.</param>
        /// <param name="deltas">The deltas in transform.</param>
        /// <returns>The transformed boundbing boxes.</returns>
        private static Tensor TransformBbox(Tensor boxes, Tensor deltas)
        {
            using (var transformBboxScope = torch.NewDisposeScope())
            {
                var mean = torch.from_array(new double[] { 0, 0, 0, 0 }).to_type(ScalarType.Float32).to(boxes.device);
                var std = torch.from_array(new double[] { 0.1, 0.1, 0.2, 0.2 }).to_type(ScalarType.Float32).to(boxes.device);

                var widths = boxes[RangeUtil.ToTensorIndex(..), RangeUtil.ToTensorIndex(..), 2] - boxes[RangeUtil.ToTensorIndex(..), RangeUtil.ToTensorIndex(..), 0];
                var heights = boxes[RangeUtil.ToTensorIndex(..), RangeUtil.ToTensorIndex(..), 3] - boxes[RangeUtil.ToTensorIndex(..), RangeUtil.ToTensorIndex(..), 1];
                var ctrX = boxes[RangeUtil.ToTensorIndex(..), RangeUtil.ToTensorIndex(..), 0] + (0.5 * widths);
                var ctrY = boxes[RangeUtil.ToTensorIndex(..), RangeUtil.ToTensorIndex(..), 1] + (0.5 * heights);

                var dx = (deltas[RangeUtil.ToTensorIndex(..), RangeUtil.ToTensorIndex(..), 0] * std[0]) + mean[0];
                var dy = (deltas[RangeUtil.ToTensorIndex(..), RangeUtil.ToTensorIndex(..), 1] * std[1]) + mean[1];
                var dw = (deltas[RangeUtil.ToTensorIndex(..), RangeUtil.ToTensorIndex(..), 2] * std[2]) + mean[2];
                var dh = (deltas[RangeUtil.ToTensorIndex(..), RangeUtil.ToTensorIndex(..), 3] * std[3]) + mean[3];

                var predCtrX = ctrX + (dx * widths);
                var predCtrY = ctrY + (dy * heights);
                var predW = torch.exp(dw) * widths;
                var predH = torch.exp(dh) * heights;

                var predBoxesX1 = predCtrX - (0.5 * predW);
                var predBoxesY1 = predCtrY - (0.5 * predH);
                var predBoxesX2 = predCtrX + (0.5 * predW);
                var predBoxesY2 = predCtrY + (0.5 * predH);

                var predBoxes = torch.stack(
                        new List<Tensor> { predBoxesX1, predBoxesY1, predBoxesX2, predBoxesY2 },
                        dim: 2);

                return predBoxes.MoveToOuterDisposeScope();
            }
        }

        /// <summary>
        /// Clip bounding boxes inside the size of image.
        /// </summary>
        /// <param name="boxes">The bounding boxes to be clipped.</param>
        /// <param name="img">The image to specify the bound of bounding box.</param>
        /// <returns>The clipped bounding boxes.</returns>
        private static Tensor ClipBoxes(Tensor boxes, Tensor img)
        {
            using (var clipBoxesScope = torch.NewDisposeScope())
            {
                var batchSize = img.shape[0];
                var numChannels = img.shape[1];
                var height = img.shape[2];
                var width = img.shape[3];

                var clippedBoxesX0 = torch.clamp(boxes[RangeUtil.ToTensorIndex(..), RangeUtil.ToTensorIndex(..), 0], min: 0);
                var clippedBoxesY0 = torch.clamp(boxes[RangeUtil.ToTensorIndex(..), RangeUtil.ToTensorIndex(..), 1], min: 0);

                var clippedBoxesX1 = torch.clamp(boxes[RangeUtil.ToTensorIndex(..), RangeUtil.ToTensorIndex(..), 2], max: width);
                var clippedBoxesY1 = torch.clamp(boxes[RangeUtil.ToTensorIndex(..), RangeUtil.ToTensorIndex(..), 3], max: height);

                var clippedBoxes = torch.stack(
                    new List<Tensor> { clippedBoxesX0, clippedBoxesY0, clippedBoxesX1, clippedBoxesY1 },
                    dim: 2);

                return clippedBoxes.MoveToOuterDisposeScope();
            }
        }
    }
}
