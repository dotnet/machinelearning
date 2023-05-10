// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.TorchSharp;
using Microsoft.ML.TorchSharp.AutoFormerV2;

namespace Microsoft.ML.AutoML.CodeGen
{
    internal partial class ObjectDetectionMulti
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, ObjectDetectionOption param)
        {
            var option = new ObjectDetectionTrainer.Options
            {
                LabelColumnName = param.LabelColumnName,
                PredictedLabelColumnName = param.PredictedLabelColumnName,
                BoundingBoxColumnName = param.BoundingBoxColumnName,
                ImageColumnName = param.ImageColumnName,
                ScoreColumnName = param.ScoreColumnName,
                MaxEpoch = param.MaxEpoch,
                InitLearningRate = param.InitLearningRate,
                WeightDecay = param.WeightDecay,
                PredictedBoundingBoxColumnName = param.PredictedBoundingBoxColumnName,
                ScoreThreshold = param.ScoreThreshold,
                IOUThreshold = param.IOUThreshold,
            };

            return context.MulticlassClassification.Trainers.ObjectDetection(option);
        }
    }
}
