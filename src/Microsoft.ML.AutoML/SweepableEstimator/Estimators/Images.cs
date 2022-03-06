// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.AutoML.CodeGen
{
    internal partial class LoadImages
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, LoadImageOption param)
        {
            return context.Transforms.LoadImages(param.OutputColumnName, param.ImageFolder, param.InputColumnName);
        }
    }

    internal partial class LoadRawImageBytes
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, LoadImageOption param)
        {
            return context.Transforms.LoadRawImageBytes(param.OutputColumnName, param.ImageFolder, param.InputColumnName);
        }
    }

    internal partial class ResizeImages
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, ResizeImageOption param)
        {
            return context.Transforms.ResizeImages(param.OutputColumnName, param.ImageWidth, param.ImageHeight, param.InputColumnName, param.Resizing, param.CropAnchor);
        }
    }

    internal partial class ExtractPixels
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, ExtractPixelsOption param)
        {
            return context.Transforms.ExtractPixels(param.OutputColumnName, param.InputColumnName, param.ColorsToExtract, param.OrderOfExtraction, outputAsFloatArray: param.OutputAsFloatArray);
        }
    }

    internal partial class ImageClassificationMulti
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, ImageClassificationOption param)
        {

            return context.MulticlassClassification.Trainers.ImageClassification(param.LabelColumnName, param.FeatureColumnName, param.ScoreColumnName);
        }
    }
}
