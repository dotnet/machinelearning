// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.AutoML;

namespace Microsoft.ML.CodeGenerator.CSharp
{
    internal interface ITransformGenerator
    {
        string GenerateTransformer();

        string[] GenerateUsings();
    }

    // For orphan Transformer ( transformer that doesn't exist in AutoML )
    internal enum SpecialTransformer
    {
        ApplyOnnxModel = 0,
        ResizeImage = 1,
        ExtractPixel = 2,
        ObjectDetectionResizeImage = 3,
    }

    internal static class TransformGeneratorFactory
    {
        internal static ITransformGenerator GetInstance(PipelineNode node)
        {
            ITransformGenerator result = null;
            if (Enum.TryParse(node.Name, out EstimatorName trainer))
            {
                switch (trainer)
                {
                    case EstimatorName.Normalizing:
                        result = new Normalizer(node);
                        break;
                    case EstimatorName.OneHotEncoding:
                        result = new OneHotEncoding(node);
                        break;
                    case EstimatorName.ColumnConcatenating:
                        result = new ColumnConcat(node);
                        break;
                    case EstimatorName.ColumnCopying:
                        result = new ColumnCopying(node);
                        break;
                    case EstimatorName.KeyToValueMapping:
                        result = new KeyToValueMapping(node);
                        break;
                    case EstimatorName.Hashing:
                        result = new Hashing(node);
                        break;
                    case EstimatorName.MissingValueIndicating:
                        result = new MissingValueIndicator(node);
                        break;
                    case EstimatorName.MissingValueReplacing:
                        result = new MissingValueReplacer(node);
                        break;
                    case EstimatorName.OneHotHashEncoding:
                        result = new OneHotHashEncoding(node);
                        break;
                    case EstimatorName.TextFeaturizing:
                        result = new TextFeaturizing(node);
                        break;
                    case EstimatorName.TypeConverting:
                        result = new TypeConverting(node);
                        break;
                    case EstimatorName.ValueToKeyMapping:
                        result = new ValueToKeyMapping(node);
                        break;
                    case EstimatorName.RawByteImageLoading:
                        result = new ImageLoadingRawBytes(node);
                        break;
                    case EstimatorName.ImageLoading:
                        result = new ImageLoading(node);
                        break;
                    default:
                        // see if node is one of those Transformer
                        return null;
                }
            }

            // For  the AzureAttach
            if (Enum.TryParse(node.Name, out SpecialTransformer transformer))
            {
                switch (transformer)
                {
                    case SpecialTransformer.ExtractPixel:
                        result = new PixelExtract(node);
                        break;
                    case SpecialTransformer.ResizeImage:
                        result = new ImageResizing(node);
                        break;
                    case SpecialTransformer.ApplyOnnxModel:
                        result = new ApplyOnnxModel(node);
                        break;
                    case SpecialTransformer.ObjectDetectionResizeImage:
                        result = new ObjectDetectionImageResizing(node);
                        break;
                    default:
                        return null;
                }
            }
            return result;
        }
    }
}
