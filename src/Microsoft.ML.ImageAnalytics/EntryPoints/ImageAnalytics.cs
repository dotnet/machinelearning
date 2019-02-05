﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.ImageAnalytics.EntryPoints;

[assembly: LoadableClass(typeof(void), typeof(ImageAnalytics), null, typeof(SignatureEntryPointModule), "ImageAnalytics")]
namespace Microsoft.ML.ImageAnalytics.EntryPoints
{
    internal static class ImageAnalytics
    {
        [TlcModule.EntryPoint(Name = "Transforms.ImageLoader", Desc = ImageLoadingTransformer.Summary,
            UserName = ImageLoadingTransformer.UserName, ShortName = ImageLoadingTransformer.LoaderSignature)]
        public static CommonOutputs.TransformOutput ImageLoader(IHostEnvironment env, ImageLoadingTransformer.Options input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "ImageLoaderTransform", input);
            var xf = ImageLoadingTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.ImageResizer", Desc = ImageResizingTransformer.Summary,
            UserName = ImageResizingTransformer.UserName, ShortName = ImageResizingTransformer.LoaderSignature)]
        public static CommonOutputs.TransformOutput ImageResizer(IHostEnvironment env, ImageResizingTransformer.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "ImageResizerTransform", input);
            var xf = ImageResizingTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.ImagePixelExtractor", Desc = ImagePixelExtractingTransformer.Summary,
            UserName = ImagePixelExtractingTransformer.UserName, ShortName = ImagePixelExtractingTransformer.LoaderSignature)]
        public static CommonOutputs.TransformOutput ImagePixelExtractor(IHostEnvironment env, ImagePixelExtractingTransformer.Options input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "ImagePixelExtractorTransform", input);
            var xf = ImagePixelExtractingTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.ImageGrayscale", Desc = ImageGrayscalingTransformer.Summary,
            UserName = ImageGrayscalingTransformer.UserName, ShortName = ImageGrayscalingTransformer.LoaderSignature)]
        public static CommonOutputs.TransformOutput ImageGrayscale(IHostEnvironment env, ImageGrayscalingTransformer.Options input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "ImageGrayscaleTransform", input);
            var xf = ImageGrayscalingTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.VectorToImage", Desc = VectorToImageTransform.Summary,
            UserName = VectorToImageTransform.UserName, ShortName = VectorToImageTransform.LoaderSignature)]
        public static CommonOutputs.TransformOutput VectorToImage(IHostEnvironment env, VectorToImageTransform.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "VectorToImageTransform", input);
            var xf = new VectorToImageTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }
    }
}
