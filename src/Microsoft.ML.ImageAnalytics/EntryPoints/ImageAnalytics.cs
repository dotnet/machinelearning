// Licensed to the .NET Foundation under one or more agreements.
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
        [TlcModule.EntryPoint(Name = "Transforms.ImageLoader", Desc = ImageLoaderTransformer.Summary,
            UserName = ImageLoaderTransformer.UserName, ShortName = ImageLoaderTransformer.LoaderSignature)]
        public static CommonOutputs.TransformOutput ImageLoader(IHostEnvironment env, ImageLoaderTransformer.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "ImageLoaderTransform", input);
            var xf = ImageLoaderTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.ImageResizer", Desc = ImageResizerTransformer.Summary,
            UserName = ImageResizerTransformer.UserName, ShortName = ImageResizerTransformer.LoaderSignature)]
        public static CommonOutputs.TransformOutput ImageResizer(IHostEnvironment env, ImageResizerTransformer.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "ImageResizerTransform", input);
            var xf = ImageResizerTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.ImagePixelExtractor", Desc = ImagePixelExtractorTransformer.Summary,
            UserName = ImagePixelExtractorTransformer.UserName, ShortName = ImagePixelExtractorTransformer.LoaderSignature)]
        public static CommonOutputs.TransformOutput ImagePixelExtractor(IHostEnvironment env, ImagePixelExtractorTransformer.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "ImagePixelExtractorTransform", input);
            var xf = ImagePixelExtractorTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.ImageGrayscale", Desc = ImageGrayscaleTransformer.Summary,
            UserName = ImageGrayscaleTransformer.UserName, ShortName = ImageGrayscaleTransformer.LoaderSignature)]
        public static CommonOutputs.TransformOutput ImageGrayscale(IHostEnvironment env, ImageGrayscaleTransformer.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "ImageGrayscaleTransform", input);
            var xf = ImageGrayscaleTransformer.Create(h, input, input.Data);
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
