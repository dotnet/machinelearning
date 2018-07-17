// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.ImageAnalytics.EntryPoints;

[assembly: LoadableClass(typeof(void), typeof(ImageAnalytics), null, typeof(SignatureEntryPointModule), "ImageAnalytics")]
namespace Microsoft.ML.Runtime.ImageAnalytics.EntryPoints
{
    public static class ImageAnalytics
    {
        [TlcModule.EntryPoint(Name = "Transforms.ImageLoader", Desc = ImageLoaderTransform.Summary,
            UserName = ImageLoaderTransform.UserName, ShortName = ImageLoaderTransform.LoaderSignature)]
        public static CommonOutputs.TransformOutput ImageLoader(IHostEnvironment env, ImageLoaderTransform.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "ImageLoaderTransform", input);
            var xf = new ImageLoaderTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.ImageResizer", Desc = ImageResizerTransform.Summary,
            UserName = ImageResizerTransform.UserName, ShortName = ImageResizerTransform.LoaderSignature)]
        public static CommonOutputs.TransformOutput ImageResizer(IHostEnvironment env, ImageResizerTransform.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "ImageResizerTransform", input);
            var xf = new ImageResizerTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.ImagePixelExtractor", Desc = ImagePixelExtractorTransform.Summary,
            UserName = ImagePixelExtractorTransform.UserName, ShortName = ImagePixelExtractorTransform.LoaderSignature)]
        public static CommonOutputs.TransformOutput ImagePixelExtractor(IHostEnvironment env, ImagePixelExtractorTransform.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "ImagePixelExtractorTransform", input);
            var xf = new ImagePixelExtractorTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.ImageGreyscale", Desc = ImageGreyscaleTransform.Summary,
            UserName = ImageGreyscaleTransform.UserName, ShortName = ImageGreyscaleTransform.LoaderSignature)]
        public static CommonOutputs.TransformOutput ImageGreyscale(IHostEnvironment env, ImageGreyscaleTransform.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "ImageGreyscaleTransform", input);
            var xf = new ImageGreyscaleTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, xf, input.Data),
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
                Model = new TransformModel(h, xf, input.Data),
                OutputData = xf
            };
        }
    }
}
