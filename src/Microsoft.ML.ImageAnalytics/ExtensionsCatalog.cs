// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.ImageAnalytics;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    public static class ImageEstimatorsCatalog
    {
        /// <summary>
        /// Converts the images to grayscale.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">The name of the columns containing the image paths(first item of the tuple), and the name of the resulting output column (second item of the tuple).</param>
        public static ImageGrayscalingEstimator Grayscale(this TransformsCatalog catalog, params (string input, string output)[] columns)
            => new ImageGrayscalingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Loads the images from a given folder.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="imageFolder">The images folder.</param>
        /// <param name="columns">The name of the columns containing the image paths(first item of the tuple), and the name of the resulting output column (second item of the tuple).</param>
        public static ImageLoadingEstimator LoadImages(this TransformsCatalog catalog, string imageFolder, params (string input, string output)[] columns)
           => new ImageLoadingEstimator(CatalogUtils.GetEnvironment(catalog), imageFolder, columns);

        /// <summary>
        /// Loads the images from a given folder.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="inputColumn">The name of the input column.</param>
        /// <param name="outputColumn">The name of the output column generated from the estimator.</param>
        /// <param name="colors">The color schema as defined in <see cref="ImagePixelExtractorTransform.ColorBits"/>.</param>
        /// <param name="interleave"></param>
        public static ImagePixelExtractingEstimator ExtractPixels(this TransformsCatalog catalog, string inputColumn, string outputColumn,
        ImagePixelExtractorTransform.ColorBits colors = ImagePixelExtractorTransform.ColorBits.Rgb, bool interleave = false)
            => new ImagePixelExtractingEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, colors, interleave);

        /// <summary>
        /// Loads the images from a given folder.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">The name of the columns containing the image paths, and per-column configurations.</param>
        public static ImagePixelExtractingEstimator ExtractPixels(this TransformsCatalog catalog, params ImagePixelExtractorTransform.ColumnInfo[] columns)
            => new ImagePixelExtractingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        ///  Resizes an image.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="inputColumn">Name of the input column.</param>
        /// <param name="outputColumn">Name of the resulting output column.</param>
        /// <param name="imageWidth">The image width.</param>
        /// <param name="imageHeight">The image height.</param>
        /// <param name="resizing">The type of image resizing as specified in <see cref="ImageResizerTransform.ResizingKind"/>.</param>
        /// <param name="cropAnchor">Where to place the anchor, to start cropping. Options defined in <see cref="ImageResizerTransform.Anchor"/></param>
        /// <returns></returns>
        public static ImageResizingEstimator Resize(this TransformsCatalog catalog, string inputColumn, string outputColumn,
        int imageWidth, int imageHeight, ImageResizerTransform.ResizingKind resizing = ImageResizerTransform.ResizingKind.IsoCrop, ImageResizerTransform.Anchor cropAnchor = ImageResizerTransform.Anchor.Center)
        => new ImageResizingEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, imageWidth, imageHeight, resizing, cropAnchor);

        /// <summary>
        /// Resizes an image.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">The names of the columns to transform.</param>
        /// <returns></returns>
        public static ImageResizingEstimator Resize(this TransformsCatalog catalog, params ImageResizerTransform.ColumnInfo[] columns)
            => new ImageResizingEstimator(CatalogUtils.GetEnvironment(catalog), columns);
    }
}
