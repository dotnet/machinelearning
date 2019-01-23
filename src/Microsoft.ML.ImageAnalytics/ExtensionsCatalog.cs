// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.ImageAnalytics;

namespace Microsoft.ML
{
    public static class ImageEstimatorsCatalog
    {
        /// <summary>
        /// Converts the images to grayscale.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">The name of the columns containing the image paths(first item of the tuple), and the name of the resulting output column (second item of the tuple).</param>
        public static ImageGrayscalingEstimator ConvertToGrayscale(this TransformsCatalog catalog, params (string name, string source)[] columns)
            => new ImageGrayscalingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Loads the images from a given folder.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="imageFolder">The images folder.</param>
        /// <param name="columns">The name of the columns containing the image paths(first item of the tuple), and the name of the resulting output column (second item of the tuple).</param>
        public static ImageLoadingEstimator LoadImages(this TransformsCatalog catalog, string imageFolder, params (string name, string source)[] columns)
           => new ImageLoadingEstimator(CatalogUtils.GetEnvironment(catalog), imageFolder, columns);

        /// <summary>
        /// Loads the images from a given folder.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="name">Name of the column resulting from the transformation of <paramref name="source"/>.</param>
        /// <param name="source">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="name"/> will be used as source.</param>
        /// <param name="colors">The color schema as defined in <see cref="ImagePixelExtractorTransformer.ColorBits"/>.</param>
        /// <param name="interleave"></param>
        public static ImagePixelExtractingEstimator ExtractPixels(this TransformsCatalog catalog, string name, string source,
        ImagePixelExtractorTransformer.ColorBits colors = ImagePixelExtractorTransformer.ColorBits.Rgb, bool interleave = false)
            => new ImagePixelExtractingEstimator(CatalogUtils.GetEnvironment(catalog), name, source, colors, interleave);

        /// <summary>
        /// Loads the images from a given folder.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">The name of the columns containing the image paths, and per-column configurations.</param>
        public static ImagePixelExtractingEstimator ExtractPixels(this TransformsCatalog catalog, params ImagePixelExtractorTransformer.ColumnInfo[] columns)
            => new ImagePixelExtractingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        ///  Resizes an image.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="source">Name of the input column.</param>
        /// <param name="name">Name of the resulting output column.</param>
        /// <param name="imageWidth">The image width.</param>
        /// <param name="imageHeight">The image height.</param>
        /// <param name="resizing">The type of image resizing as specified in <see cref="ImageResizerTransformer.ResizingKind"/>.</param>
        /// <param name="cropAnchor">Where to place the anchor, to start cropping. Options defined in <see cref="ImageResizerTransformer.Anchor"/></param>
        /// <returns></returns>
        public static ImageResizingEstimator Resize(this TransformsCatalog catalog, string name, string source,
        int imageWidth, int imageHeight, ImageResizerTransformer.ResizingKind resizing = ImageResizerTransformer.ResizingKind.IsoCrop, ImageResizerTransformer.Anchor cropAnchor = ImageResizerTransformer.Anchor.Center)
        => new ImageResizingEstimator(CatalogUtils.GetEnvironment(catalog), name, source, imageWidth, imageHeight, resizing, cropAnchor);

        /// <summary>
        /// Resizes an image.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">The names of the columns to transform.</param>
        /// <returns></returns>
        public static ImageResizingEstimator Resize(this TransformsCatalog catalog, params ImageResizerTransformer.ColumnInfo[] columns)
            => new ImageResizingEstimator(CatalogUtils.GetEnvironment(catalog), columns);
    }
}
