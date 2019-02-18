// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.ImageAnalytics;

namespace Microsoft.ML
{
    public static class ImageEstimatorsCatalog
    {
        /// <include file='doc.xml' path='doc/members/member[@name="ImageGrayscalingEstimator"]/*' />
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columnPairs">The name of the columns containing the name of the resulting output column (first item of the tuple), and the paths of the images to work on (second item of the tuple).</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[ConvertToGrayscale](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/ImageAnalytics/ConvertToGrayscale.cs)]
        /// ]]></format>
        /// </example>
        public static ImageGrayscalingEstimator ConvertToGrayscale(this TransformsCatalog catalog, params (string outputColumnName, string inputColumnName)[] columnPairs)
            => new ImageGrayscalingEstimator(CatalogUtils.GetEnvironment(catalog), columnPairs);

        /// <summary>
        /// Loads the images from the <see cref="ImageLoadingTransformer.ImageFolder" /> into memory.
        /// </summary>
        /// <remarks>
        /// The image get loaded in memory as a <see cref="System.Drawing.Bitmap" /> type.
        /// Loading is the first step of almost every pipeline that does image processing, and further analysis on images.
        /// The images to load need to be in the formats supported by <see cref = "System.Drawing.Bitmap" />.
        /// For end-to-end image processing pipelines, and scenarios in your applications, see the
        /// <a href="https://github.com/dotnet/machinelearning-samples/tree/master/samples/csharp/getting-started"> examples in the machinelearning-samples github repository.</a>
        /// <seealso cref = "ImageEstimatorsCatalog" />
        /// </remarks>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="imageFolder">The images folder.</param>
        /// <param name="columnPairs">The name of the columns containing the image paths(first item of the tuple), and the name of the resulting output column (second item of the tuple).</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[LoadImages](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/ImageAnalytics/LoadImages.cs)]
        /// ]]></format>
        /// </example>
        public static ImageLoadingEstimator LoadImages(this TransformsCatalog catalog, string imageFolder, params (string outputColumnName, string inputColumnName)[] columnPairs)
           => new ImageLoadingEstimator(CatalogUtils.GetEnvironment(catalog), imageFolder, columnPairs);

        /// <include file='doc.xml' path='doc/members/member[@name="ImagePixelExtractingEstimator"]/*' />
        /// <param name="catalog"> The transform's catalog.</param>
        /// <param name="outputColumnName"> Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName"> Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="colors">What colors to extract.</param>
        /// <param name="order">In which order extract colors from pixel.</param>
        /// <param name="interleave">Whether to interleave the pixels, meaning keep them in the <paramref name="order"/> order, or leave them in the plannar form:
        /// first output one color values for all pixels, then another color and so on.</param>
        /// <param name="scale">Scale color pixel value by this amount.</param>
        /// <param name="offset">Offset color pixel value by this amount.</param>
        /// <param name="asFloat">Output the array as float array. If false, output as byte array.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[ExtractPixels](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/ImageAnalytics/ExtractPixels.cs)]
        /// ]]></format>
        /// </example>
        public static ImagePixelExtractingEstimator ExtractPixels(this TransformsCatalog catalog,
            string outputColumnName,
            string inputColumnName = null,
            ImagePixelExtractingEstimator.ColorBits colors = ImagePixelExtractingEstimator.Defaults.Colors,
            ImagePixelExtractingEstimator.ColorsOrder order = ImagePixelExtractingEstimator.Defaults.Order,
            bool interleave = false,
            float scale = ImagePixelExtractingEstimator.Defaults.Scale,
            float offset = ImagePixelExtractingEstimator.Defaults.Offset,
            bool asFloat = ImagePixelExtractingEstimator.Defaults.Convert)
            => new ImagePixelExtractingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName, colors, order, interleave, scale, offset, asFloat);

        /// <include file='doc.xml' path='doc/members/member[@name="ImagePixelExtractingEstimator"]/*' />
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">The name of the columns containing the image paths, and per-column configurations.</param>
        public static ImagePixelExtractingEstimator ExtractPixels(this TransformsCatalog catalog, params ImagePixelExtractingEstimator.ColumnInfo[] columns)
            => new ImagePixelExtractingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <summary>
        /// Resizes the images to a new width and height.
        /// </summary>
        /// <remarks>
        /// In image processing pipelines, often machine learning practitioner make use of<a href= "https://blogs.msdn.microsoft.com/mlserver/2017/04/12/image-featurization-with-a-pre-trained-deep-neural-network-model/">
        /// pre-trained DNN featurizers</a> to extract features for usage in the machine learning algorithms.
        /// Those pre-trained models have a defined width and height for their input images, so often, after getting loaded, the images will need to get resized before
        /// further processing.
        /// The new width and height can be specified in the <paramref name="imageWidth"/> and <paramref name="imageHeight"/>
        /// <seealso cref = "ImageEstimatorsCatalog" />
        /// <seealso cref= "ImageLoadingEstimator" />
        /// </remarks >
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="inputColumnName">Name of the input column.</param>
        /// <param name="outputColumnName">Name of the resulting output column.</param>
        /// <param name="imageWidth">The transformed image width.</param>
        /// <param name="imageHeight">The transformed image height.</param>
        /// <param name="resizing"> The type of image resizing as specified in <see cref="ImageResizingEstimator.ResizingKind"/>.</param>
        /// <param name="cropAnchor">Where to place the anchor, to start cropping. Options defined in <see cref="ImageResizingEstimator.Anchor"/></param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[ResizeImages](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/ImageAnalytics/ResizeImages.cs)]
        /// ]]></format>
        /// </example>
        public static ImageResizingEstimator ResizeImages(this TransformsCatalog catalog,
            string outputColumnName,
            int imageWidth,
            int imageHeight,
            string inputColumnName = null,
            ImageResizingEstimator.ResizingKind resizing = ImageResizingEstimator.ResizingKind.IsoCrop,
            ImageResizingEstimator.Anchor cropAnchor = ImageResizingEstimator.Anchor.Center)
        => new ImageResizingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, imageWidth, imageHeight, inputColumnName, resizing, cropAnchor);

        /// <summary>
        /// Resizes the images to a new width and height.
        /// </summary>
        /// <remarks>
        /// In image processing pipelines, often machine learning practitioner make use of<a href= "https://blogs.msdn.microsoft.com/mlserver/2017/04/12/image-featurization-with-a-pre-trained-deep-neural-network-model/">
        /// pre - trained DNN featurizers</a> to extract features for usage in the machine learning algorithms.
        /// Those pre-trained models have a defined width and height for their input images, so often, after getting loaded, the images will need to get resized before
        /// further processing.
        /// The new width and height, as well as other properties of resizing, like type of scaling (uniform, or non-uniform), and whether to pad the image,
        /// or just crop it can be specified separately for each column loaded, through the <see cref="ImageResizingEstimator.ColumnInfo"/>.
        /// <seealso cref = "ImageEstimatorsCatalog" />
        /// <seealso cref= "ImageLoadingEstimator" />
        /// </remarks >
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">The names of the columns to transform.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[ResizeImages](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/ImageAnalytics/ResizeImages.cs)]
        /// ]]></format>
        /// </example>
        public static ImageResizingEstimator ResizeImages(this TransformsCatalog catalog, params ImageResizingEstimator.ColumnInfo[] columns)
            => new ImageResizingEstimator(CatalogUtils.GetEnvironment(catalog), columns);
    }
}
