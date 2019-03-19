// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;

namespace Microsoft.ML
{
    public static class ImageEstimatorsCatalog
    {
        /// <include file='doc.xml' path='doc/members/member[@name="ImageGrayscalingEstimator"]/*' />
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[ConvertToGrayscale](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/ImageAnalytics/ConvertToGrayscale.cs)]
        /// ]]></format>
        /// </example>
        public static ImageGrayscalingEstimator ConvertToGrayscale(this TransformsCatalog catalog, string outputColumnName, string inputColumnName = null)
            => new ImageGrayscalingEstimator(CatalogUtils.GetEnvironment(catalog), new[] { (outputColumnName, inputColumnName ?? outputColumnName) });

        /// <include file='doc.xml' path='doc/members/member[@name="ImageGrayscalingEstimator"]/*' />
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columns">Specifies the names of the input columns for the transformation, and their respective output column names.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[ConvertToGrayscale](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/ImageAnalytics/ConvertToGrayscale.cs)]
        /// ]]></format>
        /// </example>
        [BestFriend]
        internal static ImageGrayscalingEstimator ConvertToGrayscale(this TransformsCatalog catalog, params ColumnOptions[] columns)
            => new ImageGrayscalingEstimator(CatalogUtils.GetEnvironment(catalog), ColumnOptions.ConvertToValueTuples(columns));

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
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="imageFolder">The images folder.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[LoadImages](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/ImageAnalytics/LoadImages.cs)]
        /// ]]></format>
        /// </example>
        public static ImageLoadingEstimator LoadImages(this TransformsCatalog catalog, string outputColumnName, string imageFolder, string inputColumnName = null)
           => new ImageLoadingEstimator(CatalogUtils.GetEnvironment(catalog), imageFolder, new[] { (outputColumnName, inputColumnName ?? outputColumnName) });

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
        /// <param name="columns">Specifies the names of the input columns for the transformation, and their respective output column names.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[LoadImages](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/ImageAnalytics/LoadImages.cs)]
        /// ]]></format>
        /// </example>
        [BestFriend]
        internal static ImageLoadingEstimator LoadImages(this TransformsCatalog catalog, string imageFolder, params ColumnOptions[] columns)
           => new ImageLoadingEstimator(CatalogUtils.GetEnvironment(catalog), imageFolder, ColumnOptions.ConvertToValueTuples(columns));

        /// <include file='doc.xml' path='doc/members/member[@name="ImagePixelExtractingEstimator"]/*' />
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="colorsToExtract">The colors to extract from the image.</param>
        /// <param name="orderOfExtraction">The order in which to extract colors from pixel.</param>
        /// <param name="interleavePixelColors">Whether to interleave the pixels colors, meaning keep them in the <paramref name="orderOfExtraction"/> order, or leave them in the plannar form:
        /// all the values for one color for all pixels, then all the values for another color, and so on.</param>
        /// <param name="offsetImage">Offset each pixel's color value by this amount. Applied to color value before <paramref name="scaleImage"/>.</param>
        /// <param name="scaleImage">Scale each pixel's color value by this amount. Applied to color value after <paramref name="offsetImage"/>.</param>
        /// <param name="outputAsFloatArray">Output array as float array. If false, output as byte array and ignores <paramref name="offsetImage"/> and <paramref name="scaleImage"/>.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[ExtractPixels](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/ImageAnalytics/ExtractPixels.cs)]
        /// ]]></format>
        /// </example>
        public static ImagePixelExtractingEstimator ExtractPixels(this TransformsCatalog catalog,
            string outputColumnName,
            string inputColumnName = null,
            ImagePixelExtractingEstimator.ColorBits colorsToExtract = ImagePixelExtractingEstimator.Defaults.Colors,
            ImagePixelExtractingEstimator.ColorsOrder orderOfExtraction = ImagePixelExtractingEstimator.Defaults.Order,
            bool interleavePixelColors = false,
            float offsetImage = ImagePixelExtractingEstimator.Defaults.Offset,
            float scaleImage = ImagePixelExtractingEstimator.Defaults.Scale,
            bool outputAsFloatArray = ImagePixelExtractingEstimator.Defaults.Convert)
            => new ImagePixelExtractingEstimator(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName, colorsToExtract, orderOfExtraction, interleavePixelColors, offsetImage, scaleImage, outputAsFloatArray);

        /// <include file='doc.xml' path='doc/members/member[@name="ImagePixelExtractingEstimator"]/*' />
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columnOptions">The <see cref="ImagePixelExtractingEstimator.ColumnOptions"/> describing how the transform handles each image pixel extraction output input column pair.</param>
        [BestFriend]
        internal static ImagePixelExtractingEstimator ExtractPixels(this TransformsCatalog catalog, params ImagePixelExtractingEstimator.ColumnOptions[] columnOptions)
            => new ImagePixelExtractingEstimator(CatalogUtils.GetEnvironment(catalog), columnOptions);

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
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="imageWidth">The transformed image width.</param>
        /// <param name="imageHeight">The transformed image height.</param>
        /// <param name="resizing"> The type of image resizing as specified in <see cref="ImageResizingEstimator.ResizingKind"/>.</param>
        /// <param name="cropAnchor">Where to place the anchor, to start cropping. Options defined in <see cref="ImageResizingEstimator.Anchor"/></param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[ResizeImages](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/ImageAnalytics/ResizeImages.cs)]
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
        /// or just crop it can be specified separately for each column loaded, through the <see cref="ImageResizingEstimator.ColumnOptions"/>.
        /// <seealso cref = "ImageEstimatorsCatalog" />
        /// <seealso cref= "ImageLoadingEstimator" />
        /// </remarks >
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columnOptions">The <see cref="ImageResizingEstimator.ColumnOptions"/> describing how the transform handles each image resize column.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[ResizeImages](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/ImageAnalytics/ResizeImages.cs)]
        /// ]]></format>
        /// </example>
        [BestFriend]
        internal static ImageResizingEstimator ResizeImages(this TransformsCatalog catalog, params ImageResizingEstimator.ColumnOptions[] columnOptions)
            => new ImageResizingEstimator(CatalogUtils.GetEnvironment(catalog), columnOptions);

        /// <summary>
        /// Converts vectors of pixels into <see cref="ImageType"/> representation.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="columnOptions">The <see cref="VectorToImageConvertingEstimator.ColumnOptions"/> describing how the transform handles each vector to image conversion column pair.</param>
        [BestFriend]
        internal static VectorToImageConvertingEstimator ConvertToImage(this TransformsCatalog catalog, params VectorToImageConvertingEstimator.ColumnOptions[] columnOptions)
            => new VectorToImageConvertingEstimator(CatalogUtils.GetEnvironment(catalog), columnOptions);

        /// <summary>
        /// Converts vectors of pixels into <see cref="ImageType"/> representation.
        /// </summary>
        /// <param name="catalog">The transforms' catalog.</param>
        /// <param name="imageHeight">The height of the output images.</param>
        /// <param name="imageWidth">The width of the output images.</param>
        /// <param name="outputColumnName"> Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName"> Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="colorsPresent">Specifies which <see cref="ImagePixelExtractingEstimator.ColorBits"/> are in present the input pixel vectors. The order of colors is specified in <paramref name="orderOfColors"/>.</param>
        /// <param name="orderOfColors">The order in which colors are presented in the input vector.</param>
        /// <param name="interleavedColors">Whether the pixels are interleaved, meaning whether they are in <paramref name="orderOfColors"/> order, or separated in the planar form:
        /// all the values for one color for all pixels, then all the values for another color and so on.</param>
        /// <param name="scaleImage">The values are scaled by this value before being converted to pixels. Applied to vector value before <paramref name="offsetImage"/>.</param>
        /// <param name="offsetImage">The offset is subtracted before converting the values to pixels. Applied to vector value after <paramref name="scaleImage"/>.</param>
        /// <param name="defaultAlpha">Default value for alpha color, would be overriden if <paramref name="colorsPresent"/> contains <see cref="ImagePixelExtractingEstimator.ColorBits.Alpha"/>.</param>
        /// <param name="defaultRed">Default value for red color, would be overriden if <paramref name="colorsPresent"/> contains <see cref="ImagePixelExtractingEstimator.ColorBits.Red"/>.</param>
        /// <param name="defaultGreen">Default value for grenn color, would be overriden if <paramref name="colorsPresent"/> contains <see cref="ImagePixelExtractingEstimator.ColorBits.Green"/>.</param>
        /// <param name="defaultBlue">Default value for blue color, would be overriden if <paramref name="colorsPresent"/> contains <see cref="ImagePixelExtractingEstimator.ColorBits.Blue"/>.</param>
        public static VectorToImageConvertingEstimator ConvertToImage(this TransformsCatalog catalog, int imageHeight, int imageWidth, string outputColumnName, string inputColumnName = null,
            ImagePixelExtractingEstimator.ColorBits colorsPresent = ImagePixelExtractingEstimator.Defaults.Colors,
            ImagePixelExtractingEstimator.ColorsOrder orderOfColors = ImagePixelExtractingEstimator.Defaults.Order,
            bool interleavedColors = ImagePixelExtractingEstimator.Defaults.Interleave,
            float scaleImage = VectorToImageConvertingEstimator.Defaults.Scale,
            float offsetImage = VectorToImageConvertingEstimator.Defaults.Offset,
            int defaultAlpha = VectorToImageConvertingEstimator.Defaults.DefaultAlpha,
            int defaultRed = VectorToImageConvertingEstimator.Defaults.DefaultRed,
            int defaultGreen = VectorToImageConvertingEstimator.Defaults.DefaultGreen,
            int defaultBlue = VectorToImageConvertingEstimator.Defaults.DefaultBlue)
            => new VectorToImageConvertingEstimator(CatalogUtils.GetEnvironment(catalog), imageHeight, imageWidth, outputColumnName, inputColumnName, colorsPresent, orderOfColors, interleavedColors, scaleImage, offsetImage);
    }
}
