// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Drawing;
using Microsoft.ML.StaticPipe;

namespace Microsoft.ML.Runtime.ImageAnalytics
{
    /// <summary>
    /// A type used in the generic argument to <see cref="Scalar{T}"/>. We must simultaneously distinguish
    /// between a <see cref="ImageType"/> of fixed (with <see cref="Bitmap"/>) and unfixed (with this type),
    /// in the static pipelines.
    /// </summary>
    public class UnknownSizeBitmap { private UnknownSizeBitmap() { } }

    /// <summary>
    /// Extension methods for the static-pipeline over <see cref="PipelineColumn"/> objects.
    /// </summary>
    public static class ImageStaticPipe
    {
        /// <summary>
        /// Load an image from an input column that holds the paths to images.
        /// </summary>
        /// <param name="path">The scalar text column that holds paths to the images</param>
        /// <param name="relativeTo">If specified, paths are considered to be relative to this directory.
        /// However, since the transform can be persisted across machines, it is generally considered more
        /// safe for users to simply always make their input paths absolute.</param>
        /// <returns>The loaded images</returns>
        /// <seealso cref="ImageLoadingEstimator"/>
        public static Custom<UnknownSizeBitmap> LoadAsImage(this Scalar<string> path, string relativeTo = null)
        {
            Contracts.CheckValue(path, nameof(path));
            Contracts.CheckValueOrNull(relativeTo);
            return new ImageLoadingEstimator.OutPipelineColumn(path, relativeTo);
        }

        /// <summary>
        /// Converts the image to grayscale.
        /// </summary>
        /// <param name="input">The image to convert</param>
        /// <returns>The grayscale images</returns>
        /// <seealso cref="ImageGrayscalingEstimator"/>
        public static Custom<UnknownSizeBitmap> AsGrayscale(this Custom<UnknownSizeBitmap> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new ImageGrayscalingEstimator.OutPipelineColumn<UnknownSizeBitmap>(input);
        }

        /// <summary>
        /// Converts the image to grayscale.
        /// </summary>
        /// <param name="input">The image to convert</param>
        /// <returns>The grayscale images</returns>
        /// <seealso cref="ImageGrayscalingEstimator"/>
        public static Custom<Bitmap> AsGrayscale(this Custom<Bitmap> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new ImageGrayscalingEstimator.OutPipelineColumn<Bitmap>(input);
        }

        /// <summary>
        /// Given a column of images of unfixed size, resize the images so they have uniform size.
        /// </summary>
        /// <param name="input">The input images</param>
        /// <param name="width">The width to resize to</param>
        /// <param name="height">The height to resize to</param>
        /// <param name="resizing">The type of resizing to do</param>
        /// <param name="cropAnchor">If cropping is necessary, at what position will the image be fixed?</param>
        /// <returns>The now uniformly sized images</returns>
        /// <seealso cref="ImageResizingEstimator"/>
        public static Custom<Bitmap> Resize(this Custom<UnknownSizeBitmap> input, int width, int height,
            ImageResizerTransform.ResizingKind resizing = ImageResizerTransform.ResizingKind.IsoCrop,
            ImageResizerTransform.Anchor cropAnchor = ImageResizerTransform.Anchor.Center)
        {
            Contracts.CheckValue(input, nameof(input));
            Contracts.CheckParam(width > 0, nameof(width), "Must be positive");
            Contracts.CheckParam(height > 0, nameof(height), "Must be positive");
            Contracts.CheckParam(Enum.IsDefined(typeof(ImageResizerTransform.ResizingKind), resizing), nameof(resizing), "Undefined value detected");
            Contracts.CheckParam(Enum.IsDefined(typeof(ImageResizerTransform.Anchor), cropAnchor), nameof(cropAnchor), "Undefined value detected");

            return new ImageResizingEstimator.OutPipelineColumn(input, width, height, resizing, cropAnchor);
        }

        /// <summary>
        /// Given a column of images, resize them to a new fixed size.
        /// </summary>
        /// <param name="input">The input images</param>
        /// <param name="width">The width to resize to</param>
        /// <param name="height">The height to resize to</param>
        /// <param name="resizing">The type of resizing to do</param>
        /// <param name="cropAnchor">If cropping is necessary, at what </param>
        /// <returns>The resized images</returns>
        /// <seealso cref="ImageResizingEstimator"/>
        public static Custom<Bitmap> Resize(this Custom<Bitmap> input, int width, int height,
            ImageResizerTransform.ResizingKind resizing = ImageResizerTransform.ResizingKind.IsoCrop,
            ImageResizerTransform.Anchor cropAnchor = ImageResizerTransform.Anchor.Center)
        {
            Contracts.CheckValue(input, nameof(input));
            Contracts.CheckParam(width > 0, nameof(width), "Must be positive");
            Contracts.CheckParam(height > 0, nameof(height), "Must be positive");
            Contracts.CheckParam(Enum.IsDefined(typeof(ImageResizerTransform.ResizingKind), resizing), nameof(resizing), "Undefined value detected");
            Contracts.CheckParam(Enum.IsDefined(typeof(ImageResizerTransform.Anchor), cropAnchor), nameof(cropAnchor), "Undefined value detected");

            return new ImageResizingEstimator.OutPipelineColumn(input, width, height, resizing, cropAnchor);
        }

        /// <summary>
        /// Vectorizes the image as the numeric values of its pixels converted and possibly transformed to floating point values.
        /// The output vector is output in height then width major order, with the channels being the most minor (if
        /// <paramref name="interleaveArgb"/> is true) or major (if <paramref name="interleaveArgb"/> is false) dimension.
        /// </summary>
        /// <param name="input">The input image to extract</param>
        /// <param name="useAlpha">Whether the alpha channel should be extracted</param>
        /// <param name="useRed">Whether the red channel should be extracted</param>
        /// <param name="useGreen">Whether the green channel should be extracted</param>
        /// <param name="useBlue">Whether the blue channel should be extracted</param>
        /// <param name="interleaveArgb">Whether the pixel values should be interleaved, as opposed to being separated by channel</param>
        /// <param name="scale">Scale the normally 0 through 255 pixel values by this amount</param>
        /// <param name="offset">Add this amount to the pixel values, before scaling</param>
        /// <returns>The vectorized image</returns>
        /// <seealso cref="ImagePixelExtractingEstimator"/>
        public static Vector<float> ExtractPixels(this Custom<Bitmap> input, bool useAlpha = false, bool useRed = true,
            bool useGreen = true, bool useBlue = true, bool interleaveArgb = false, float scale = 1.0f, float offset = 0.0f)
        {
            var colParams = new ImagePixelExtractorTransform.Column
            {
                UseAlpha = useAlpha,
                UseRed = useRed,
                UseGreen = useGreen,
                UseBlue = useBlue,
                InterleaveArgb = interleaveArgb,
                Scale = scale,
                Offset = offset,
                Convert = true
            };
            return new ImagePixelExtractingEstimator.OutPipelineColumn<float>(input, colParams);
        }

        /// <summary>
        /// Vectorizes the image as the numeric byte values of its pixels.
        /// The output vector is output in height then width major order, with the channels being the most minor (if
        /// <paramref name="interleaveArgb"/> is true) or major (if <paramref name="interleaveArgb"/> is false) dimension.
        /// </summary>
        /// <param name="input">The input image to extract</param>
        /// <param name="useAlpha">Whether the alpha channel should be extracted</param>
        /// <param name="useRed">Whether the red channel should be extracted</param>
        /// <param name="useGreen">Whether the green channel should be extracted</param>
        /// <param name="useBlue">Whether the blue channel should be extracted</param>
        /// <param name="interleaveArgb">Whether the pixel values should be interleaved, as opposed to being separated by channel</param>
        /// <returns>The vectorized image</returns>
        /// <seealso cref="ImagePixelExtractingEstimator"/>
        public static Vector<byte> ExtractPixelsAsBytes(this Custom<Bitmap> input, bool useAlpha = false, bool useRed = true,
            bool useGreen = true, bool useBlue = true, bool interleaveArgb = false)
        {
            var colParams = new ImagePixelExtractorTransform.Column
            {
                UseAlpha = useAlpha,
                UseRed = useRed,
                UseGreen = useGreen,
                UseBlue = useBlue,
                InterleaveArgb = interleaveArgb,
                Convert = false
            };
            return new ImagePixelExtractingEstimator.OutPipelineColumn<byte>(input, colParams);
        }
    }
}
