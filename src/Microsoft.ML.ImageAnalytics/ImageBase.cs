// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Provides the base class for the image provider which allow registering a provider to use instead of the default provider.
    /// </summary>
    internal abstract class ImageProvider
    {
        internal static ImageProvider DefaultProvider { get; set; }

        /// <summary>
        /// Register an image provider to use instead of the default provider.
        /// </summary>
        /// <param name="provider">A provider to use for imaging operations.</param>
        public void RegisterDefaultProvider(ImageProvider provider) => DefaultProvider = provider;

        /// <summary>
        /// Create image object from a stream.
        /// </summary>
        /// <param name="imageStream">The stream to create the image from.</param>
        /// <returns>Image object.</returns>
        public abstract ImageBase CreateImageFromStream(Stream imageStream);

        /// <summary>
        /// Create image object from the pixel data buffer span.
        /// </summary>
        /// <param name="width">The width of the image in pixels.</param>
        /// <param name="height">The height of the image in pixels.</param>
        /// <param name="imagePixelData">The pixels data to create the image from.</param>
        /// <returns>Image object.</returns>
        public abstract ImageBase CreateBgra32ImageFromPixelData(int width, int height, Span<byte> imagePixelData);
    }

    /// <summary>
    /// The mode to decide how the image should be resized.
    /// </summary>
    public enum ImageResizeMode
    {
        /// <summary>
        /// Pads the resized image to fit the bounds of its container.
        /// </summary>
        Pad,

        /// <summary>
        /// Ignore aspect ratio and squeeze/stretch into target dimensions.
        /// </summary>
        Fill,

        /// <summary>
        /// Resized image to fit the bounds of its container using cropping with top anchor.
        /// </summary>
        CropAnchorTop,

        /// <summary>
        /// Resized image to fit the bounds of its container using cropping with bottom anchor.
        /// </summary>
        CropAnchorBottom,

        /// <summary>
        /// Resized image to fit the bounds of its container using cropping with left anchor.
        /// </summary>
        CropAnchorLeft,

        /// <summary>
        /// Resized image to fit the bounds of its container using cropping with right anchor.
        /// </summary>
        CropAnchorRight,

        /// <summary>
        /// Resized image to fit the bounds of its container using cropping with central anchor.
        /// </summary>
        CropAnchorCentral
    }

    /// <summary>
    /// Base class provide all interfaces for imaging operations.
    /// </summary>
    public abstract class ImageBase : IDisposable
    {
        /// <summary>
        /// Gets or sets the image tag.
        /// </summary>
        public string Tag { get; set; }

        /// <summary>
        /// Gets the image width in pixels.
        /// </summary>
        public abstract int Width { get; }

        /// <summary>
        /// Gets the image height in pixels.
        /// </summary>
        public abstract int Height { get; }

        /// <summary>
        /// Gets how many bits per pixel used by current image object.
        /// </summary>
        public abstract int BitsPerPixel { get; }

        /// <summary>
        /// Create image object from a stream.
        /// </summary>
        /// <param name="imageStream">The stream to create the image from.</param>
        /// <returns>Image object.</returns>
        public static ImageBase CreateFromStream(Stream imageStream)
        {
            ImageProvider provider = ImageProvider.DefaultProvider;
            return provider is not null ? provider.CreateImageFromStream(imageStream) : SkiaSharpImage.Create(imageStream);
        }

        /// <summary>
        /// Create BRGA32 pixel format image object from the pixel data buffer span.
        /// </summary>
        /// <param name="width">The width of the image in pixels.</param>
        /// <param name="height">The height of the image in pixels.</param>
        /// <param name="imagePixelData">The pixels data to create the image from.</param>
        /// <returns>Image object.</returns>
        public static ImageBase CreateBgra32Image(int width, int height, Span<byte> imagePixelData)
        {
            ImageProvider provider = ImageProvider.DefaultProvider;
            return provider is not null ? provider.CreateBgra32ImageFromPixelData(width, height, imagePixelData) : SkiaSharpImage.CreateFromPixelData(width, height, imagePixelData);
        }

        /// <summary>
        /// Clones the current image with resizing it.
        /// </summary>
        /// <param name="width">The new width of the image.</param>
        /// <param name="height">The new height of the image.</param>
        /// <param name="mode">How to resize the image.</param>
        /// <returns>The new cloned image.</returns>
        public abstract ImageBase CloneWithResizing(int width, int height, ImageResizeMode mode);

        /// <summary>
        /// Clones the current image with grayscale.
        /// </summary>
        /// <returns>The new cloned image.</returns>
        public abstract ImageBase CloneWithGrayscale();

        /// <summary>
        /// Gets the image pixel data and how the colors are ordered in the used pixel format.
        /// </summary>
        /// <param name="alphaIndex">The index of the alpha in the pixel format.</param>
        /// <param name="redIndex">The index of the red color in the pixel format.</param>
        /// <param name="greenIndex">The index of the green color in the pixel format.</param>
        /// <param name="blueIndex">The index of the blue color in the pixel format.</param>
        /// <returns>The buffer containing the image pixel data.</returns>
        public abstract ReadOnlySpan<byte> Get32bbpImageData(out int alphaIndex, out int redIndex, out int greenIndex, out int blueIndex);

        /// <summary>
        /// Save the current image to a file.
        /// </summary>
        /// <param name="imagePath">The path of the file to save the image to.</param>
        /// <remarks>The saved image encoding will be detected from the file extension.</remarks>
        public abstract void Save(string imagePath);

        /// <summary>
        /// Releases the unmanaged resources used by the image object and optionally releases the managed resources.
        /// </summary>
        /// <param name="disposing">true to release both managed and unmanaged resources; false to release only unmanaged resources.</param>
        protected virtual void Dispose(bool disposing)
        {
        }

        /// <summary>
        /// Releases all resources used by the image object.
        /// </summary>
        public void Dispose() => Dispose(disposing: true);
    }

    internal class SkiaSharpImage : ImageBase
    {
        private SKBitmap _image;

        private SkiaSharpImage(SKBitmap image)
        {
            Debug.Assert(image is not null);

            // Most of the time SkiaSharp create images with Bgra8888 or Rgba8888 pixel format.
            if (image.Info.ColorType != SKColorType.Bgra8888 && image.Info.ColorType != SKColorType.Rgba8888)
            {
                if (!image.CanCopyTo(SKColorType.Bgra8888))
                {
                    throw new InvalidOperationException("Unsupported image format.");
                }

                SKBitmap image1 = image.Copy(SKColorType.Bgra8888);
                image.Dispose();
                image = image1;
            }

            _image = image;
        }

        public static ImageBase Create(Stream imageStream)
        {
            if (imageStream is null)
            {
                throw new ArgumentNullException(nameof(imageStream));
            }

            SKBitmap image = SKBitmap.Decode(imageStream);
            if (image is null)
            {
                throw new ArgumentException($"Invalid input stream contents", nameof(imageStream));
            }

            return new SkiaSharpImage(image);
        }

        public static unsafe ImageBase CreateFromPixelData(int width, int height, Span<byte> imagePixelData)
        {
            if (imagePixelData.Length != width * height * 4)
            {
                throw new ArgumentException($"Invalid {nameof(imagePixelData)} buffer size.");
            }

            SKBitmap image = new SKBitmap(new SKImageInfo(width, height, SKColorType.Bgra8888));

            Debug.Assert(image.Info.BitsPerPixel == 32);
            Debug.Assert(image.RowBytes * image.Height == width * height * 4);

            imagePixelData.CopyTo(new Span<byte>(image.GetPixels().ToPointer(), image.Width * image.Height * 4));

            return new SkiaSharpImage(image);
        }

        public override ReadOnlySpan<byte> Get32bbpImageData(out int alphaIndex, out int redIndex, out int greenIndex, out int blueIndex)
        {
            ThrowInvalidOperationExceptionIfDisposed();

            if (_image.Info.ColorType == SKColorType.Rgba8888)
            {
                redIndex = 0;
                greenIndex = 1;
                blueIndex = 2;
                alphaIndex = 3;
            }
            else
            {
                Debug.Assert(_image.Info.ColorType == SKColorType.Bgra8888);
                blueIndex = 0;
                greenIndex = 1;
                redIndex = 2;
                alphaIndex = 3;
            }

            Debug.Assert(_image.Info.BytesPerPixel == 4);

            return _image.GetPixelSpan();
        }

        public override ImageBase CloneWithResizing(int width, int height, ImageResizeMode mode)
        {
            ThrowInvalidOperationExceptionIfDisposed();

            SKBitmap image = mode switch
            {
                ImageResizeMode.Pad => ResizeWithPadding(width, height),
                ImageResizeMode.Fill => ResizeFull(width, height),
                >= ImageResizeMode.CropAnchorTop and <= ImageResizeMode.CropAnchorCentral => ResizeWithCrop(width, height, mode),
                _ => throw new ArgumentException($"Invalid resize mode value.", nameof(mode))
            };

            if (image is null)
            {
                throw new InvalidOperationException($"Couldn't resize the image");
            }

            return new SkiaSharpImage(image);
        }

        private SKBitmap ResizeFull(int width, int height) => _image.Resize(new SKSizeI(width, height), SKFilterQuality.None);

        private SKBitmap ResizeWithPadding(int width, int height)
        {
            float widthAspect = (float)width / _image.Width;
            float heightAspect = (float)height / _image.Height;
            int destX = 0;
            int destY = 0;
            float aspect;

            if (heightAspect < widthAspect)
            {
                aspect = heightAspect;
                destX = (int)((width - (_image.Width * aspect)) / 2);
            }
            else
            {
                aspect = widthAspect;
                destY = (int)((height - (_image.Height * aspect)) / 2);
            }

            int destWidth = (int)(_image.Width * aspect);
            int destHeight = (int)(_image.Height * aspect);

            SKBitmap destBitmap = new SKBitmap(width, height, isOpaque: true);
            SKRect srcRect = new SKRect(0, 0, _image.Width, _image.Height);
            SKRect destRect = new SKRect(destX, destY, destX + destWidth, destY + destHeight);

            using SKCanvas canvas = new SKCanvas(destBitmap);
            using SKPaint paint = new SKPaint() { FilterQuality = SKFilterQuality.High };

            canvas.DrawBitmap(_image, srcRect, destRect, paint);

            return destBitmap;
        }

        private SKBitmap ResizeWithCrop(int width, int height, ImageResizeMode mode)
        {
            float widthAspect = (float)width / _image.Width;
            float heightAspect = (float)height / _image.Height;
            int destX = 0;
            int destY = 0;
            float aspect;

            if (heightAspect < widthAspect)
            {
                aspect = widthAspect;
                switch (mode)
                {
                    case ImageResizeMode.CropAnchorTop:
                        destY = 0;
                        break;
                    case ImageResizeMode.CropAnchorBottom:
                        destY = (int)(height - (_image.Height * aspect));
                        break;
                    default:
                        destY = (int)((height - (_image.Height * aspect)) / 2);
                        break;
                }
            }
            else
            {
                aspect = heightAspect;
                switch (mode)
                {
                    case ImageResizeMode.CropAnchorLeft:
                        destX = 0;
                        break;
                    case ImageResizeMode.CropAnchorRight:
                        destX = (int)(width - (_image.Width * aspect));
                        break;
                    default:
                        destX = (int)((width - (_image.Width * aspect)) / 2);
                        break;
                }
            }

            int destWidth = (int)(_image.Width * aspect);
            int destHeight = (int)(_image.Height * aspect);

            SKBitmap dst = new SKBitmap(width, height, isOpaque: true);

            SKRect srcRect = new SKRect(0, 0, _image.Width, _image.Height);
            SKRect destRect = new SKRect(destX, destY, destX + destWidth, destY + destHeight);

            using SKCanvas canvas = new SKCanvas(dst);
            using SKPaint paint = new SKPaint() { FilterQuality = SKFilterQuality.High };

            canvas.DrawBitmap(_image, srcRect, destRect, paint);

            return dst;
        }

        // This matrix get multiplied to every pixel matrix [R G B A W] to average the colors values to get the grayscale effect.
        private static readonly SKColorFilter _grayscaleColorMatrix = SKColorFilter.CreateColorMatrix(new float[]
                                                                        {
                                                                            0.3f, 0.59f, 0.11f, 0, 0,
                                                                            0.3f, 0.59f, 0.11f, 0, 0,
                                                                            0.3f, 0.59f, 0.11f, 0, 0,
                                                                            0,    0,     0,     1, 0
                                                                        });

        public override ImageBase CloneWithGrayscale()
        {
            ThrowInvalidOperationExceptionIfDisposed();

            SKBitmap dst = new SKBitmap(_image.Width, _image.Height, isOpaque: true);
            using SKPaint paint = new SKPaint()
            {
                ColorFilter = _grayscaleColorMatrix,
                FilterQuality = SKFilterQuality.High
            };

            SKBitmap destBitmap = new SKBitmap(_image.Width, _image.Height, isOpaque: true);
            using SKCanvas canvas = new SKCanvas(destBitmap);
            canvas.DrawBitmap(_image, 0f, 0f, paint: paint);
            return new SkiaSharpImage(destBitmap);
        }

        public override int Width
        {
            get
            {
                ThrowInvalidOperationExceptionIfDisposed();
                return _image.Width;
            }
        }

        public override int Height
        {
            get
            {
                ThrowInvalidOperationExceptionIfDisposed();
                return _image.Height;
            }
        }

        public override int BitsPerPixel
        {
            get
            {
                ThrowInvalidOperationExceptionIfDisposed();
                Debug.Assert(_image.Info.BitsPerPixel == 32);
                return _image.Info.BitsPerPixel;
            }
        }

        private static readonly Dictionary<string, SKEncodedImageFormat> _extensionToEncodingFormat = new Dictionary<string, SKEncodedImageFormat>(StringComparer.OrdinalIgnoreCase)
        {
            { ".bmp", SKEncodedImageFormat.Bmp },
            { ".png", SKEncodedImageFormat.Png },
            { ".jpg", SKEncodedImageFormat.Jpeg },
            { ".jpeg", SKEncodedImageFormat.Jpeg },
            { ".gif", SKEncodedImageFormat.Gif },
            { ".ico", SKEncodedImageFormat.Ico },
            { ".astc", SKEncodedImageFormat.Astc },
            { ".avif", SKEncodedImageFormat.Avif },
            { ".dng", SKEncodedImageFormat.Dng },
            { ".heif", SKEncodedImageFormat.Heif },
            { ".ktx", SKEncodedImageFormat.Ktx },
            { ".pkm", SKEncodedImageFormat.Pkm },
            { ".wbmp", SKEncodedImageFormat.Wbmp },
            { ".webp", SKEncodedImageFormat.Webp }
        };

        public override void Save(string imagePath)
        {
            ThrowInvalidOperationExceptionIfDisposed();
            string ext = Path.GetExtension(imagePath);

            if (!_extensionToEncodingFormat.TryGetValue(ext, out SKEncodedImageFormat encodingFormat))
            {
                throw new ArgumentException($"Path with invalid image file extension.", nameof(imagePath));
            }

            using var stream = new FileStream(imagePath, FileMode.Create, FileAccess.Write);
            SKData data = _image.Encode(encodingFormat, 100);
            data.SaveTo(stream);
        }

        protected override void Dispose(bool disposing)
        {
            if (_image != null)
            {
                _image.Dispose();
                _image = null;
            }
        }

        private void ThrowInvalidOperationExceptionIfDisposed()
        {
            if (_image is null)
            {
                throw new InvalidOperationException("Object is disposed.");
            }
        }
    }
}
