// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Transforms.Image;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using static Microsoft.ML.Transforms.Image.ImagePixelExtractingEstimator;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Provide interfaces for imaging operations.
    /// </summary>
    public sealed class MLImage : IDisposable
    {
        private SKBitmap _image;
        private MLPixelFormat _pixelFormat;
        private string _tag;

        // disallow instantiating image object from the default constructor
        private MLImage()
        {
        }

        /// <summary>
        /// Create a new MLImage instance from a stream.
        /// </summary>
        /// <param name="imageStream">The stream to create the image from.</param>
        /// <returns>MLImage object.</returns>
        public static MLImage CreateFromStream(Stream imageStream)
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

            return new MLImage(image);
        }

        /// <summary>
        /// Create a new MLImage instance from a stream.
        /// </summary>
        /// <param name="imagePath">The image file path to create the image from.</param>
        /// <returns>MLImage object.</returns>
        public static MLImage CreateFromFile(string imagePath)
        {
            if (imagePath is null)
            {
                throw new ArgumentNullException(nameof(imagePath));
            }

            SKBitmap image = SKBitmap.Decode(imagePath);
            if (image is null)
            {
                throw new ArgumentException($"Invalid path", nameof(imagePath));
            }

            return new MLImage(image);
        }

        /// <summary>
        /// Creates MLImage object from the pixel data span.
        /// </summary>
        /// <param name="width">The width of the image in pixels.</param>
        /// <param name="height">The height of the image in pixels.</param>
        /// <param name="pixelFormat">The pixel format to create the image with.</param>
        /// <param name="imagePixelData">The pixels data to create the image from.</param>
        /// <returns>MLImage object.</returns>
        public static unsafe MLImage CreateFromPixels(int width, int height, MLPixelFormat pixelFormat, ReadOnlySpan<byte> imagePixelData)
        {
            if (pixelFormat != MLPixelFormat.Bgra32 && pixelFormat != MLPixelFormat.Rgba32)
            {
                throw new ArgumentException($"Unsupported pixel format", nameof(pixelFormat));
            }

            if (width <= 0)
            {
                throw new ArgumentException($"Invalid width value.", nameof(width));
            }

            if (height <= 0)
            {
                throw new ArgumentException($"Invalid height value.", nameof(height));
            }

            if (imagePixelData.Length != width * height * 4)
            {
                throw new ArgumentException($"Invalid {nameof(imagePixelData)} buffer size.");
            }

            SKBitmap image = new SKBitmap(new SKImageInfo(width, height, pixelFormat == MLPixelFormat.Bgra32 ? SKColorType.Bgra8888 : SKColorType.Rgba8888));

            Debug.Assert(image.Info.BitsPerPixel == 32);
            Debug.Assert(image.RowBytes * image.Height == width * height * 4);

            imagePixelData.CopyTo(new Span<byte>(image.GetPixels().ToPointer(), image.Width * image.Height * 4));

            return new MLImage(image);
        }

        /// <summary>
        /// Gets the pixel format for this Image.
        /// </summary>
        public MLPixelFormat PixelFormat
        {
            get
            {
                ThrowInvalidOperationExceptionIfDisposed();
                return _pixelFormat;
            }

            private set
            {
                Debug.Assert(_image is not null);
                _pixelFormat = value;
            }
        }

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "<Pending>")]
        public byte[] GetBGRPixels
        {
            get
            {
                ThrowInvalidOperationExceptionIfDisposed();

                // 3 is because we only want RGB not alpha channels
                byte[] pixels = new byte[Height * Width * 3];

                var pixelData = _image.Pixels;
                int idx = 0;
                for (int i = 0; i < Height * Width * 3;)
                {

                    pixels[i++] = pixelData[idx].Blue;
                    pixels[i++] = pixelData[idx].Green;
                    pixels[i++] = pixelData[idx++].Red;
                }

                return pixels;
            }
        }

        /// <summary>
        /// Gets the image pixel data.
        /// </summary>
        public unsafe ReadOnlySpan<byte> Pixels
        {
            get
            {
                ThrowInvalidOperationExceptionIfDisposed();
                Debug.Assert(_image.Info.BytesPerPixel == 4);

                var pixelsPtr = _image.GetPixels();
                if (pixelsPtr == IntPtr.Zero || _image.ByteCount <= 0)
                    throw new InvalidOperationException("Pixel data is unavailable.");

                return new ReadOnlySpan<byte>(pixelsPtr.ToPointer(), _image.ByteCount);
            }
        }

        /// <summary>
        /// Gets or sets the image tag.
        /// </summary>
        public string Tag
        {
            get
            {
                ThrowInvalidOperationExceptionIfDisposed();
                return _tag;
            }

            set
            {
                ThrowInvalidOperationExceptionIfDisposed();
                _tag = value;
            }
        }

        /// <summary>
        /// Gets the image width in pixels.
        /// </summary>
        public int Width
        {
            get
            {
                ThrowInvalidOperationExceptionIfDisposed();
                return _image.Width;
            }
        }

        /// <summary>
        /// Gets the image height in pixels.
        /// </summary>
        public int Height
        {
            get
            {
                ThrowInvalidOperationExceptionIfDisposed();
                return _image.Height;
            }
        }

        /// <summary>
        /// Gets how many bits per pixel used by current image object.
        /// </summary>
        public int BitsPerPixel
        {
            get
            {
                ThrowInvalidOperationExceptionIfDisposed();
                Debug.Assert(_image.Info.BitsPerPixel == 32);
                return _image.Info.BitsPerPixel;
            }
        }

        /// <summary>
        /// Save the current image to a file.
        /// </summary>
        /// <param name="imagePath">The path of the file to save the image to.</param>
        /// <remarks>The saved image encoding will be detected from the file extension.</remarks>
        public void Save(string imagePath)
        {
            ThrowInvalidOperationExceptionIfDisposed();
            string ext = Path.GetExtension(imagePath);

            if (!_extensionToEncodingFormat.TryGetValue(ext, out SKEncodedImageFormat encodingFormat))
            {
                throw new ArgumentException($"Path has invalid image file extension.", nameof(imagePath));
            }

            using SKData data = _image.Encode(encodingFormat, 100);
            if (data is null)
            {
                throw new ArgumentException($"Saving image with the format '{ext}' is not supported. Try save it with `Jpeg`, `Png`, or `Webp` format.", nameof(imagePath));
            }

            using var stream = new FileStream(imagePath, FileMode.Create, FileAccess.Write);
            data.SaveTo(stream);
        }

        /// <summary>
        /// Disposes the image.
        /// </summary>
        public void Dispose()
        {
            if (_image != null)
            {
                _image.Dispose();
                _image = null;
            }
        }

        private MLImage(SKBitmap image)
        {
            _image = image;

            PixelFormat = _image.Info.ColorType switch
            {
                SKColorType.Bgra8888 => MLPixelFormat.Bgra32,
                SKColorType.Rgba8888 => MLPixelFormat.Rgba32,
                _ => CloneImageToSupportedFormat()
            };
        }

        private MLPixelFormat CloneImageToSupportedFormat()
        {
            Debug.Assert(_image.Info.ColorType != SKColorType.Bgra8888 && _image.Info.ColorType != SKColorType.Rgba8888);

            if (!_image.CanCopyTo(SKColorType.Bgra8888))
            {
                throw new InvalidOperationException("Unsupported image format.");
            }

            SKBitmap image1 = _image.Copy(SKColorType.Bgra8888);
            _image.Dispose();
            _image = image1;
            return MLPixelFormat.Bgra32;
        }

        internal MLImage CloneWithResizing(int width, int height, ImageResizeMode mode)
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

            return new MLImage(image);
        }

        private SKBitmap ResizeFull(int width, int height) => _image.Resize(new SKSizeI(width, height), new SKSamplingOptions(SKFilterMode.Nearest));

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
            using SKImage image = SKImage.FromBitmap(_image);

            canvas.DrawImage(image, srcRect, destRect, new SKSamplingOptions(SKCubicResampler.Mitchell));

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
            using SKImage image = SKImage.FromBitmap(_image);

            canvas.DrawImage(image, srcRect, destRect, new SKSamplingOptions(SKCubicResampler.Mitchell));

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

        internal MLImage CloneWithGrayscale()
        {
            ThrowInvalidOperationExceptionIfDisposed();

            SKBitmap dst = new SKBitmap(_image.Width, _image.Height, isOpaque: true);
            using SKPaint paint = new SKPaint()
            {
                ColorFilter = _grayscaleColorMatrix,
            };

            SKBitmap destBitmap = new SKBitmap(_image.Width, _image.Height, isOpaque: true);
            using SKCanvas canvas = new SKCanvas(destBitmap);
            canvas.DrawBitmap(_image, 0f, 0f, paint: paint);
            return new MLImage(destBitmap);
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

        private void ThrowInvalidOperationExceptionIfDisposed()
        {
            if (_image is null)
            {
                throw new InvalidOperationException("Object is disposed.");
            }
        }
    }

    /// <summary>
    /// Specifies the format of the color data for each pixel in the image.
    /// </summary>
    public enum MLPixelFormat
    {
        /// <summary>
        /// The pixel format is unknown.
        /// </summary>
        Unknown,

        /// <summary>
        /// Specifies that the format is 32 bits per pixel; 8 bits each are used for the blue, green, red, and alpha components.
        /// The color components are stored in blue, green, red, and alpha order
        /// </summary>
        Bgra32,

        /// <summary>
        /// Specifies that the format is 32 bits per pixel; 8 bits each are used for the red, green, blue, and alpha components.
        /// The color components are stored in red, green, blue, and alpha order
        /// </summary>
        Rgba32
    }

    /// <summary>
    /// The mode to decide how the image should be resized.
    /// </summary>
    internal enum ImageResizeMode
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
}
