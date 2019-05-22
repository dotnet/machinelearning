// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Drawing;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms.Image
{
    /// <summary>
    /// Allows a member to be marked as a <see cref="VectorDataViewType"/>, primarily allowing one to set
    /// the dimensionality of the resulting array.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property, AllowMultiple = false, Inherited = true)]
    public sealed class ImageTypeAttribute : Attribute
    {
        /// <summary>
        /// The height of the image type.
        /// </summary>
        internal int Height { get; }

        /// <summary>
        /// The width of the image type.
        /// </summary>
        internal int Width { get; }

        /// <summary>
        /// Create an image type without knowing its height and width.
        /// </summary>
        public ImageTypeAttribute()
        {
        }

        /// <summary>
        /// Create an image type with known height and width.
        /// </summary>
        public ImageTypeAttribute(int height, int width)
        {
            Contracts.CheckParam(width > 0, nameof(width), "Should be positive number");
            Contracts.CheckParam(height > 0, nameof(height), "Should be positive number");
            Height = height;
            Width = width;
        }

        public override int GetHashCode()
        {
            return Hashing.CombineHash(Height.GetHashCode(), Width.GetHashCode());
        }
    }

    public sealed class ImageDataViewType : StructuredDataViewType
    {
        public readonly int Height;
        public readonly int Width;

        static ImageDataViewType()
        {
            DataViewTypeManager.Register(new ImageDataViewType(), typeof(Bitmap));
        }

        public ImageDataViewType(int height, int width)
           : base(typeof(Bitmap))
        {
            Contracts.CheckParam(height > 0, nameof(height), "Must be positive.");
            Contracts.CheckParam(width > 0, nameof(width), " Must be positive.");
            Contracts.CheckParam((long)height * width <= int.MaxValue / 4, nameof(height), nameof(height) + " * " + nameof(width) + " is too large.");

            Height = height;
            Width = width;
        }

        public ImageDataViewType() : base(typeof(Bitmap))
        {
        }

        public override bool Equals(DataViewType other)
        {
            if (other == this)
                return true;
            if (!(other is ImageDataViewType tmp))
                return false;
            if (Height != tmp.Height)
                return false;
            return Width == tmp.Width;
        }

        public override bool Equals(object other)
        {
            return other is DataViewType tmp && Equals(tmp);
        }

        public override int GetHashCode()
        {
            return Hashing.CombineHash(Height.GetHashCode(), Width.GetHashCode());
        }

        public override string ToString()
        {
            if (Height == 0 && Width == 0)
                return "Image";
            return string.Format("Image<{0}, {1}>", Height, Width);
        }
    }
}
