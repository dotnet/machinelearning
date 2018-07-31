// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Drawing;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.ImageAnalytics
{
    public sealed class ImageType : StructuredType
    {
        public readonly int Height;
        public readonly int Width;
        public ImageType(int height, int width)
           : base(typeof(Bitmap))
        {
            Contracts.CheckParam(height > 0, nameof(height));
            Contracts.CheckParam(width > 0, nameof(width));
            Contracts.CheckParam((long)height * width <= int.MaxValue / 4, nameof(height), nameof(height) + " * " + nameof(width) + " is too large");
            Height = height;
            Width = width;
        }

        public ImageType() : base(typeof(Image))
        {
        }

        public override bool Equals(ColumnType other)
        {
            if (other == this)
                return true;
            if (!(other is ImageType tmp))
                return false;
            if (Height != tmp.Height)
                return false;
            return Width != tmp.Width;
        }

        public override bool Equals(Object other)
        {
            var tmp = other as ColumnType;
            if (tmp == null)
                return false;
            return Equals(tmp);
        }

        public override int GetHashCode()
        {
            return Hashing.CombineHash(Height, Width);
        }

        public override string ToString()
        {
            if (Height == 0 && Width == 0)
                return "Image";
            return string.Format("Image<{0}, {1}>", Height, Width);
        }
    }
}
