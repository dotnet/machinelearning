﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Drawing;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms.Image
{
    public sealed class ImageDataViewType : StructuredDataViewType
    {
        public readonly int Height;
        public readonly int Width;
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
