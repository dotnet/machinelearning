// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Internal.Utilities;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;
using Tensorflow;

namespace Microsoft.ML.Transforms
{
    [BestFriend]
    internal static class TensorTypeExtensions
    {
        public static void ToScalar<T>(this Tensor tensor, ref T dst) where T : unmanaged
        {
            if (typeof(T).as_dtype() != tensor.dtype)
                throw new NotSupportedException();

            unsafe
            {
                dst = *(T*)tensor.buffer;
            }

        }

        public static void CopyTo<T>(this Tensor tensor, Span<T> values) where T: unmanaged
        {
            if (typeof(T).as_dtype() != tensor.dtype)
                throw new NotSupportedException();

            unsafe
            {
                var len = checked((int)tensor.size);
                var src = (T*)tensor.buffer;
                var span = new Span<T>(src, len);
                span.CopyTo(values);
            }
        }

        public static void ToArray<T>(this Tensor tensor, ref T[] array) where T : unmanaged
        {
            Utils.EnsureSize(ref array, (int)tensor.size, (int)tensor.size, false);
            var span = new Span<T>(array);

            CopyTo(tensor, span);
        }
    }
}
