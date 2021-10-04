// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Text;
using Microsoft.ML.Internal.Utilities;
using Tensorflow;
using Utils = Microsoft.ML.Internal.Utilities.Utils;

namespace Microsoft.ML.TensorFlow
{
    [BestFriend]
    internal static class TensorTypeExtensions
    {
        public static void ToScalar<T>(this Tensor tensor, ref T dst) where T : unmanaged
        {
            //In ML.NET we are using ReadOnlyMemory<Char> to store string data but ReadOnlyMemory<Char>
            //is not valid data type for tensorflow.net and exception will thrown if we call as_dtype method
            //so we specially deal with string type here.
            //Get string data first then convert to ReadOnlyMemory<Char> and assign value to dst.
            if (typeof(T) == typeof(ReadOnlyMemory<char>))
            {
                dst = (T)(object)tensor.StringData()[0].AsMemory();
                return;
            }

            if (typeof(T).as_dtype() != tensor.dtype)
                throw new NotSupportedException();

            unsafe
            {
                dst = *(T*)tensor.buffer;
            }

        }

        public static void CopyTo<T>(this Tensor tensor, Span<T> values) where T : unmanaged
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
