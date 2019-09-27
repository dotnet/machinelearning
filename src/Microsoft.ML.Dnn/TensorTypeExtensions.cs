using System;
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
            if (typeof(T).as_dtype() == tensor.dtype)
            {
                unsafe
                {
                    dst = *(T*)tensor.buffer;
                }
            }
        }

        public static void ToSpan<T>(this Tensor tensor, Span<T> values) where T: unmanaged
        {
            if (typeof(T).as_dtype() == tensor.dtype)
            {
                unsafe
                {
                    var len = (long)tensor.size;
                    fixed (T* dst = values)
                    {
                        var src = (T*)tensor.buffer;
                        len *= ((long)tensor.itemsize);
                        System.Buffer.MemoryCopy(src, dst, len, len);
                    }
                }
            }
        }

        public static void ToArray<T>(this Tensor tensor, ref T[] array) where T : unmanaged
        {
            ulong arrayLen = 0;
            if (array != null)
                arrayLen = (ulong)array.Length;

            if (array == null || arrayLen == 0 || arrayLen < tensor.size)
            {
                array = new T[tensor.size];
                arrayLen = tensor.size;
            }

            if (typeof(T).as_dtype() == tensor.dtype)
            {
                unsafe
                {
                    var len = (long)tensor.size;
                    fixed (T* dst = array)
                    {
                        var src = (T*)tensor.buffer;
                        len *= ((long)tensor.itemsize);
                        System.Buffer.MemoryCopy(src, dst, len, len);
                    }
                }
            }
        }
    }
}
