// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.ML.Internal.Utilities;

namespace Microsoft.ML.Trainers.FastTree
{
    /// <summary>
    /// This class contains extension methods that support binary serialization of some base C# types
    /// and arrays of these types.
    /// SizeInBytes - the number of bytes in the binary representation
    /// type.ToByteArray(buffer, ref position) - will write the binary representation of the type to
    ///   the byte buffer at the given position, and will increment the position to the end of
    ///   the representation
    /// byte[].ToXXX(ref position) - converts the binary representation back into the original type
    /// </summary>
    internal static class ToByteArrayExtensions
    {
        // byte

        public static int SizeInBytes(this byte a)
        {
            return sizeof(byte);
        }

        public static void ToByteArray(this byte a, byte[] buffer, ref int position)
        {
            buffer[position] = a;
            position++;
        }

        public static byte ToByte(this byte[] buffer, ref int position)
        {
            byte a = buffer[position];
            position++;
            return a;
        }

        // short

        public static int SizeInBytes(this short a)
        {
            return sizeof(short);
        }

        public static void ToByteArray(this short a, byte[] buffer, ref int position)
        {
            // Per docs, MemoryMarshal.Write<...> is safe for T: = short.
            // It writes machine-endian and handles unaligned byte buffers properly.
            MemoryMarshal.Write(buffer.AsSpan(position), ref a);
            position += sizeof(short);
        }

        public static short ToShort(this byte[] buffer, ref int position)
        {
            short a = BitConverter.ToInt16(buffer, position);
            position += sizeof(short);
            return a;
        }

        // ushort

        public static int SizeInBytes(this ushort a)
        {
            return sizeof(ushort);
        }

        public static void ToByteArray(this ushort a, byte[] buffer, ref int position)
        {
            // Per docs, MemoryMarshal.Write<...> is safe for T: = ushort.
            // It writes machine-endian and handles unaligned byte buffers properly.
            MemoryMarshal.Write(buffer.AsSpan(position), ref a);
            position += sizeof(ushort);
        }

        public static ushort ToUShort(this byte[] buffer, ref int position)
        {
            ushort a = BitConverter.ToUInt16(buffer, position);
            position += sizeof(ushort);
            return a;
        }

        // int

        public static int SizeInBytes(this int a)
        {
            return sizeof(int);
        }

        public static void ToByteArray(this int a, byte[] buffer, ref int position)
        {
            // Per docs, MemoryMarshal.Write<...> is safe for T: = int.
            // It writes machine-endian and handles unaligned byte buffers properly.
            MemoryMarshal.Write(buffer.AsSpan(position), ref a);
            position += sizeof(int);
        }

        public static int ToInt(this byte[] buffer, ref int position)
        {
            int a = BitConverter.ToInt32(buffer, position);
            position += sizeof(int);
            return a;
        }

        // uint

        public static int SizeInBytes(this uint a)
        {
            return sizeof(uint);
        }

        public static void ToByteArray(this uint a, byte[] buffer, ref int position)
        {
            // Per docs, MemoryMarshal.Write<...> is safe for T: = uint.
            // It writes machine-endian and handles unaligned byte buffers properly.
            MemoryMarshal.Write(buffer.AsSpan(position), ref a);
            position += sizeof(uint);
        }

        public static uint ToUInt(this byte[] buffer, ref int position)
        {
            uint a = BitConverter.ToUInt32(buffer, position);
            position += sizeof(uint);
            return a;
        }

        // long

        public static int SizeInBytes(this long a)
        {
            return sizeof(long);
        }

        public static void ToByteArray(this long a, byte[] buffer, ref int position)
        {
            // Per docs, MemoryMarshal.Write<...> is safe for T: = long.
            // It writes machine-endian and handles unaligned byte buffers properly.
            MemoryMarshal.Write(buffer.AsSpan(position), ref a);
            position += sizeof(long);
        }

        public static long ToLong(this byte[] buffer, ref int position)
        {
            long a = BitConverter.ToInt64(buffer, position);
            position += sizeof(long);
            return a;
        }

        // ulong

        public static int SizeInBytes(this ulong a)
        {
            return sizeof(ulong);
        }

        public static void ToByteArray(this ulong a, byte[] buffer, ref int position)
        {
            // Per docs, MemoryMarshal.Write<...> is safe for T: = ulong.
            // It writes machine-endian and handles unaligned byte buffers properly.
            MemoryMarshal.Write(buffer.AsSpan(position), ref a);
            position += sizeof(ulong);
        }

        public static ulong ToULong(this byte[] buffer, ref int position)
        {
            ulong a = BitConverter.ToUInt64(buffer, position);
            position += sizeof(ulong);
            return a;
        }

        // float

        public static int SizeInBytes(this float a)
        {
            return sizeof(float);
        }

        public static void ToByteArray(this float a, byte[] buffer, ref int position)
        {
            // Per docs, MemoryMarshal.Write<...> is safe for T: = float.
            // It writes machine-endian and handles unaligned byte buffers properly.
            MemoryMarshal.Write(buffer.AsSpan(position), ref a);
            position += sizeof(float);
        }

        public static float ToFloat(this byte[] buffer, ref int position)
        {
            float a = BitConverter.ToSingle(buffer, position);
            position += sizeof(float);
            return a;
        }

        // double

        public static int SizeInBytes(this double a)
        {
            return sizeof(double);
        }

        public static void ToByteArray(this double a, byte[] buffer, ref int position)
        {
            // Per docs, MemoryMarshal.Write<...> is safe for T: = double.
            // It writes machine-endian and handles unaligned byte buffers properly.
            MemoryMarshal.Write(buffer.AsSpan(position), ref a);
            position += sizeof(double);
        }

        public static double ToDouble(this byte[] buffer, ref int position)
        {
            double a = BitConverter.ToDouble(buffer, position);
            position += sizeof(double);
            return a;
        }

        // string

        public static int SizeInBytes(this string a)
        {
            return checked(sizeof(int) + Encoding.Unicode.GetByteCount(a));
        }

        public static void ToByteArray(this string a, byte[] buffer, ref int position)
        {
            byte[] bytes = Encoding.Unicode.GetBytes(a);
            bytes.Length.ToByteArray(buffer, ref position);
            Array.Copy(bytes, 0, buffer, position, bytes.Length);
            position += bytes.Length;
        }

        public static byte[] ToByteArray(this string a)
        {
            byte[] bytes = Encoding.Unicode.GetBytes(a);
            byte[] allBytes = new byte[bytes.Length + sizeof(int)];
            int position = 0;
            bytes.Length.ToByteArray(allBytes, ref position);
            Array.Copy(bytes, 0, allBytes, position, bytes.Length);
            return allBytes;
        }

        public static string ToString(this byte[] buffer, ref int position)
        {
            int length = buffer.ToInt(ref position);
            string a = Encoding.Unicode.GetString(buffer, position, length);
            position += length;
            return a;
        }

        // byte[]

        public static int SizeInBytes(this byte[] a)
        {
            return checked(sizeof(int) + Utils.Size(a) * sizeof(byte));
        }

        public static void ToByteArray(this byte[] a, byte[] buffer, ref int position)
        {
            a.Length.ToByteArray(buffer, ref position);
            Array.Copy(a, 0, buffer, position, a.Length);
            position += a.Length;
        }

        public static byte[] ToByteArray(this byte[] buffer, ref int position)
        {
            int length = buffer.ToInt(ref position);
            byte[] a = new byte[length];

            Array.Copy(buffer, position, a, 0, length);
            position += length;
            return a;
        }

        // short[]

        public static int SizeInBytes(this short[] a)
        {
            return checked(sizeof(int) + Utils.Size(a) * sizeof(short));
        }

        public static void ToByteArray(this short[] a, byte[] buffer, ref int position)
        {
            int length = a.Length;
            length.ToByteArray(buffer, ref position);

            // MemoryMarshal.AsBytes<short> is type-safe but could fail if the source buffer is so long
            // that its byte length can't be represented as an int32. In this case, we're ok with
            // AsBytes throwing an exception early, since we know the length of our destination byte
            // buffer is limited to an int32 length anyway.
            MemoryMarshal.AsBytes(a.AsSpan()).CopyTo(buffer.AsSpan(position));
            position += length * sizeof(short);
        }

        public static short[] ToShortArray(this byte[] buffer, ref int position)
        {
            int length = buffer.ToInt(ref position); // reading trusted length from input stream
            int byteLength = checked(length * sizeof(short)); // if this overflows, we couldn't have populated buffer anyway
            short[] a = new short[length];

            // MemoryMarshal.AsBytes<short> is type-safe. The checked block above prevents failure here.
            buffer.AsSpan(position, byteLength).CopyTo(MemoryMarshal.AsBytes(a.AsSpan()));
            position += byteLength;

            return a;
        }

        // ushort[]

        public static int SizeInBytes(this ushort[] a)
        {
            return checked(sizeof(int) + Utils.Size(a) * sizeof(ushort));
        }

        public static void ToByteArray(this ushort[] a, byte[] buffer, ref int position)
        {
            int length = a.Length;
            length.ToByteArray(buffer, ref position);

            // MemoryMarshal.AsBytes<ushort> is type-safe but could fail if the source buffer is so long
            // that its byte length can't be represented as an int32. In this case, we're ok with
            // AsBytes throwing an exception early, since we know the length of our destination byte
            // buffer is limited to an int32 length anyway.
            MemoryMarshal.AsBytes(a.AsSpan()).CopyTo(buffer.AsSpan(position));
            position += length * sizeof(ushort);
        }

        public static ushort[] ToUShortArray(this byte[] buffer, ref int position)
        {
            int length = buffer.ToInt(ref position); // reading trusted length from input stream
            int byteLength = checked(length * sizeof(ushort)); // if this overflows, we couldn't have populated buffer anyway
            ushort[] a = new ushort[length];

            // MemoryMarshal.AsBytes<ushort> is type-safe. The checked block above prevents failure here.
            buffer.AsSpan(position, byteLength).CopyTo(MemoryMarshal.AsBytes(a.AsSpan()));
            position += byteLength;

            return a;
        }

        // int[]

        public static int SizeInBytes(this int[] array)
        {
            return checked(sizeof(int) + Utils.Size(array) * sizeof(int));
        }

        public static void ToByteArray(this int[] a, byte[] buffer, ref int position)
        {
            int length = Utils.Size(a);
            length.ToByteArray(buffer, ref position);

            // MemoryMarshal.AsBytes<int> is type-safe but could fail if the source buffer is so long
            // that its byte length can't be represented as an int32. In this case, we're ok with
            // AsBytes throwing an exception early, since we know the length of our destination byte
            // buffer is limited to an int32 length anyway.
            MemoryMarshal.AsBytes(a.AsSpan()).CopyTo(buffer.AsSpan(position));
            position += length * sizeof(int);
        }

        public static int[] ToIntArray(this byte[] buffer, ref int position)
            => buffer.ToIntArray(ref position, buffer.ToInt(ref position));

        public static int[] ToIntArray(this byte[] buffer, ref int position, int length)
        {
            if (length == 0)
                return null;

            int byteLength = checked(length * sizeof(int)); // if this overflows, we couldn't have populated buffer anyway
            int[] a = new int[length];

            // MemoryMarshal.AsBytes<int> is type-safe. The checked block above prevents failure here.
            buffer.AsSpan(position, byteLength).CopyTo(MemoryMarshal.AsBytes(a.AsSpan()));
            position += byteLength;

            return a;
        }

        // uint[]

        public static int SizeInBytes(this uint[] array)
        {
            return checked(sizeof(int) + Utils.Size(array) * sizeof(uint));
        }

        public static void ToByteArray(this uint[] a, byte[] buffer, ref int position)
        {
            int length = a.Length;
            length.ToByteArray(buffer, ref position);

            // MemoryMarshal.AsBytes<uint> is type-safe but could fail if the source buffer is so long
            // that its byte length can't be represented as an int32. In this case, we're ok with
            // AsBytes throwing an exception early, since we know the length of our destination byte
            // buffer is limited to an int32 length anyway.
            MemoryMarshal.AsBytes(a.AsSpan()).CopyTo(buffer.AsSpan(position));
            position += length * sizeof(uint);
        }

        public static uint[] ToUIntArray(this byte[] buffer, ref int position)
        {
            int length = buffer.ToInt(ref position); // reading trusted length from input stream
            int byteLength = checked(length * sizeof(uint)); // if this overflows, we couldn't have populated buffer anyway
            uint[] a = new uint[length];

            // MemoryMarshal.AsBytes<uint> is type-safe. The checked block above prevents failure here.
            buffer.AsSpan(position, byteLength).CopyTo(MemoryMarshal.AsBytes(a.AsSpan()));
            position += byteLength;

            return a;
        }

        // long[]

        public static int SizeInBytes(this long[] array)
        {
            return checked(sizeof(int) + Utils.Size(array) * sizeof(long));
        }

        public static void ToByteArray(this long[] a, byte[] buffer, ref int position)
        {
            int length = a.Length;
            length.ToByteArray(buffer, ref position);

            // MemoryMarshal.AsBytes<long> is type-safe but could fail if the source buffer is so long
            // that its byte length can't be represented as an int32. In this case, we're ok with
            // AsBytes throwing an exception early, since we know the length of our destination byte
            // buffer is limited to an int32 length anyway.
            MemoryMarshal.AsBytes(a.AsSpan()).CopyTo(buffer.AsSpan(position));
            position += length * sizeof(long);
        }

        public static long[] ToLongArray(this byte[] buffer, ref int position)
        {
            int length = buffer.ToInt(ref position); // reading trusted length from input stream
            int byteLength = checked(length * sizeof(long)); // if this overflows, we couldn't have populated buffer anyway
            long[] a = new long[length];

            // MemoryMarshal.AsBytes<long> is type-safe. The checked block above prevents failure here.
            buffer.AsSpan(position, byteLength).CopyTo(MemoryMarshal.AsBytes(a.AsSpan()));
            position += byteLength;

            return a;
        }

        // ulong[]

        public static int SizeInBytes(this ulong[] array)
        {
            return checked(sizeof(int) + Utils.Size(array) * sizeof(ulong));
        }

        public static void ToByteArray(this ulong[] a, byte[] buffer, ref int position)
        {
            int length = a.Length;
            length.ToByteArray(buffer, ref position);

            // MemoryMarshal.AsBytes<ulong> is type-safe but could fail if the source buffer is so long
            // that its byte length can't be represented as an int32. In this case, we're ok with
            // AsBytes throwing an exception early, since we know the length of our destination byte
            // buffer is limited to an int32 length anyway.
            MemoryMarshal.AsBytes(a.AsSpan()).CopyTo(buffer.AsSpan(position));
            position += length * sizeof(ulong);
        }

        public static ulong[] ToULongArray(this byte[] buffer, ref int position)
        {
            int length = buffer.ToInt(ref position); // reading trusted length from input stream
            int byteLength = checked(length * sizeof(ulong)); // if this overflows, we couldn't have populated buffer anyway
            ulong[] a = new ulong[length];

            // MemoryMarshal.AsBytes<ulong> is type-safe. The checked block above prevents failure here.
            buffer.AsSpan(position, byteLength).CopyTo(MemoryMarshal.AsBytes(a.AsSpan()));
            position += byteLength;

            return a;
        }

        // float[]

        public static int SizeInBytes(this float[] array)
        {
            return checked(sizeof(int) + Utils.Size(array) * sizeof(float));
        }

        public static void ToByteArray(this float[] a, byte[] buffer, ref int position)
        {
            int length = a.Length;
            length.ToByteArray(buffer, ref position);

            // MemoryMarshal.AsBytes<float> is type-safe but could fail if the source buffer is so long
            // that its byte length can't be represented as an int32. In this case, we're ok with
            // AsBytes throwing an exception early, since we know the length of our destination byte
            // buffer is limited to an int32 length anyway.
            MemoryMarshal.AsBytes(a.AsSpan()).CopyTo(buffer.AsSpan(position));
            position += length * sizeof(float);
        }

        public static float[] ToFloatArray(this byte[] buffer, ref int position)
        {
            int length = buffer.ToInt(ref position); // reading trusted length from input stream
            int byteLength = checked(length * sizeof(float)); // if this overflows, we couldn't have populated buffer anyway
            float[] a = new float[length];

            // MemoryMarshal.AsBytes<float> is type-safe. The checked block above prevents failure here.
            buffer.AsSpan(position, byteLength).CopyTo(MemoryMarshal.AsBytes(a.AsSpan()));
            position += byteLength;

            return a;
        }

        // double[]

        public static int SizeInBytes(this double[] array)
        {
            return checked(sizeof(int) + Utils.Size(array) * sizeof(double));
        }

        public static void ToByteArray(this double[] a, byte[] buffer, ref int position)
        {
            int length = a.Length;
            length.ToByteArray(buffer, ref position);

            // MemoryMarshal.AsBytes<double> is type-safe but could fail if the source buffer is so long
            // that its byte length can't be represented as an int32. In this case, we're ok with
            // AsBytes throwing an exception early, since we know the length of our destination byte
            // buffer is limited to an int32 length anyway.
            MemoryMarshal.AsBytes(a.AsSpan()).CopyTo(buffer.AsSpan(position));
            position += length * sizeof(double);
        }

        public static double[] ToDoubleArray(this byte[] buffer, ref int position)
        {
            int length = buffer.ToInt(ref position); // reading trusted length from input stream
            int byteLength = checked(length * sizeof(double)); // if this overflows, we couldn't have populated buffer anyway
            double[] a = new double[length];

            // MemoryMarshal.AsBytes<double> is type-safe. The checked block above prevents failure here.
            buffer.AsSpan(position, byteLength).CopyTo(MemoryMarshal.AsBytes(a.AsSpan()));
            position += byteLength;

            return a;
        }

        // double[][]

        public static int SizeInBytes(this double[][] array)
        {
            if (Utils.Size(array) == 0)
                return sizeof(int);
            return checked(sizeof(int) + array.Sum(x => x.SizeInBytes()));
        }

        public static void ToByteArray(this double[][] a, byte[] buffer, ref int position)
        {
            a.Length.ToByteArray(buffer, ref position);
            for (int i = 0; i < a.Length; ++i)
            {
                a[i].ToByteArray(buffer, ref position);
            }
        }

        public static double[][] ToDoubleJaggedArray(this byte[] buffer, ref int position)
        {
            int length = buffer.ToInt(ref position); // reading trusted length from input stream
            double[][] a = new double[length][];
            for (int i = 0; i < a.Length; ++i)
            {
                a[i] = buffer.ToDoubleArray(ref position);
            }
            return a;
        }

        // string[]

        public static long SizeInBytes(this string[] array)
        {
            long length = sizeof(int);
            for (int i = 0; i < Utils.Size(array); ++i)
            {
                length += array[i].SizeInBytes();
            }
            return length;
        }

        public static void ToByteArray(this string[] a, byte[] buffer, ref int position)
        {
            Utils.Size(a).ToByteArray(buffer, ref position);
            for (int i = 0; i < Utils.Size(a); ++i)
            {
                a[i].ToByteArray(buffer, ref position);
            }
        }

        public static string[] ToStringArray(this byte[] buffer, ref int position)
        {
            int length = buffer.ToInt(ref position); // reading trusted length from input stream
            string[] a = new string[length];
            for (int i = 0; i < a.Length; ++i)
            {
                a[i] = buffer.ToString(ref position);
            }
            return a;
        }
    }
}
