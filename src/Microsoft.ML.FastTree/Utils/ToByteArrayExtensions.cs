// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Text;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.FastTree.Internal
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
    public static class ToByteArrayExtensions
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

        public unsafe static void ToByteArray(this short a, byte[] buffer, ref int position)
        {
            fixed (byte* pBuffer = buffer)
            {
                short* pDest = (short*)(pBuffer + position);
                *pDest = a;
            }
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

        public unsafe static void ToByteArray(this ushort a, byte[] buffer, ref int position)
        {
            fixed (byte* pBuffer = buffer)
            {
                ushort* pDest = (ushort*)(pBuffer + position);
                *pDest = a;
            }
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

        public unsafe static void ToByteArray(this int a, byte[] buffer, ref int position)
        {
            fixed (byte* pBuffer = buffer)
            {
                int* pDest = (int*)(pBuffer + position);
                *pDest = a;
            }
            position += sizeof(int);
        }

        public unsafe static int ToInt(this byte[] buffer, ref int position)
        {
            int a;
            fixed (byte* pBuffer = buffer)
            {
                int* pIntBuffer = (int*)(pBuffer + position);
                a = *pIntBuffer;
            }
            position += sizeof(int);
            return a;
        }

        // uint

        public static int SizeInBytes(this uint a)
        {
            return sizeof(uint);
        }

        public unsafe static void ToByteArray(this uint a, byte[] buffer, ref int position)
        {
            fixed (byte* pBuffer = buffer)
            {
                uint* pDest = (uint*)(pBuffer + position);
                *pDest = a;
            }
            position += sizeof(uint);
        }

        public unsafe static uint ToUInt(this byte[] buffer, ref int position)
        {
            uint a;
            fixed (byte* pBuffer = buffer)
            {
                uint* pIntBuffer = (uint*)(pBuffer + position);
                a = *pIntBuffer;
            }
            position += sizeof(uint);
            return a;
        }

        // long

        public static int SizeInBytes(this long a)
        {
            return sizeof(long);
        }

        public unsafe static void ToByteArray(this long a, byte[] buffer, ref int position)
        {
            fixed (byte* pBuffer = buffer)
            {
                long* pDest = (long*)(pBuffer + position);
                *pDest = a;
            }
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

        public unsafe static void ToByteArray(this ulong a, byte[] buffer, ref int position)
        {
            fixed (byte* pBuffer = buffer)
            {
                ulong* pDest = (ulong*)(pBuffer + position);
                *pDest = a;
            }
            position += sizeof(ulong);
        }

        public static ulong ToULong(this byte[] buffer, ref int position)
        {
            ulong a = BitConverter.ToUInt64(buffer, position);
            position += sizeof(ulong);
            return a;
        }

        // UInt128

        public static MD5Hash ToUInt128(this byte[] buffer, ref int position)
        {
            MD5Hash a = new MD5Hash
            {
                Prefix = buffer.ToULong(ref position),
                Suffix = buffer.ToULong(ref position)
            };
            return a;
        }

        // float

        public static int SizeInBytes(this float a)
        {
            return sizeof(float);
        }

        public unsafe static void ToByteArray(this float a, byte[] buffer, ref int position)
        {
            fixed (byte* pBuffer = buffer)
            {
                float* pDest = (float*)(pBuffer + position);
                *pDest = a;
            }
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

        public unsafe static void ToByteArray(this double a, byte[] buffer, ref int position)
        {
            fixed (byte* pBuffer = buffer)
            {
                double* pDest = (double*)(pBuffer + position);
                *pDest = a;
            }
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
            return sizeof(int) + Encoding.Unicode.GetByteCount(a);
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
            return sizeof(int) + Utils.Size(a) * sizeof(byte);
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
            return sizeof(int) + Utils.Size(a) * sizeof(short);
        }

        public unsafe static void ToByteArray(this short[] a, byte[] buffer, ref int position)
        {
            int length = a.Length;
            length.ToByteArray(buffer, ref position);

            fixed (byte* tmpBuffer = buffer)
            fixed (short* pA = a)
            {
                short* pBuffer = (short*)(tmpBuffer + position);
                for (int i = 0; i < length; ++i)
                    pBuffer[i] = pA[i];
            }
            position += length * sizeof(short);
        }

        public unsafe static short[] ToShortArray(this byte[] buffer, ref int position)
        {
            int length = buffer.ToInt(ref position);
            short[] a = new short[length];

            fixed (byte* tmpBuffer = buffer)
            fixed (short* pA = a)
            {
                short* pBuffer = (short*)(tmpBuffer + position);
                for (int i = 0; i < length; ++i)
                    pA[i] = pBuffer[i];
            }
            position += length * sizeof(short);

            return a;
        }

        // ushort[]

        public static int SizeInBytes(this ushort[] a)
        {
            return sizeof(int) + Utils.Size(a) * sizeof(ushort);
        }

        public unsafe static void ToByteArray(this ushort[] a, byte[] buffer, ref int position)
        {
            int length = a.Length;
            length.ToByteArray(buffer, ref position);

            fixed (byte* tmpBuffer = buffer)
            fixed (ushort* pA = a)
            {
                ushort* pBuffer = (ushort*)(tmpBuffer + position);
                for (int i = 0; i < length; ++i)
                    pBuffer[i] = pA[i];
            }
            position += length * sizeof(ushort);
        }

        public unsafe static ushort[] ToUShortArray(this byte[] buffer, ref int position)
        {
            int length = buffer.ToInt(ref position);
            ushort[] a = new ushort[length];

            fixed (byte* tmpBuffer = buffer)
            fixed (ushort* pA = a)
            {
                ushort* pBuffer = (ushort*)(tmpBuffer + position);
                for (int i = 0; i < length; ++i)
                    pA[i] = pBuffer[i];
            }
            position += length * sizeof(ushort);

            return a;
        }

        // int[]

        public static int SizeInBytes(this int[] array)
        {
            return sizeof(int) + Utils.Size(array) * sizeof(int);
        }

        public unsafe static void ToByteArray(this int[] a, byte[] buffer, ref int position)
        {
            int length = Utils.Size(a);
            length.ToByteArray(buffer, ref position);

            fixed (byte* tmpBuffer = buffer)
            fixed (int* pA = a)
            {
                int* pBuffer = (int*)(tmpBuffer + position);
                for (int i = 0; i < length; ++i)
                    pBuffer[i] = pA[i];
            }
            position += length * sizeof(int);
        }

        public unsafe static int[] ToIntArray(this byte[] buffer, ref int position)
            => buffer.ToIntArray(ref position, buffer.ToInt(ref position));

        public unsafe static int[] ToIntArray(this byte[] buffer, ref int position, int length)
        {
            if (length == 0)
                return null;

            int[] a = new int[length];

            fixed (byte* tmpBuffer = buffer)
            fixed (int* pA = a)
            {
                int* pBuffer = (int*)(tmpBuffer + position);
                for (int i = 0; i < length; ++i)
                    pA[i] = pBuffer[i];
            }
            position += length * sizeof(int);

            return a;
        }

        // uint[]

        public static int SizeInBytes(this uint[] array)
        {
            return sizeof(int) + Utils.Size(array) * sizeof(uint);
        }

        public unsafe static void ToByteArray(this uint[] a, byte[] buffer, ref int position)
        {
            int length = a.Length;
            length.ToByteArray(buffer, ref position);

            fixed (byte* tmpBuffer = buffer)
            fixed (uint* pA = a)
            {
                uint* pBuffer = (uint*)(tmpBuffer + position);
                for (int i = 0; i < length; ++i)
                    pBuffer[i] = pA[i];
            }
            position += length * sizeof(uint);
        }

        public unsafe static uint[] ToUIntArray(this byte[] buffer, ref int position)
        {
            int length = buffer.ToInt(ref position);
            uint[] a = new uint[length];

            fixed (byte* tmpBuffer = buffer)
            fixed (uint* pA = a)
            {
                uint* pBuffer = (uint*)(tmpBuffer + position);
                for (int i = 0; i < length; ++i)
                    pA[i] = pBuffer[i];
            }
            position += length * sizeof(uint);

            return a;
        }

        // long[]

        public static int SizeInBytes(this long[] array)
        {
            return sizeof(int) + Utils.Size(array) * sizeof(long);
        }

        public unsafe static void ToByteArray(this long[] a, byte[] buffer, ref int position)
        {
            int length = a.Length;
            length.ToByteArray(buffer, ref position);

            fixed (byte* tmpBuffer = buffer)
            fixed (long* pA = a)
            {
                long* pBuffer = (long*)(tmpBuffer + position);
                for (int i = 0; i < length; ++i)
                    pBuffer[i] = pA[i];
            }
            position += length * sizeof(long);
        }

        public unsafe static long[] ToLongArray(this byte[] buffer, ref int position)
        {
            int length = buffer.ToInt(ref position);
            long[] a = new long[length];

            fixed (byte* tmpBuffer = buffer)
            fixed (long* pA = a)
            {
                long* pBuffer = (long*)(tmpBuffer + position);
                for (int i = 0; i < length; ++i)
                    pA[i] = pBuffer[i];
            }
            position += length * sizeof(long);

            return a;
        }

        // ulong[]

        public static int SizeInBytes(this ulong[] array)
        {
            return sizeof(int) + Utils.Size(array) * sizeof(ulong);
        }

        public unsafe static void ToByteArray(this ulong[] a, byte[] buffer, ref int position)
        {
            int length = a.Length;
            length.ToByteArray(buffer, ref position);

            fixed (byte* tmpBuffer = buffer)
            fixed (ulong* pA = a)
            {
                ulong* pBuffer = (ulong*)(tmpBuffer + position);
                for (int i = 0; i < length; ++i)
                    pBuffer[i] = pA[i];
            }
            position += length * sizeof(ulong);
        }

        public unsafe static ulong[] ToULongArray(this byte[] buffer, ref int position)
        {
            int length = buffer.ToInt(ref position);
            ulong[] a = new ulong[length];

            fixed (byte* tmpBuffer = buffer)
            fixed (ulong* pA = a)
            {
                ulong* pBuffer = (ulong*)(tmpBuffer + position);
                for (int i = 0; i < length; ++i)
                    pA[i] = pBuffer[i];
            }
            position += length * sizeof(ulong);

            return a;
        }

        // UInt128[]

        public static int SizeInBytes(this MD5Hash[] array)
        {
            return sizeof(int) + Utils.Size(array) * MD5Hash.SizeInBytes();
        }

        public static void ToByteArray(this MD5Hash[] a, byte[] buffer, ref int position)
        {
            a.Length.ToByteArray(buffer, ref position);
            for (int i = 0; i < a.Length; ++i)
            {
                a[i].ToByteArray(buffer, ref position);
            }
        }

        public unsafe static MD5Hash[] ToUInt128Array(this byte[] buffer, ref int position)
        {
            int length = buffer.ToInt(ref position);
            MD5Hash[] a = new MD5Hash[length];
            for (int i = 0; i < length; ++i)
            {
                a[i] = buffer.ToUInt128(ref position);
            }
            return a;
        }

        // float[]

        public static int SizeInBytes(this float[] array)
        {
            return sizeof(int) + Utils.Size(array) * sizeof(float);
        }

        public unsafe static void ToByteArray(this float[] a, byte[] buffer, ref int position)
        {
            int length = a.Length;
            length.ToByteArray(buffer, ref position);

            fixed (byte* tmpBuffer = buffer)
            fixed (float* pA = a)
            {
                float* pBuffer = (float*)(tmpBuffer + position);
                for (int i = 0; i < length; ++i)
                    pBuffer[i] = pA[i];
            }
            position += length * sizeof(float);
        }

        public unsafe static float[] ToFloatArray(this byte[] buffer, ref int position)
        {
            int length = buffer.ToInt(ref position);
            float[] a = new float[length];

            fixed (byte* tmpBuffer = buffer)
            fixed (float* pA = a)
            {
                float* pBuffer = (float*)(tmpBuffer + position);
                for (int i = 0; i < length; ++i)
                    pA[i] = pBuffer[i];
            }
            position += length * sizeof(float);

            return a;
        }

        // double[]

        public static int SizeInBytes(this double[] array)
        {
            return sizeof(int) + Utils.Size(array) * sizeof(double);
        }

        public unsafe static void ToByteArray(this double[] a, byte[] buffer, ref int position)
        {
            int length = Utils.Size(a);
            length.ToByteArray(buffer, ref position);

            fixed (byte* tmpBuffer = buffer)
            fixed (double* pA = a)
            {
                double* pBuffer = (double*)(tmpBuffer + position);
                for (int i = 0; i < length; ++i)
                    pBuffer[i] = pA[i];
            }
            position += length * sizeof(double);
        }

        public unsafe static double[] ToDoubleArray(this byte[] buffer, ref int position)
        {
            int length = buffer.ToInt(ref position);
            double[] a = new double[length];

            fixed (byte* tmpBuffer = buffer)
            fixed (double* pA = a)
            {
                double* pBuffer = (double*)(tmpBuffer + position);
                for (int i = 0; i < length; ++i)
                    pA[i] = pBuffer[i];
            }
            position += length * sizeof(double);

            return a;
        }

        // double[][]

        public static int SizeInBytes(this double[][] array)
        {
            if (Utils.Size(array) == 0)
                return sizeof(int);
            return sizeof(int) + array.Sum(x => x.SizeInBytes());
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
            int length = buffer.ToInt(ref position);
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
            int length = buffer.ToInt(ref position);
            string[] a = new string[length];
            for (int i = 0; i < a.Length; ++i)
            {
                a[i] = buffer.ToString(ref position);
            }
            return a;
        }
    }
}
