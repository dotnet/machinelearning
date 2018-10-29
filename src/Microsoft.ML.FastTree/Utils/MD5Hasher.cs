// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Internal.Utilities;
using System;
using System.IO;
using System.Security.Cryptography;

namespace Microsoft.ML.Trainers.FastTree.Internal
{
    public struct MD5Hash
    {
        public UInt64 Prefix;
        public UInt64 Suffix;

        internal MD5Hash(byte[] array)
        {
            Contracts.Assert(Utils.Size(array) == SizeInBytes());
            Prefix = BitConverter.ToUInt64(array, 0);
            Suffix = BitConverter.ToUInt64(array, 8);
        }

        public static MD5Hash operator ^(MD5Hash first, MD5Hash second)
        {
            MD5Hash result = new MD5Hash
            {
                Prefix = first.Prefix ^ second.Prefix,
                Suffix = first.Suffix ^ second.Suffix
            };
            return result;
        }

        public static int SizeInBytes() { return 16; }

        public void ToByteArray(byte[] buffer, ref int position)
        {
            Prefix.ToByteArray(buffer, ref position);
            Suffix.ToByteArray(buffer, ref position);
        }
    }

    public static class MD5Hasher
    {
        public static MD5Hash Hash(byte[] array)
        {
            // REVIEW: Consider using murmur hash for this. Or at least, make
            // this more memory efficient.
            var hasher = new MD5CryptoServiceProvider();
            return new MD5Hash(hasher.ComputeHash(array));
        }

        private static MD5Hash Hash(Stream stream)
        {
            var hasher = new MD5CryptoServiceProvider();
            return new MD5Hash(hasher.ComputeHash(stream));
        }

        private static unsafe MD5Hash Hash(byte* ptr, int length)
        {
            var stream = new UnmanagedMemoryStream(ptr, length);
            return Hash(stream);
        }

        public static MD5Hash Hash(string str)
        {
            MemoryStream stream = new MemoryStream();
            StreamWriter writer = new StreamWriter(stream);
            writer.Write(str);
            writer.Flush();
            stream.Seek(0, SeekOrigin.Begin);
            return Hash(stream);
        }

        public static MD5Hash Hash(int a)
        {
            unsafe
            {
                return Hash((byte*)&a, sizeof(int));
            }
        }

        public static MD5Hash Hash(short[] array)
        {
            unsafe
            {
                fixed (short* pArray = array)
                {
                    byte* bArray = (byte*)pArray;
                    int byteLength = array.Length * sizeof(short);
                    return Hash(bArray, byteLength);
                }
            }
        }

        public static MD5Hash Hash(ushort[] array)
        {
            unsafe
            {
                fixed (ushort* pArray = array)
                {
                    byte* bArray = (byte*)pArray;
                    int byteLength = array.Length * sizeof(ushort);
                    return Hash(bArray, byteLength);
                }
            }
        }

        public static MD5Hash Hash(int[] array)
        {
            unsafe
            {
                fixed (int* pArray = array)
                {
                    byte* bArray = (byte*)pArray;
                    int byteLength = array.Length * sizeof(int);
                    return Hash(bArray, byteLength);
                }
            }
        }

        public static MD5Hash Hash(uint[] array)
        {
            unsafe
            {
                fixed (uint* pArray = array)
                {
                    byte* bArray = (byte*)pArray;
                    int byteLength = array.Length * sizeof(uint);
                    return Hash(bArray, byteLength);
                }
            }
        }

        public static MD5Hash Hash(double[] array)
        {
            unsafe
            {
                fixed (double* pArray = array)
                {
                    byte* bArray = (byte*)pArray;
                    int byteLength = array.Length * sizeof(double);
                    return Hash(bArray, byteLength);
                }
            }
        }
    }
}
