// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.AutoML
{
    internal static class MLNetUtils
    {
        public static int[] GetRandomPermutation(Random rand, int size)
        {
            var res = GetIdentityPermutation(size);
            Shuffle<int>(rand, res);
            return res;
        }

        public static int[] GetIdentityPermutation(int size)
        {
            var res = new int[size];
            for (int i = 0; i < size; i++)
                res[i] = i;
            return res;
        }

        public static void Shuffle<T>(Random rand, Span<T> rgv)
        {
            for (int iv = 0; iv < rgv.Length; iv++)
                Swap(ref rgv[iv], ref rgv[iv + rand.Next(rgv.Length - iv)]);
        }

        public static void Swap<T>(ref T a, ref T b)
        {
            T temp = a;
            a = b;
            b = temp;
        }

        public static int Size<T>(T[] x)
        {
            return x == null ? 0 : x.Length;
        }
    }
}
