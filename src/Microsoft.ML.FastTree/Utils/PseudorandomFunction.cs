// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;

namespace Microsoft.ML.Runtime.FastTree.Internal
{
    /// <summary>
    /// This class defines a psuedorandom function, mapping a number to
    /// a hard to predict but deterministic other number, through some
    /// nefarious means.
    /// </summary>
    public sealed class PseudorandomFunction
    {
        private readonly int[][] _data;
        private static readonly int[] _periodics = new int[] { 32, 27, 25, 49, 11, 13, 17, 23, 29, 31, 37, 41, 43, 47 };

        public PseudorandomFunction(Random rand)
        {
            _data = _periodics.Select(x => Enumerable.Range(0, x).Select(y => rand.Next(-1, int.MaxValue) + 1).ToArray()).ToArray();
        }

        public int Apply(ulong seed)
        {
            int val = 0;
            foreach (int[] data in _data)
                val ^= data[(int)(seed % (ulong)(data.Length))];
            return val;
        }
    }
}
