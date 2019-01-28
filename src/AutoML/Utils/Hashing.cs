// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Auto
{
    internal static class Hashing
    {
        public static uint CombineHash(uint u1, uint u2)
        {
            return ((u1 << 7) | (u1 >> 25)) ^ u2;
        }

        public static int CombineHash(int n1, int n2)
        {
            return (int)CombineHash((uint)n1, (uint)n2);
        }

        /// <summary>
        /// Creates a combined hash of possibly heterogenously typed values.
        /// </summary>
        /// <param name="startHash">The leading hash, incorporated into the final hash</param>
        /// <param name="os">A variable list of objects, where null is a valid value</param>
        /// <returns>The combined hash incorpoating a starting hash, and the hash codes
        /// of all input values</returns>
        public static int CombinedHash(int startHash, params object[] os)
        {
            foreach (object o in os)
                startHash = CombineHash(startHash, o == null ? 0 : o.GetHashCode());
            return startHash;
        }
    }
}
