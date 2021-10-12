// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security;
using Microsoft.ML.Internal.CpuMath.Core;

namespace Microsoft.ML.Internal.CpuMath
{
    [BestFriend]
    internal static class IntUtils
    {
        /// <summary>
        /// Add src to the 128 bits contained in dst. Ignores overflow, that is, the addition is done modulo 2^128.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Add(ref ulong dstHi, ref ulong dstLo, ulong src)
        {
            if ((dstLo += src) < src)
                dstHi++;
        }

        /// <summary>
        /// Add src to dst. Ignores overflow, that is, the addition is done modulo 2^128.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Add(ref ulong dstHi, ref ulong dstLo, ulong srcHi, ulong srcLo)
        {
            if ((dstLo += srcLo) < srcLo)
                dstHi++;
            dstHi += srcHi;
        }

        /// <summary>
        /// Subtract src from the 128 bits contained in dst. Ignores overflow, that is, the subtraction is
        /// done modulo 2^128.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Sub(ref ulong dstHi, ref ulong dstLo, ulong src)
        {
            if (dstLo < src)
                dstHi--;
            dstLo -= src;
        }

        /// <summary>
        /// Subtract src from dst. Ignores overflow, that is, the subtraction is done modulo 2^128.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Sub(ref ulong dstHi, ref ulong dstLo, ulong srcHi, ulong srcLo)
        {
            dstHi -= srcHi;
            if (dstLo < srcLo)
                dstHi--;
            dstLo -= srcLo;
        }

        /// <summary>
        /// Return true if a is less than b.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool LessThan(ulong a1, ulong a0, ulong b1, ulong b0)
        {
            return a1 < b1 || a1 == b1 && a0 < b0;
        }

        /// <summary>
        /// Multiple the two 64-bit values to produce 128 bit result.
        /// </summary>
        [DllImport(Thunk.NativePath), SuppressUnmanagedCodeSecurity]
        private static extern ulong Mul64(ulong a, ulong b, out ulong hi);

        /// <summary>
        /// Return the number of zero bits on the high end.
        /// </summary>
        private static int CbitHighZero(ulong u)
        {
            if (u == 0)
                return 64;

            int cbit = 0;
            if ((u & 0xFFFFFFFF00000000) == 0)
            {
                cbit += 32;
                u <<= 32;
            }
            if ((u & 0xFFFF000000000000) == 0)
            {
                cbit += 16;
                u <<= 16;
            }
            if ((u & 0xFF00000000000000) == 0)
            {
                cbit += 8;
                u <<= 8;
            }
            if ((u & 0xF000000000000000) == 0)
            {
                cbit += 4;
                u <<= 4;
            }
            if ((u & 0xC000000000000000) == 0)
            {
                cbit += 2;
                u <<= 2;
            }
            if ((u & 0x8000000000000000) == 0)
                cbit += 1;
            return cbit;
        }
    }
}
