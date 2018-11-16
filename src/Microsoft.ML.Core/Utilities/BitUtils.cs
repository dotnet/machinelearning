// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.CompilerServices;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    internal static partial class Utils
    {
        private const int CbitUint = 32;
        private const int CbitUlong = 64;

        // Various bit/byte manipulation methods.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong MakeUlong(uint uHi, uint uLo)
        {
            return ((ulong)uHi << CbitUint) | uLo;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint GetLo(ulong uu)
        {
            // REVIEW: Work around Dev10 Bug 884217: JIT64  -Silent bad codegen for accessing 4-byte parts of 8-byte locals
            // http://vstfdevdiv:8080/WorkItemTracking/WorkItem.aspx?artifactMoniker=884217
            // return (uint)uu;
            return (uint)(uu & 0xFFFFFFFF);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint GetHi(ulong uu)
        {
            return (uint)(uu >> CbitUint);
        }

        public static uint Abs(int a)
        {
            uint mask = (uint)(a >> 31);
            return ((uint)a ^ mask) - mask;
        }

        public static ulong Abs(long a)
        {
            ulong mask = (ulong)(a >> 63);
            return ((ulong)a ^ mask) - mask;
        }

        public static int CbitHighZero(uint u)
        {
            if (u == 0)
                return 32;

            int cbit = 0;
            if ((u & 0xFFFF0000) == 0)
            {
                cbit += 16;
                u <<= 16;
            }
            if ((u & 0xFF000000) == 0)
            {
                cbit += 8;
                u <<= 8;
            }
            if ((u & 0xF0000000) == 0)
            {
                cbit += 4;
                u <<= 4;
            }
            if ((u & 0xC0000000) == 0)
            {
                cbit += 2;
                u <<= 2;
            }
            if ((u & 0x80000000) == 0)
                cbit += 1;
            return cbit;
        }

        public static int IbitHigh(uint u)
        {
            if (u == 0)
                return -1;

            int ibit = 31;
            if ((u & 0xFFFF0000) == 0)
            {
                ibit -= 16;
                u <<= 16;
            }
            if ((u & 0xFF000000) == 0)
            {
                ibit -= 8;
                u <<= 8;
            }
            if ((u & 0xF0000000) == 0)
            {
                ibit -= 4;
                u <<= 4;
            }
            if ((u & 0xC0000000) == 0)
            {
                ibit -= 2;
                u <<= 2;
            }
            if ((u & 0x80000000) == 0)
                ibit -= 1;
            return ibit;
        }

        public static int CbitLowZero(uint u)
        {
            if (u == 0)
                return 32;

            int cbit = 0;
            if ((u & 0x0000FFFF) == 0)
            {
                cbit += 16;
                u >>= 16;
            }
            if ((u & 0x000000FF) == 0)
            {
                cbit += 8;
                u >>= 8;
            }
            if ((u & 0x0000000F) == 0)
            {
                cbit += 4;
                u >>= 4;
            }
            if ((u & 0x00000003) == 0)
            {
                cbit += 2;
                u >>= 2;
            }
            if ((u & 0x00000001) == 0)
                cbit += 1;
            return cbit;
        }

        public static int CbitHighZero(ulong uu)
        {
            if ((uu & 0xFFFFFFFF00000000) == 0)
                return 32 + CbitHighZero(GetLo(uu));
            return CbitHighZero(GetHi(uu));
        }

        public static int CbitLowZero(ulong uu)
        {
            uint u = GetLo(uu);
            if (u == 0)
                return 32 + CbitLowZero(GetHi(uu));
            return CbitLowZero(u);
        }

        public static int Cbit(uint u)
        {
            u = (u & 0x55555555) + ((u >> 1) & 0x55555555);
            u = (u & 0x33333333) + ((u >> 2) & 0x33333333);
            u = (u & 0x0F0F0F0F) + ((u >> 4) & 0x0F0F0F0F);
            u = (u & 0x00FF00FF) + ((u >> 8) & 0x00FF00FF);
            return (int)((ushort)u + (ushort)(u >> 16));
        }

        public static int Cbit(ulong uu)
        {
            uu = (uu & 0x5555555555555555) + ((uu >> 1) & 0x5555555555555555);
            uu = (uu & 0x3333333333333333) + ((uu >> 2) & 0x3333333333333333);
            uu = (uu & 0x0F0F0F0F0F0F0F0F) + ((uu >> 4) & 0x0F0F0F0F0F0F0F0F);
            uu = (uu & 0x00FF00FF00FF00FF) + ((uu >> 8) & 0x00FF00FF00FF00FF);
            uu = (uu & 0x0000FFFF0000FFFF) + ((uu >> 16) & 0x0000FFFF0000FFFF);
            return (int)(GetLo(uu) + GetHi(uu));
        }

        // Builds a mask with bits below ibit all set. Only works for 0 <= ibit < 32.
        // Use MaskBelowEx to extend to 0 <= ibit < 64, in particular, for 32.
        public static uint UMaskBelow(int ibit)
        {
            Contracts.Assert(0 <= ibit && ibit < CbitUint, "UMaskBelow is designed to work for 0 <= ibit < 32");
            return (uint)(1U << ibit) - 1;
        }

        // Builds a mask with bits below ibit all set. Only works for 0 <= ibit < 64.
        public static uint UMaskBelowEx(int ibit)
        {
            Contracts.Assert(0 <= ibit && ibit < CbitUlong, "UMaskBelowEx is designed to work for 0 <= ibit < 64");
            // To handle 32 <= ibit < 64 properly, we have to start with 1LU, since otherwise the C# compiler only shifts
            // by the low 5 bits of ibit.
            return (uint)(1LU << ibit) - 1;
        }

        // Builds a mask with bits below ibit all set. Only works for 0 <= ibit < 64.
        public static ulong UuMaskBelow(int ibit)
        {
            Contracts.Assert(0 <= ibit && ibit < CbitUlong, "UuMaskBelow is designed to work for 0 <= ibit < 64");
            return (1LU << ibit) - 1;
        }

        public static bool IsPowerOfTwo(int x)
        {
            return (x & (x - 1)) == 0;
        }

        public static bool IsPowerOfTwo(uint x)
        {
            return (x & (x - 1)) == 0;
        }

        public static bool IsPowerOfTwo(long x)
        {
            return (x & (x - 1)) == 0;
        }

        public static bool IsPowerOfTwo(ulong x)
        {
            return (x & (x - 1)) == 0;
        }
    }
}
