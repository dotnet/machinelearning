// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.CompilerServices;

namespace Microsoft.ML.Runtime.Data
{
    using R4 = Single;
    using R8 = Double;
    using BL = DvBool;
    using TX = DvText;

    public delegate bool RefPredicate<T>(ref T value);

    /// <summary>
    /// Utilities for IDV standard types, including proper NA semantics.
    /// </summary>
    public static class TypeUtils
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsNA(this R4 src) { return R4.IsNaN(src); }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsNA(this R8 src) { return R8.IsNaN(src); }

        #region R4
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BL Eq(this R4 a, R4 b)
        {
            return a == b ? BL.True : a.IsNA() || b.IsNA() ? BL.NA : BL.False;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BL Ne(this R4 a, R4 b)
        {
            return a != b ? a.IsNA() || b.IsNA() ? BL.NA : BL.True : BL.False;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BL Lt(this R4 a, R4 b)
        {
            return a < b ? BL.True : a >= b ? BL.False : BL.NA;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BL Le(this R4 a, R4 b)
        {
            return a <= b ? BL.True : a > b ? BL.False : BL.NA;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BL Ge(this R4 a, R4 b)
        {
            return a >= b ? BL.True : a < b ? BL.False : BL.NA;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BL Gt(this R4 a, R4 b)
        {
            return a > b ? BL.True : a <= b ? BL.False : BL.NA;
        }
        #endregion R4

        #region R8
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BL Eq(this R8 a, R8 b)
        {
            return a == b ? BL.True : a.IsNA() || b.IsNA() ? BL.NA : BL.False;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BL Ne(this R8 a, R8 b)
        {
            return a != b ? a.IsNA() || b.IsNA() ? BL.NA : BL.True : BL.False;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BL Lt(this R8 a, R8 b)
        {
            return a < b ? BL.True : a >= b ? BL.False : BL.NA;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BL Le(this R8 a, R8 b)
        {
            return a <= b ? BL.True : a > b ? BL.False : BL.NA;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BL Ge(this R8 a, R8 b)
        {
            return a >= b ? BL.True : a < b ? BL.False : BL.NA;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BL Gt(this R8 a, R8 b)
        {
            return a > b ? BL.True : a <= b ? BL.False : BL.NA;
        }
        #endregion R8
    }
}
