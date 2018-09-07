// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.CompilerServices;

namespace Microsoft.ML.Runtime.Data
{
    using R4 = Single;
    using R8 = Double;

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
    }
}
