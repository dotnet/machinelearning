// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.Data.Analysis
{
    internal interface IArithmetic<T>
        where T : unmanaged
    {
        //Binary operations
        void HandleOperation(BinaryOperation operation, ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination);
        void HandleOperation(BinaryOperation operation, ReadOnlySpan<T> x, T y, Span<T> destination);
        void HandleOperation(BinaryOperation operation, T x, ReadOnlySpan<T> y, Span<T> destination);

        T HandleOperation(BinaryOperation operation, T x, T y);

        //Binary Int operations
        void HandleOperation(BinaryIntOperation operation, ReadOnlySpan<T> x, int y, Span<T> destination);

        //Comparison operations
        void HandleOperation(ComparisonOperation operation, ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination);
        void HandleOperation(ComparisonOperation operation, ReadOnlySpan<T> x, T y, Span<bool> destination);
    }
}
