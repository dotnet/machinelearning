// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.Data.Analysis
{
    internal interface IPrimitiveDataFrameColumnArithmetic<T>
        where T : unmanaged
    {
        void HandleOperation(BinaryOperation operation, Span<T> left, Span<byte> leftValidity, ReadOnlySpan<T> right, ReadOnlySpan<byte> rightValidity);

        void HandleOperation(ComparisonOperation operation, ReadOnlySpan<T> left, ReadOnlySpan<T> right, PrimitiveColumnContainer<bool> container, long offset);
        void HandleOperation(ComparisonScalarOperation operation, ReadOnlySpan<T> left, T right, PrimitiveColumnContainer<bool> container, long offset);

        void HandleOperation(BinaryScalarOperation operation, Span<T> left, T right);
        void HandleOperation(BinaryScalarOperation operation, T left, Span<T> right, ReadOnlySpan<byte> rightValidity);

        void HandleOperation(BinaryIntOperation operation, Span<T> left, int right);
    }
}
