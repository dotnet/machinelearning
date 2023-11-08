// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.CompilerServices;
using System.Numerics.Tensors;
using Microsoft.ML.Internal.CpuMath.Core;

namespace Microsoft.ML.Internal.CpuMath
{
    [BestFriend]
    internal static partial class CpuMathUtils
    {
        /// <summary>
        /// Adds a value to a destination.
        /// </summary>
        /// <param name="value">The value to add.</param>
        /// <param name="destination">The destination to add the value to.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Add(float value, Span<float> destination)
        {
            Contracts.AssertNonEmpty(destination);
            TensorPrimitives.Add(destination, value, destination);
        }

        /// <summary>
        /// Scales a value to a destination.
        /// </summary>
        /// <param name="value">The value to add.</param>
        /// <param name="destination">The destination to add the value to.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Scale(float value, Span<float> destination)
        {
            Contracts.AssertNonEmpty(destination);
            TensorPrimitives.Multiply(destination, value, destination);
        }

        /// <summary>
        /// Scales a values by a source to a destination.
        /// destination = value * source
        /// </summary>
        /// <param name="value">The value to scale by.</param>
        /// <param name="source">The source values.</param>
        /// <param name="destination">The destination.</param>
        /// <param name="count">The count of items.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Scale(float value, ReadOnlySpan<float> source, Span<float> destination, int count)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.AssertNonEmpty(destination);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= source.Length);
            Contracts.Assert(count <= destination.Length);

            TensorPrimitives.Multiply(source.Slice(0, count), value, destination);
        }

        /// <summary>
        /// Add to the destination from the source by scale.
        /// </summary>
        /// <param name="scale">The scale to add by.</param>
        /// <param name="source">The source values.</param>
        /// <param name="destination">The destination values.</param>
        /// <param name="count">The count of items.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void AddScale(float scale, ReadOnlySpan<float> source, Span<float> destination, int count)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.AssertNonEmpty(destination);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= source.Length);
            Contracts.Assert(count <= destination.Length);

            TensorPrimitives.MultiplyAdd(source.Slice(0, count), scale, destination.Slice(0, count), destination);
        }

        /// <summary>
        /// Add to the destination by scale and source into a new result.
        /// </summary>
        /// <param name="scale">The scale to add by.</param>
        /// <param name="source">The source values.</param>
        /// <param name="destination">The destination values.</param>
        /// <param name="result">A new collection of values to be returned.</param>
        /// <param name="count">The count of items.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void AddScaleCopy(float scale, ReadOnlySpan<float> source, ReadOnlySpan<float> destination, Span<float> result, int count)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.AssertNonEmpty(destination);
            Contracts.AssertNonEmpty(result);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= source.Length);
            Contracts.Assert(count <= destination.Length);
            Contracts.Assert(count <= result.Length);

            TensorPrimitives.MultiplyAdd(source.Slice(0, count), scale, destination.Slice(0, count), result.Slice(0, count));
        }

        /// <summary>
        /// Add from a source to a destination.
        /// </summary>
        /// <param name="source">The source values.</param>
        /// <param name="destination">The destination values.</param>
        /// <param name="count">The count of items.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Add(ReadOnlySpan<float> source, Span<float> destination, int count)
        {
            Contracts.AssertNonEmpty(source);
            Contracts.AssertNonEmpty(destination);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= source.Length);
            Contracts.Assert(count <= destination.Length);

            TensorPrimitives.Add(source.Slice(0, count), destination.Slice(0, count), destination.Slice(0, count));
        }

        /// <summary>
        /// Multiply each element with left and right elements.
        /// </summary>
        /// <param name="left">The left element.</param>
        /// <param name="right">The right element.</param>
        /// <param name="destination">The destination values.</param>
        /// <param name="count">The count of items.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void MulElementWise(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> destination, int count)
        {
            Contracts.AssertNonEmpty(left);
            Contracts.AssertNonEmpty(right);
            Contracts.AssertNonEmpty(destination);
            Contracts.Assert(count > 0);
            Contracts.Assert(count <= left.Length);
            Contracts.Assert(count <= right.Length);
            Contracts.Assert(count <= destination.Length);

            TensorPrimitives.Multiply(left.Slice(0, count), right.Slice(0, count), destination.Slice(0, count));
        }

        /// <summary>
        /// Sum the values in the source.
        /// </summary>
        /// <param name="source">The source values.</param>
        /// <returns>The sum of all items in <paramref name="source"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Sum(ReadOnlySpan<float> source)
        {
            Contracts.AssertNonEmpty(source);

            return TensorPrimitives.Sum(source);
        }

        /// <summary>
        /// Sum the squares of each item in the source.
        /// </summary>
        /// <param name="source">The source values.</param>
        /// <returns>The sum of the squares of all items in <paramref name="source"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SumSq(ReadOnlySpan<float> source)
        {
            Contracts.AssertNonEmpty(source);

            return TensorPrimitives.SumOfSquares(source);
        }

        /// <summary>
        /// Sum the absolute value of each item in the source.
        /// </summary>
        /// <param name="source">The source values.</param>
        /// <returns>The sum of all absolute value of the items in <paramref name="source"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SumAbs(ReadOnlySpan<float> source)
        {
            Contracts.AssertNonEmpty(source);

            return TensorPrimitives.SumOfMagnitudes(source);
        }

        /// <summary>
        /// Returns the dot product of each item in the left and right spans.
        /// </summary>
        /// <param name="left">The left span.</param>
        /// <param name="right">The right span.</param>
        /// <param name="count">The count of items.</param>
        /// <returns>The dot product.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float DotProductDense(ReadOnlySpan<float> left, ReadOnlySpan<float> right, int count)
        {
            Contracts.AssertNonEmpty(left);
            Contracts.AssertNonEmpty(right);
            Contracts.Assert(count > 0);
            Contracts.Assert(left.Length >= count);
            Contracts.Assert(right.Length >= count);

            return TensorPrimitives.Dot(left.Slice(0, count), right.Slice(0, count));
        }
    }
}
