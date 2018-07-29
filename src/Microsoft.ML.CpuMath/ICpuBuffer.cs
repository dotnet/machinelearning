// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Float = System.Single;

namespace Microsoft.ML.Runtime.Internal.CpuMath
{
    using Conditional = System.Diagnostics.ConditionalAttribute;

    public interface ICpuBuffer<T> : IEnumerable<T>, IDisposable
        where T : struct
    {
        int ValueCount { get; }

        /// <summary>
        /// Assign random values using the given random function.
        /// </summary>
        void Randomize(Func<T> rand);

        /// <summary>
        /// Set all values to zero.
        /// </summary>
        void Zero();

        /// <summary>
        /// Copy the values into dst, starting at slot ivDst and advancing ivDst.
        /// </summary>
        void CopyTo(T[] dst, ref int ivDst);

        /// <summary>
        /// Copy values from the given src array into this buffer, starting at the given index in src,
        /// </summary>
        void CopyFrom(T[] src, ref int ivSrc);
    }

    /// <summary>
    /// A logical math vector.
    /// </summary>
    public interface ICpuVector : ICpuBuffer<Float>
    {
        /// <summary>
        /// The vector size
        /// </summary>
        int VectorSize { get; }

        /// <summary>
        /// Get the i'th component of the vector.
        /// </summary>
        Float GetValue(int i);
    }

    public interface ICpuMatrix : ICpuBuffer<Float>
    {
        /// <summary>
        /// The row count
        /// </summary>
        int RowCount { get; }

        /// <summary>
        /// the column count
        /// </summary>
        int ColCount { get; }
    }

    /// <summary>
    /// A 2-dimensional matrix.
    /// </summary>
    public interface ICpuFullMatrix : ICpuMatrix
    {
        /// <summary>
        /// Copy the values for the given row into dst, starting at slot ivDst.
        /// </summary>
        void CopyTo(int row, Float[] dst, ref int ivDst);

        /// <summary>
        /// Zero out the items with the given indices.
        /// The indices contain the logical indices to the vectorized representation of the matrix,
        /// which can be different depending on whether the matrix is row-major or column-major.
        /// </summary>
        void ZeroItems(int[] indices);
    }
}