// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Various methods for creating <see cref="VBufferEditor{T}"/> instances.
    /// </summary>
    public static class VBufferEditor
    {
        /// <summary>
        /// Creates a <see cref="VBufferEditor{T}"/> with the same shape
        /// (length and density) as the <paramref name="destination"/>.
        /// </summary>
        public static VBufferEditor<T> CreateFromBuffer<T>(
            ref VBuffer<T> destination)
        {
            return destination.GetEditor();
        }

        /// <summary>
        /// Creates a <see cref="VBufferEditor{T}"/> using
        /// <paramref name="destination"/>'s values and indices buffers.
        /// </summary>
        /// <param name="destination">
        /// The destination buffer.
        /// </param>
        /// <param name="newLogicalLength">
        /// The logical length of the new buffer being edited.
        /// </param>
        /// <param name="valuesCount">
        /// The optional number of physical values to be represented in the buffer.
        /// The buffer will be dense if <paramref name="valuesCount"/> is omitted.
        /// </param>
        /// <param name="keepOldOnResize">
        /// True means that the old buffer values and indices are preserved, if possible (Array.Resize is called).
        /// False means that a new array will be allocated, if necessary.
        /// </param>
        /// <param name="requireIndicesOnDense">
        /// True means to ensure the Indices buffer is available, even if the buffer will be dense.
        /// </param>
        public static VBufferEditor<T> Create<T>(
            ref VBuffer<T> destination,
            int newLogicalLength,
            int? valuesCount = null,
            bool keepOldOnResize = false,
            bool requireIndicesOnDense = false)
        {
            return destination.GetEditor(
                newLogicalLength,
                valuesCount,
                keepOldOnResize: keepOldOnResize,
                requireIndicesOnDense: requireIndicesOnDense);
        }

        internal static VBufferEditor<T> Create<T>(
            ref VBuffer<T> destination,
            int newLogicalLength,
            int valuesCount,
            int maxValuesCapacity)
        {
            return destination.GetEditor(
                newLogicalLength,
                valuesCount,
                maxValuesCapacity);
        }
    }

    /// <summary>
    /// An object capable of editing a <see cref="VBuffer{T}"/> by filling out
    /// <see cref="Values"/> (and <see cref="Indices"/> if the buffer is not dense).
    /// </summary>
    public readonly ref struct VBufferEditor<T>
    {
        private readonly int _logicalLength;
        private readonly T[] _values;
        private readonly int[] _indices;

        /// <summary>
        /// The mutable span of values.
        /// </summary>
        public readonly Span<T> Values;

        /// <summary>
        /// The mutable span of indices.
        /// </summary>
        public readonly Span<int> Indices;

        /// <summary>
        /// Gets a value indicating whether a new Values array was allocated.
        /// </summary>
        public bool CreatedNewValues { get; }

        /// <summary>
        /// Gets a value indicating whether a new Indices array was allocated.
        /// </summary>
        public bool CreatedNewIndices { get; }

        internal VBufferEditor(int logicalLength,
            int physicalValuesCount,
            T[] values,
            int[] indices,
            bool requireIndicesOnDense,
            bool createdNewValues,
            bool createdNewIndices)
        {
            _logicalLength = logicalLength;
            _values = values;
            _indices = indices;

            bool isDense = logicalLength == physicalValuesCount;

            Values = _values.AsSpan(0, physicalValuesCount);
            Indices = !isDense || requireIndicesOnDense ? _indices.AsSpan(0, physicalValuesCount) : default;

            CreatedNewValues = createdNewValues;
            CreatedNewIndices = createdNewIndices;
        }

        /// <summary>
        /// Commits the edits and creates a new <see cref="VBuffer{T}"/> using
        /// the current Values and Indices.
        /// </summary>
        /// <returns>
        /// The newly created <see cref="VBuffer{T}"/>.
        /// </returns>
        public VBuffer<T> Commit()
        {
            return new VBuffer<T>(_logicalLength, Values.Length, _values, _indices);
        }

        /// <summary>
        /// Commits the edits and creates a new <see cref="VBuffer{T}"/> using
        /// the current Values and Indices, while allowing to truncate the length
        /// of Values and Indices.
        /// </summary>
        /// <param name="physicalValuesCount">
        /// The new number of physical values to be represented in the created buffer.
        /// </param>
        /// <returns>
        /// The newly created <see cref="VBuffer{T}"/>.
        /// </returns>
        /// <remarks>
        /// CommitTruncated allows to modify the length of the explicitly
        /// defined values.
        /// This is useful in sparse situations where the <see cref="VBufferEditor{T}"/>
        /// was created with a larger physical value count than was needed
        /// because the final value count was not known at creation time.
        /// </remarks>
        public VBuffer<T> CommitTruncated(int physicalValuesCount)
        {
            Contracts.CheckParam(physicalValuesCount <= Values.Length, nameof(physicalValuesCount), "Updating physicalValuesCount during CommitTruncated cannot be greater than the original physicalValuesCount value used in Create.");

            return new VBuffer<T>(_logicalLength, physicalValuesCount, _values, _indices);
        }
    }
}
