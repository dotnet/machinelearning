// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Describes column information such as name and the source columns indices that this
    /// column encapsulates.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property, AllowMultiple = false, Inherited = true)]
    public sealed class LoadColumnAttribute : Attribute
    {
        /// <summary>
        /// Initializes new instance of <see cref="LoadColumnAttribute"/>.
        /// </summary>
        /// <param name="columnIndex">The index of the column in the text file.</param>
        public LoadColumnAttribute(int columnIndex)
        {
            Sources = new List<TextLoader.Range>();
            Sources.Add(new TextLoader.Range(columnIndex));
        }

        /// <summary>
        /// Initializes new instance of <see cref="LoadColumnAttribute"/>.
        /// </summary>
        /// <param name="start">The starting column index, for the range.</param>
        /// <param name="end">The ending column index, for the range.</param>
        public LoadColumnAttribute(int start, int end)
        {
            Sources = new List<TextLoader.Range>();
            Sources.Add(new TextLoader.Range(start, end));
        }

        /// <summary>
        /// Initializes new instance of <see cref="LoadColumnAttribute"/>.
        /// </summary>
        /// <param name="columnIndexes">Distinct text file column indices to load as part of this column.</param>
        public LoadColumnAttribute(int[] columnIndexes)
        {
            Sources = new List<TextLoader.Range>();
            foreach (var col in columnIndexes)
                Sources.Add(new TextLoader.Range(col));
        }

        internal List<TextLoader.Range> Sources;
    }
}
