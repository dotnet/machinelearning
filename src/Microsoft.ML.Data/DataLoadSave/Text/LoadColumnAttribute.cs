// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Allow member to specify mapping to field(s) in text file.
    /// To override name of <see cref="IDataView"/> column use <see cref="ColumnNameAttribute"/>.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property, AllowMultiple = false, Inherited = true)]
    public sealed class LoadColumnAttribute : Attribute
    {
        /// <summary>
        /// Maps member to specific field in text file.
        /// </summary>
        /// <param name="fieldIndex">The index of the field in the text file.</param>
        public LoadColumnAttribute(int fieldIndex)
        {
            Sources = new List<TextLoader.Range>();
            Sources.Add(new TextLoader.Range(fieldIndex));
        }

        /// <summary>
        /// Maps member to range of fields in text file.
        /// </summary>
        /// <param name="start">The starting field index, for the range.</param>
        /// <param name="end">The ending field index, for the range.</param>
        public LoadColumnAttribute(int start, int end)
        {
            Sources = new List<TextLoader.Range>();
            Sources.Add(new TextLoader.Range(start, end));
        }

        /// <summary>
        /// Maps member to set of fields in text file.
        /// </summary>
        /// <param name="columnIndexes">Distinct text file field indices to load as part of this column.</param>
        public LoadColumnAttribute(int[] columnIndexes)
        {
            Sources = new List<TextLoader.Range>();
            foreach (var col in columnIndexes)
                Sources.Add(new TextLoader.Range(col));
        }

        internal List<TextLoader.Range> Sources;
    }
}
