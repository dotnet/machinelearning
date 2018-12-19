// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;

namespace Microsoft.ML.Data
{
#pragma warning disable 618
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
             : this(columnIndex.ToString())
        {
            Sources.Add(new TextLoader.Range(columnIndex));
        }

        /// <summary>
        /// Initializes new instance of <see cref="LoadColumnAttribute"/>.
        /// </summary>
        /// <param name="start">The starting column index, for the range.</param>
        /// <param name="end">The ending column index, for the range.</param>
        public LoadColumnAttribute(int start, int end)
             : this(start) //REVIEW  this is incorrect, but it is just temporary there, until the Legacy API's TextLoader gets deleted.
        {
            Sources.Add(new TextLoader.Range(start, end));
        }

        /// <summary>
        /// Initializes new instance of <see cref="LoadColumnAttribute"/>.
        /// </summary>
        /// <param name="columnIndexes">Distinct text file column indices to load as part of this column.</param>
        public LoadColumnAttribute(int[] columnIndexes)
            : this(columnIndexes[0]) // REVIEW: this is incorrect, but it is just temporary there, until the Legacy API's TextLoader gets deleted.
        {
            foreach (var col in columnIndexes)
                Sources.Add(new TextLoader.Range(col));
        }

        [Obsolete("Should be deleted together with the Legacy project.")]
        private LoadColumnAttribute(string start)
        {
            Sources = new List<TextLoader.Range>();
            Start = start;
        }

        internal List<TextLoader.Range> Sources;

        [Obsolete("Should be deleted together with the Legacy project.")]
        [BestFriend]
        internal string Start { get; }
    }
#pragma warning restore 618
}
