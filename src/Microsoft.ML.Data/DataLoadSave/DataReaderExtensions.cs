// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Data.DataView;
using Microsoft.ML.Data;

namespace Microsoft.ML
{
    public static class DataReaderExtensions
    {
        /// <summary>
        /// Reads data from one or more file <paramref name="path"/> into an <see cref="IDataView"/>.
        /// </summary>
        /// <param name="reader">The reader to use.</param>
        /// <param name="path">One or more paths from which to load data.</param>
        public static IDataView Read(this IDataReader<IMultiStreamSource> reader, params string[] path)
            => reader.Read(new MultiFileSource(path));
    }
}
