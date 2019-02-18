// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Data.DataView;
using Microsoft.ML.Data;

namespace Microsoft.ML
{
    public static class DataReaderExtensions
    {
        public static IDataView Read(this IDataReader<IMultiStreamSource> reader, string path)
        {
            return reader.Read(new MultiFileSource(path));
        }

        public static IDataView Read(this IDataReader<IMultiStreamSource> reader, params string[] path)
        {
            return reader.Read(new MultiFileSource(path));
        }
    }
}
