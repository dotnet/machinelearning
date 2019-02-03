// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.StaticPipe
{
    public static class LocalPathReader
    {
        public static DataView<TShape> Read<TShape>(this DataReader<IMultiStreamSource, TShape> reader, string path)
        {
            return reader.Read(new MultiFileSource(path));
        }

        public static DataView<TShape> Read<TShape>(this DataReader<IMultiStreamSource, TShape> reader, params string[] path)
        {
            return reader.Read(new MultiFileSource(path));
        }
    }
}