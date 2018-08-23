// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Data.StaticPipe
{
    public sealed class DataReader<TIn, TTupleShape>
        : IDataReader<TIn>
    {
        private readonly IDataReader<TIn> _inner;

        ISchema IDataReader<TIn>.GetOutputSchema() => _inner.GetOutputSchema();
        IDataView IDataReader<TIn>.Read(TIn input) => _inner.Read(input);

        internal DataReader(IDataReader<TIn> inner)
        {
            _inner = inner;
        }
    }
}
