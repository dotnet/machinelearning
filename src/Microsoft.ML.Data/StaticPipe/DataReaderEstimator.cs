// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Data.StaticPipe
{
    public abstract class DataReaderEstimator<TIn, TTupleShape>
        : BlockMaker<TTupleShape>, IDataReaderEstimator<TIn, IDataReader<TIn>>
    {
        IDataReader<TIn> IDataReaderEstimator<TIn, IDataReader<TIn>>.Fit(TIn input)
            => FitCore(input);

        protected abstract IDataReader<TIn> FitCore(TIn input);

        SchemaShape IDataReaderEstimator<TIn, IDataReader<TIn>>.GetOutputSchema()
            => GetOutputSchemaCore();

        protected abstract SchemaShape GetOutputSchemaCore();

        public DataReader<TIn, TTupleShape> Fit(TIn input)
        {
            var reader = FitCore(input);
            return new DataReader<TIn, TTupleShape>(reader);
        }
    }

    public abstract class Estimator<TTupleInShape, TTupleOutShape, TTransformer>
        : BlockMaker<TTupleOutShape>, IEstimator<TTransformer>
        where TTransformer : ITransformer
    {
        TTransformer IEstimator<TTransformer>.Fit(IDataView input)
            => FitCore(input);

        protected abstract TTransformer FitCore(IDataView input);

        SchemaShape IEstimator<TTransformer>.GetOutputSchema(SchemaShape inputSchema)
            => GetOutputSchemaCore(inputSchema);

        protected abstract SchemaShape GetOutputSchemaCore(SchemaShape inputSchema);
    }
}
