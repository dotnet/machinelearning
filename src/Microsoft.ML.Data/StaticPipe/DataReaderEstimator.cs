// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Data.StaticPipe.Runtime;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Data.StaticPipe
{
    public sealed class DataReaderEstimator<TIn, TTupleShape, TDataReader> : SchemaBearing<TTupleShape>
        where TDataReader : class, IDataReader<TIn>
    {
        public IDataReaderEstimator<TIn, TDataReader> AsDynamic { get; }

        internal DataReaderEstimator(IHostEnvironment env, IDataReaderEstimator<TIn, TDataReader> estimator, StaticSchemaShape shape)
            : base(env, shape)
        {
            Env.AssertValue(estimator);

            AsDynamic = estimator;
            Shape.Check(Env, AsDynamic.GetOutputSchema());
        }

        public DataReader<TIn, TTupleShape> Fit(TIn input)
        {
            Contracts.Assert(nameof(Fit) == nameof(IDataReaderEstimator<TIn, TDataReader>.Fit));

            var reader = AsDynamic.Fit(input);
            return new DataReader<TIn, TTupleShape>(Env, reader, Shape);
        }

        public DataReaderEstimator<TIn, TNewOut, IDataReader<TIn>> Append<TNewOut, TTrans>(Estimator<TTupleShape, TNewOut, TTrans> est)
            where TTrans : class, ITransformer
        {
            Contracts.Assert(nameof(Append) == nameof(CompositeReaderEstimator<TIn, ITransformer>.Append));

            var readerEst = AsDynamic.Append(est.AsDynamic);
            return new DataReaderEstimator<TIn, TNewOut, IDataReader<TIn>>(Env, readerEst, est.Shape);
        }
    }
}
