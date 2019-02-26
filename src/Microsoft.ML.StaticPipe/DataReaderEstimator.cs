// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.StaticPipe
{
    public sealed class DataReaderEstimator<TIn, TShape, TDataReader> : SchemaBearing<TShape>
        where TDataReader : class, IDataLoader<TIn>
    {
        public IDataLoaderEstimator<TIn, TDataReader> AsDynamic { get; }

        internal DataReaderEstimator(IHostEnvironment env, IDataLoaderEstimator<TIn, TDataReader> estimator, StaticSchemaShape shape)
            : base(env, shape)
        {
            Env.AssertValue(estimator);

            AsDynamic = estimator;
            Shape.Check(Env, AsDynamic.GetOutputSchema());
        }

        public DataReader<TIn, TShape> Fit(TIn input)
        {
            Contracts.Assert(nameof(Fit) == nameof(IDataLoaderEstimator<TIn, TDataReader>.Fit));

            var reader = AsDynamic.Fit(input);
            return new DataReader<TIn, TShape>(Env, reader, Shape);
        }

        public DataReaderEstimator<TIn, TNewOut, IDataLoader<TIn>> Append<TNewOut, TTrans>(Estimator<TShape, TNewOut, TTrans> est)
            where TTrans : class, ITransformer
        {
            Contracts.Assert(nameof(Append) == nameof(CompositeReaderEstimator<TIn, ITransformer>.Append));

            var readerEst = AsDynamic.Append(est.AsDynamic);
            return new DataReaderEstimator<TIn, TNewOut, IDataLoader<TIn>>(Env, readerEst, est.Shape);
        }
    }
}
