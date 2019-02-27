﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.StaticPipe
{
    public sealed class DataReader<TIn, TShape> : SchemaBearing<TShape>
    {
        public IDataReader<TIn> AsDynamic { get; }

        internal DataReader(IHostEnvironment env, IDataReader<TIn> reader, StaticSchemaShape shape)
            : base(env, shape)
        {
            Env.AssertValue(reader);

            AsDynamic = reader;
            Shape.Check(Env, AsDynamic.GetOutputSchema());
        }

        public DataReaderEstimator<TIn, TNewOut, IDataReader<TIn>> Append<TNewOut, TTrans>(Estimator<TShape, TNewOut, TTrans> estimator)
            where TTrans : class, ITransformer
        {
            Contracts.Assert(nameof(Append) == nameof(CompositeReaderEstimator<TIn, ITransformer>.Append));

            var readerEst = AsDynamic.Append(estimator.AsDynamic);
            return new DataReaderEstimator<TIn, TNewOut, IDataReader<TIn>>(Env, readerEst, estimator.Shape);
        }

        public DataReader<TIn, TNewShape> Append<TNewShape, TTransformer>(Transformer<TShape, TNewShape, TTransformer> transformer)
            where TTransformer : class, ITransformer
        {
            Env.CheckValue(transformer, nameof(transformer));
            Env.Assert(nameof(Append) == nameof(CompositeReaderEstimator<TIn, ITransformer>.Append));

            var reader = AsDynamic.Append(transformer.AsDynamic);
            return new DataReader<TIn, TNewShape>(Env, reader, transformer.Shape);
        }

        public DataView<TShape> Read(TIn input)
        {
            // We cannot check the value of input since it may not be a reference type, and it is not clear
            // that there is an absolute case for insisting that the input type be a reference type, and much
            // less further that null inputs will never be correct. So we rely on the wrapping object to make
            // that determination.
            Env.Assert(nameof(Read) == nameof(IDataReader<TIn>.Read));

            var data = AsDynamic.Read(input);
            return new DataView<TShape>(Env, data, Shape);
        }
    }
}
