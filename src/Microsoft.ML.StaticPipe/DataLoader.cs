// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.StaticPipe
{
    public sealed class DataLoader<TIn, TShape> : SchemaBearing<TShape>
    {
        public IDataLoader<TIn> AsDynamic { get; }

        internal DataLoader(IHostEnvironment env, IDataLoader<TIn> loader, StaticSchemaShape shape)
            : base(env, shape)
        {
            Env.AssertValue(loader);

            AsDynamic = loader;
            Shape.Check(Env, AsDynamic.GetOutputSchema());
        }

        public DataLoaderEstimator<TIn, TNewOut, IDataLoader<TIn>> Append<TNewOut, TTrans>(Estimator<TShape, TNewOut, TTrans> estimator)
            where TTrans : class, ITransformer
        {
            Contracts.Assert(nameof(Append) == nameof(CompositeLoaderEstimator<TIn, ITransformer>.Append));

            var loaderEst = AsDynamic.Append(estimator.AsDynamic);
            return new DataLoaderEstimator<TIn, TNewOut, IDataLoader<TIn>>(Env, loaderEst, estimator.Shape);
        }

        public DataLoader<TIn, TNewShape> Append<TNewShape, TTransformer>(Transformer<TShape, TNewShape, TTransformer> transformer)
            where TTransformer : class, ITransformer
        {
            Env.CheckValue(transformer, nameof(transformer));
            Env.Assert(nameof(Append) == nameof(CompositeLoaderEstimator<TIn, ITransformer>.Append));

            var loader = AsDynamic.Append(transformer.AsDynamic);
            return new DataLoader<TIn, TNewShape>(Env, loader, transformer.Shape);
        }

        public DataView<TShape> Load(TIn input)
        {
            // We cannot check the value of input since it may not be a reference type, and it is not clear
            // that there is an absolute case for insisting that the input type be a reference type, and much
            // less further that null inputs will never be correct. So we rely on the wrapping object to make
            // that determination.
            Env.Assert(nameof(Load) == nameof(IDataLoader<TIn>.Load));

            var data = AsDynamic.Load(input);
            return new DataView<TShape>(Env, data, Shape);
        }
    }
}
