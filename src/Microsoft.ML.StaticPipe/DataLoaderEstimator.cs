// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.StaticPipe
{
    public sealed class DataLoaderEstimator<TIn, TShape, TDataLoader> : SchemaBearing<TShape>
        where TDataLoader : class, IDataLoader<TIn>
    {
        public IDataLoaderEstimator<TIn, TDataLoader> AsDynamic { get; }

        internal DataLoaderEstimator(IHostEnvironment env, IDataLoaderEstimator<TIn, TDataLoader> estimator, StaticSchemaShape shape)
            : base(env, shape)
        {
            Env.AssertValue(estimator);

            AsDynamic = estimator;
            Shape.Check(Env, AsDynamic.GetOutputSchema());
        }

        public DataLoader<TIn, TShape> Fit(TIn input)
        {
            Contracts.Assert(nameof(Fit) == nameof(IDataLoaderEstimator<TIn, TDataLoader>.Fit));

            var loader = AsDynamic.Fit(input);
            return new DataLoader<TIn, TShape>(Env, loader, Shape);
        }

        public DataLoaderEstimator<TIn, TNewOut, IDataLoader<TIn>> Append<TNewOut, TTrans>(Estimator<TShape, TNewOut, TTrans> est)
            where TTrans : class, ITransformer
        {
            Contracts.Assert(nameof(Append) == nameof(CompositeLoaderEstimator<TIn, ITransformer>.Append));

            var loaderEst = AsDynamic.Append(est.AsDynamic);
            return new DataLoaderEstimator<TIn, TNewOut, IDataLoader<TIn>>(Env, loaderEst, est.Shape);
        }
    }
}
