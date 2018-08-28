// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Data.StaticPipe
{
    public sealed class Estimator<TTupleInShape, TTupleOutShape, TTransformer>
        : BlockMaker<TTupleOutShape>
        where TTransformer : class, ITransformer
    {
        public IEstimator<TTransformer> Wrapped { get; }

        public Estimator(IHostEnvironment env, IEstimator<TTransformer> estimator)
            : base(env)
        {
            Env.CheckValue(estimator, nameof(estimator));
            Wrapped = estimator;
        }

        public Transformer<TTupleInShape, TTupleOutShape, TTransformer> Fit(DataView<TTupleInShape> view)
        {
            Contracts.Assert(nameof(Fit) == nameof(IEstimator<TTransformer>.Fit));

            var trans = Wrapped.Fit(view.Wrapped);
            return new Transformer<TTupleInShape, TTupleOutShape, TTransformer>(Env, trans);
        }
    }
}
