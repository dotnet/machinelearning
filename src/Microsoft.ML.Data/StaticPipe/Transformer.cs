// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Data.StaticPipe
{
    public sealed class Transformer<TTupleInShape, TTupleOutShape, TTransformer>
        where TTransformer : class, ITransformer
    {
        private readonly IHostEnvironment _env;
        public TTransformer Wrapped { get; }

        public Transformer(IHostEnvironment env, TTransformer transformer)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(transformer, nameof(transformer));

            _env = env;
            Wrapped = transformer;
        }

        public Transformer<TTupleInShape, TTupleNewOutShape, TransformerChain<TNewTransformer>>
            Append<TTupleNewOutShape, TNewTransformer>(Transformer<TTupleOutShape, TTupleNewOutShape, TNewTransformer> transformer)
            where TNewTransformer : class, ITransformer
        {
            Contracts.Assert(nameof(Append) == nameof(LearningPipelineExtensions.Append));

            var trans = Wrapped.Append(transformer.Wrapped);
            return new Transformer<TTupleInShape, TTupleNewOutShape, TransformerChain<TNewTransformer>>(_env, trans);
        }

        public DataView<TTupleOutShape> Transform(DataView<TTupleInShape> input)
        {
            Contracts.Assert(nameof(Transform) == nameof(ITransformer.Transform));
            _env.CheckValue(input, nameof(input));

            var view = Wrapped.Transform(input.Wrapped);
            return new DataView<TTupleOutShape>(_env, view);
        }
    }
}
