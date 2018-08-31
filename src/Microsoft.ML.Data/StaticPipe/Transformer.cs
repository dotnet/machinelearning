// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Data.StaticPipe
{
    public sealed class Transformer<TTupleInShape, TTupleOutShape, TTransformer> : SchemaBearing<TTupleOutShape>
        where TTransformer : class, ITransformer
    {
        public TTransformer AsDynamic { get; }

        public Transformer(IHostEnvironment env, TTransformer transformer)
            : base(env)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(transformer, nameof(transformer));

            AsDynamic = transformer;
        }

        public Transformer<TTupleInShape, TTupleNewOutShape, TransformerChain<TNewTransformer>>
            Append<TTupleNewOutShape, TNewTransformer>(Transformer<TTupleOutShape, TTupleNewOutShape, TNewTransformer> transformer)
            where TNewTransformer : class, ITransformer
        {
            Contracts.Assert(nameof(Append) == nameof(LearningPipelineExtensions.Append));

            var trans = AsDynamic.Append(transformer.AsDynamic);
            return new Transformer<TTupleInShape, TTupleNewOutShape, TransformerChain<TNewTransformer>>(Env, trans);
        }

        public DataView<TTupleOutShape> Transform(DataView<TTupleInShape> input)
        {
            Env.Assert(nameof(Transform) == nameof(ITransformer.Transform));
            Env.CheckValue(input, nameof(input));

            var view = AsDynamic.Transform(input.AsDynamic);
            return new DataView<TTupleOutShape>(Env, view);
        }
    }
}
