// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Data.StaticPipe.Runtime;

namespace Microsoft.ML.Data.StaticPipe
{
    public sealed class Transformer<TTupleInShape, TTupleOutShape, TTransformer> : SchemaBearing<TTupleOutShape>
        where TTransformer : class, ITransformer
    {
        public TTransformer AsDynamic { get; }
        private readonly StaticSchemaShape _inShape;

        internal Transformer(IHostEnvironment env, TTransformer transformer, StaticSchemaShape inShape, StaticSchemaShape outShape)
            : base(env, outShape)
        {
            Env.AssertValue(transformer);
            Env.AssertValue(inShape);
            AsDynamic = transformer;
            _inShape = inShape;
            // The ability to check at runtime is limited. We could check during transformation time on the input data view.
        }

        public Transformer<TTupleInShape, TTupleNewOutShape, TransformerChain<TNewTransformer>>
            Append<TTupleNewOutShape, TNewTransformer>(Transformer<TTupleOutShape, TTupleNewOutShape, TNewTransformer> transformer)
            where TNewTransformer : class, ITransformer
        {
            Env.Assert(nameof(Append) == nameof(LearningPipelineExtensions.Append));

            var trans = AsDynamic.Append(transformer.AsDynamic);
            return new Transformer<TTupleInShape, TTupleNewOutShape, TransformerChain<TNewTransformer>>(Env, trans, _inShape, transformer.Shape);
        }

        public DataView<TTupleOutShape> Transform(DataView<TTupleInShape> input)
        {
            Env.Assert(nameof(Transform) == nameof(ITransformer.Transform));
            Env.CheckValue(input, nameof(input));

            var view = AsDynamic.Transform(input.AsDynamic);
            return new DataView<TTupleOutShape>(Env, view, Shape);
        }
    }
}
