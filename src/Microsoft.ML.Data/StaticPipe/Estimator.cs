// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data.StaticPipe.Runtime;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Data.StaticPipe
{
    public sealed class Estimator<TTupleInShape, TTupleOutShape, TTransformer> : SchemaBearing<TTupleOutShape>
        where TTransformer : class, ITransformer
    {
        public IEstimator<TTransformer> AsDynamic { get; }
        private readonly StaticSchemaShape _inShape;

        internal Estimator(IHostEnvironment env, IEstimator<TTransformer> estimator, StaticSchemaShape inShape, StaticSchemaShape outShape)
            : base(env, outShape)
        {
            Env.CheckValue(estimator, nameof(estimator));
            AsDynamic = estimator;
            _inShape = inShape;
            // Our ability to check estimators at constructor time is somewaht limited. During fit though we could.
            // Fortunately, estimators are one of the least likely things that users will freqeuently declare the
            // types of on their own.
        }

        public Transformer<TTupleInShape, TTupleOutShape, TTransformer> Fit(DataView<TTupleInShape> view)
        {
            Contracts.Assert(nameof(Fit) == nameof(IEstimator<TTransformer>.Fit));
            _inShape.Check(Env, view.AsDynamic.Schema);

            var trans = AsDynamic.Fit(view.AsDynamic);
            return new Transformer<TTupleInShape, TTupleOutShape, TTransformer>(Env, trans, _inShape, Shape);
        }

        public Estimator<TTupleInShape, TTupleNewOutShape, ITransformer> Append<TTupleNewOutShape>(Estimator<TTupleOutShape, TTupleNewOutShape, ITransformer> estimator)
        {
            Env.CheckValue(estimator, nameof(estimator));

            var est = AsDynamic.Append(estimator.AsDynamic);
            return new Estimator<TTupleInShape, TTupleNewOutShape, ITransformer>(Env, est, _inShape, estimator.Shape);
        }

        public Estimator<TTupleInShape, TTupleNewOutShape, ITransformer> Append<[IsShape] TTupleNewOutShape>(Func<TTupleOutShape, TTupleNewOutShape> mapper)
        {
            Contracts.CheckValue(mapper, nameof(mapper));

            using (var ch = Env.Start(nameof(Append)))
            {
                var method = mapper.Method;

                // Construct the dummy column structure, then apply the mapping.
                var input = StaticPipeInternalUtils.MakeAnalysisInstance<TTupleOutShape>(out var fakeReconciler);
                KeyValuePair<string, PipelineColumn>[] inPairs = StaticPipeInternalUtils.GetNamesValues(input, method.GetParameters()[0]);

                // Initially we suppose we've only assigned names to the inputs.
                var inputColToName = new Dictionary<PipelineColumn, string>();
                foreach (var p in inPairs)
                    inputColToName[p.Value] = p.Key;
                string NameMap(PipelineColumn col)
                {
                    inputColToName.TryGetValue(col, out var val);
                    return val;
                }

                var readerEst = StaticPipeUtils.GeneralFunctionAnalyzer(Env, ch, input, fakeReconciler, mapper, out var estTail, NameMap);
                ch.Assert(readerEst == null);
                ch.AssertValue(estTail);

                var est = AsDynamic.Append(estTail);
                var newOut = StaticSchemaShape.Make<TTupleNewOutShape>(method.ReturnParameter);
                var toReturn = new Estimator<TTupleInShape, TTupleNewOutShape, ITransformer>(Env, est, _inShape, newOut);
                ch.Done();
                return toReturn;
            }
        }
    }
}
