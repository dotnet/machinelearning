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

        public Estimator(IHostEnvironment env, IEstimator<TTransformer> estimator)
            : base(env)
        {
            Env.CheckValue(estimator, nameof(estimator));
            AsDynamic = estimator;
        }

        public Transformer<TTupleInShape, TTupleOutShape, TTransformer> Fit(DataView<TTupleInShape> view)
        {
            Contracts.Assert(nameof(Fit) == nameof(IEstimator<TTransformer>.Fit));

            var trans = AsDynamic.Fit(view.AsDynamic);
            return new Transformer<TTupleInShape, TTupleOutShape, TTransformer>(Env, trans);
        }

        public Estimator<TTupleInShape, TTupleNewOutShape, ITransformer> Append<TTupleNewOutShape>(Estimator<TTupleOutShape, TTupleNewOutShape, ITransformer> estimator)
        {
            Env.CheckValue(estimator, nameof(estimator));

            var est = AsDynamic.Append(estimator.AsDynamic);
            return new Estimator<TTupleInShape, TTupleNewOutShape, ITransformer>(Env, est);
        }

        public Estimator<TTupleInShape, TTupleNewOutShape, ITransformer> Append<TTupleNewOutShape>(Func<TTupleOutShape, TTupleNewOutShape> mapper)
        {
            Contracts.CheckValue(mapper, nameof(mapper));

            using (var ch = Env.Start(nameof(Append)))
            {
                var method = mapper.Method;

                // Construct the dummy column structure, then apply the mapping.
                var input = PipelineColumnAnalyzer.MakeAnalysisInstance<TTupleOutShape>(out var fakeReconciler);
                KeyValuePair<string, PipelineColumn>[] inPairs = PipelineColumnAnalyzer.GetNames(input, method.GetParameters()[0]);

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
                var toReturn = new Estimator<TTupleInShape, TTupleNewOutShape, ITransformer>(Env, est);
                ch.Done();
                return toReturn;
            }
        }
    }

    public static class Estimator
    {
        /// <summary>
        /// Create an object that can be used as the start of a new pipeline, that assumes it uses
        /// something with the sahape of <typeparamref name="TTupleShape"/> as its input schema shape.
        /// The returned object is an empty estimator.
        /// </summary>
        /// <param name="fromSchema"></param>
        /// <returns></returns>
        public static Estimator<TTupleShape, TTupleShape, ITransformer> MakeNew<TTupleShape>(SchemaBearing<TTupleShape> fromSchema)
        {
            Contracts.CheckValue(fromSchema, nameof(fromSchema));
            return fromSchema.MakeNewEstimator();
        }
    }
}
