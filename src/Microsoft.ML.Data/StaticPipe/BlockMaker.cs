// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data.StaticPipe.Runtime;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data.StaticPipe
{
    public abstract class BlockMaker<TTupleShape>
    {
        private protected readonly IHostEnvironment Env;

        /// <summary>
        /// Constructor for a block maker.
        /// </summary>
        /// <param name="env"></param>
        private protected BlockMaker(IHostEnvironment env)
        {
            Contracts.CheckValue(env, nameof(env));
            Env = env;
        }

        public Estimator<TTupleShape, TTupleOutShape, ITransformer> CreateEstimator<TTupleOutShape>(Func<TTupleShape, TTupleOutShape> mapper)
        {
            Contracts.CheckValue(mapper, nameof(mapper));

            using (var ch = Env.Start(nameof(CreateEstimator)))
            {
                var method = mapper.Method;

                // Construct the dummy column structure, then apply the mapping.
                var input = PipelineColumnAnalyzer.CreateAnalysisInstance<TTupleShape>(out var fakeReconciler);
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

                var readerEst = StaticPipeUtils.GeneralFunctionAnalyzer(Env, ch, input, fakeReconciler, mapper, out var est, NameMap);
                ch.Assert(readerEst == null);
                ch.AssertValue(est);
                ch.Done();

                return new Estimator<TTupleShape, TTupleOutShape, ITransformer>(Env, est);
            }
        }
    }
}
