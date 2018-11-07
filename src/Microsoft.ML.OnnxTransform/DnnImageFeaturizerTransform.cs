// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.StaticPipe.Runtime;
using System;
using System.Collections.Generic;

namespace Microsoft.ML.Transforms
{
    public class DnnImageModelSelector
    {
        public readonly IHostEnvironment Env;
        public readonly string Input;
        public readonly string Output;

        public DnnImageModelSelector(IHostEnvironment env, string input, string output)
        {
            Env = env;
            Input = input;
            Output = output;
        }
    }

    public sealed class DnnImageFeaturizerEstimator : IEstimator<TransformerChain<OnnxTransform>>
    {
        private readonly EstimatorChain<OnnxTransform> _modelChain;

        public DnnImageFeaturizerEstimator(IHostEnvironment env, Func<DnnImageModelSelector, EstimatorChain<OnnxTransform>> model, string input, string output)
        {
            _modelChain = model(new DnnImageModelSelector(env, input, output));
        }

        public TransformerChain<OnnxTransform> Fit(IDataView input)
        {
            return _modelChain.Fit(input);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            return _modelChain.GetOutputSchema(inputSchema);
        }
    }

    public static class DnnImageFeaturizerStaticExtensions
    {
        private sealed class OutColumn : Vector<float>
        {
            public PipelineColumn Input { get; }

            public OutColumn(Vector<float> input, Func<DnnImageModelSelector, EstimatorChain<OnnxTransform>> model)
                : base(new Reconciler(model), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly Func<DnnImageModelSelector, EstimatorChain<OnnxTransform>> _model;

            public Reconciler(Func<DnnImageModelSelector, EstimatorChain<OnnxTransform>> model)
            {
                _model = model;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);

                var outCol = (OutColumn)toOutput[0];
                return new DnnImageFeaturizerEstimator(env, _model, inputNames[outCol.Input], outputNames[outCol]);
            }
        }

        public static Vector<float> DnnImageFeaturizer(this Vector<float> input, Func<DnnImageModelSelector, EstimatorChain<OnnxTransform>> model)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutColumn(input, model);
        }
    }
}
