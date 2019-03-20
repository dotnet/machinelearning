// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Transforms.Onnx;

namespace Microsoft.ML.Transforms.StaticPipe
{
    public static class OnnxStaticExtensions
    {

        private sealed class OutColumn : Vector<float>
        {
            public PipelineColumn Input { get; }

            public OutColumn(Vector<float> input, string modelFile)
                : base(new Reconciler(modelFile), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly string _modelFile;

            public Reconciler(string modelFile)
            {
                Contracts.AssertNonEmpty(modelFile);
                _modelFile = modelFile;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);

                var outCol = (OutColumn)toOutput[0];
                return new OnnxScoringEstimator(env,  new[] { outputNames[outCol] }, new[] { inputNames[outCol.Input] }, _modelFile);
            }
        }

        /// <summary>
        /// Run a Onnx model on the input column and extract one output column.
        /// The inputs and outputs are matched to Onnx graph nodes by name.
        /// </summary>
        public static Vector<float> ApplyOnnxModel(this Vector<float> input, string modelFile)
        {
            Contracts.CheckValue(input, nameof(input));
            Contracts.CheckNonEmpty(modelFile, nameof(modelFile));
            return new OutColumn(input, modelFile);
        }
    }
}
