// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.StaticPipe;

namespace Microsoft.ML.Transforms.StaticPipe
{
    public static class TensorFlowStaticExtensions
    {
        private sealed class OutColumn : Vector<float>
        {
            public PipelineColumn Input { get; }

            public OutColumn(Vector<float> input, string modelFile, bool addBatchDimensionInput)
                : base(new Reconciler(modelFile, addBatchDimensionInput), input)
            {
                Input = input;
            }

            public OutColumn(Vector<float> input, TensorFlowModel tensorFlowModel, bool addBatchDimensionInput)
                : base(new Reconciler(tensorFlowModel, addBatchDimensionInput), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly string _modelFile;
            private readonly TensorFlowModel _tensorFlowModel;
            private readonly bool _addBatchDimensionInput;

            public Reconciler(string modelFile, bool addBatchDimensionInput)
            {
                Contracts.AssertNonEmpty(modelFile);
                _modelFile = modelFile;
                _tensorFlowModel = null;
                _addBatchDimensionInput = addBatchDimensionInput;
            }

            public Reconciler(TensorFlowModel tensorFlowModel, bool addBatchDimensionInput)
            {
                Contracts.CheckValue(tensorFlowModel, nameof(tensorFlowModel));

                _modelFile = null;
                _tensorFlowModel = tensorFlowModel;
                _addBatchDimensionInput = addBatchDimensionInput;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);

                var outCol = (OutColumn)toOutput[0];
                if (_modelFile == null)
                    return new TensorFlowEstimator(env, new[] { outputNames[outCol] }, new[] { inputNames[outCol.Input] }, _tensorFlowModel, _addBatchDimensionInput);
                else
                    return new TensorFlowEstimator(env, new[] { outputNames[outCol] }, new[] { inputNames[outCol.Input] }, _modelFile, _addBatchDimensionInput);
            }
        }

        // REVIEW: this method only covers one use case of using TensorFlow models: consuming one
        // input and producing one output of floats.
        // We could consider selectively adding some more extensions to enable common scenarios.
        /// <summary>
        /// Load the TensorFlow model from <paramref name="modelFile"/> and run it on the input column and extract one output column.
        /// The inputs and outputs are matched to TensorFlow graph nodes by name.
        /// </summary>
        public static Vector<float> ApplyTensorFlowGraph(this Vector<float> input, string modelFile, bool addBatchDimensionInput = false)
        {
            Contracts.CheckValue(input, nameof(input));
            Contracts.CheckNonEmpty(modelFile, nameof(modelFile));
            return new OutColumn(input, modelFile, addBatchDimensionInput);
        }

        /// <summary>
        /// Run a TensorFlow model provided through <paramref name="tensorFlowModel"/> on the input column and extract one output column.
        /// The inputs and outputs are matched to TensorFlow graph nodes by name.
        /// </summary>
        public static Vector<float> ApplyTensorFlowGraph(this Vector<float> input, TensorFlowModel tensorFlowModel, bool addBatchDimensionInput = false)
        {
            Contracts.CheckValue(input, nameof(input));
            Contracts.CheckValue(tensorFlowModel, nameof(tensorFlowModel));
            return new OutColumn(input, tensorFlowModel, addBatchDimensionInput);
        }
    }
}
