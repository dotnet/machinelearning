// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Transforms.Onnx;

namespace Microsoft.ML.Transforms.StaticPipe
{
    public static class DnnImageFeaturizerStaticExtensions
    {
        private sealed class OutColumn : Vector<float>
        {
            public PipelineColumn Input { get; }

            public OutColumn(Vector<float> input, Func<DnnImageFeaturizerInput, EstimatorChain<ColumnCopyingTransformer>> modelFactory)
                : base(new Reconciler(modelFactory), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly Func<DnnImageFeaturizerInput, EstimatorChain<ColumnCopyingTransformer>> _modelFactory;

            public Reconciler(Func<DnnImageFeaturizerInput, EstimatorChain<ColumnCopyingTransformer>> modelFactory)
            {
                _modelFactory = modelFactory;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);

                var outCol = (OutColumn)toOutput[0];
                return new DnnImageFeaturizerEstimator(env, outputNames[outCol], _modelFactory, inputNames[outCol.Input]);
            }
        }

        /// <summary>
        /// Creates and applies a DnnImageFeaturizer transform to be used by the static API.
        /// <see cref="DnnImageFeaturizerEstimator"/> for more information about how the transformation works.
        /// </summary>
        /// <param name="input">Vector of image pixel weights.</param>
        /// <param name="modelFactory">An extension method on the <see cref="DnnImageModelSelector"/> that creates a chain of two
        /// <see cref="OnnxScoringEstimator"/>s (one for preprocessing and one with a pretrained image DNN) with specific models
        /// included in a package together with that extension method.
        /// For an example, see Microsoft.ML.DnnImageFeaturizer.ResNet18 </param>
        /// <returns>A vector of float feature weights based on the input image.</returns>
        public static Vector<float> DnnImageFeaturizer(this Vector<float> input, Func<DnnImageFeaturizerInput, EstimatorChain<ColumnCopyingTransformer>> modelFactory)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutColumn(input, modelFactory);
        }
    }
}
