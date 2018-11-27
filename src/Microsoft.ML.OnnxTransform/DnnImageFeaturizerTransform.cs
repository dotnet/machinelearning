﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.StaticPipe.Runtime;
using System;
using System.Collections.Generic;

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// This is a helper class that is required to use the <see cref="DnnImageFeaturizerEstimator"/>.
    /// Note that by default, it is not usable as it does not have any valid methods that return an <see cref="EstimatorChain{TLastTransformer}"/>
    /// that is used by the DnnImageFeaturizeEstimator.
    /// In order to use this, at least one model project with the corresponding extension methods must by included.
    /// See Microsoft.ML.DNNImageFeaturizer.ResNet18 for an example.
    /// </summary>
    public sealed class DnnImageModelSelector
    {
    }

    /// <summary>
    /// This is a helper class used to store all the inputs to an extension method on a DnnImageModelSelector required to return
    /// a chain of two <see cref="OnnxScoringEstimator"/>s.
    /// </summary>
    public sealed class DnnImageFeaturizerInput
    {
        public IHostEnvironment Environment { get; }
        public string InputColumn { get; }
        public DnnImageModelSelector ModelSelector { get; }
        public string OutputColumn { get; }

        public DnnImageFeaturizerInput(IHostEnvironment env, string inputColumn, string outputColumn, DnnImageModelSelector modelSelector)
        {
            Environment = env;
            InputColumn = inputColumn;
            OutputColumn = outputColumn;
            ModelSelector = modelSelector;
        }
    }

    /// <summary>
    /// The Dnn Image Featurizer is just a wrapper around two <see cref="OnnxScoringEstimator"/>s and three <see cref="ColumnCopyingEstimator"/>
    /// with present pretrained DNN models. The ColumnsCopying are there to allow arbitrary column input and output names, as by default
    /// the ONNXTransform requires the names of the columns to be identical to the names of the ONNX model nodes.
    /// Note that because of this, it only works on Windows machines as that is a constraint of the OnnxTransform.
    /// </summary>
    public sealed class DnnImageFeaturizerEstimator : IEstimator<TransformerChain<ColumnCopyingTransformer>>
    {
        private readonly EstimatorChain<ColumnCopyingTransformer> _modelChain;

        /// <summary>
        /// Constructor for the estimator for a DnnImageFeaturizer transform.
        /// </summary>
        /// <param name="env">Host environment.</param>
        /// <param name="modelFactory">An extension method on the <see cref="DnnImageModelSelector"/> that creates a chain of two
        /// <see cref="OnnxScoringEstimator"/>s (one for preprocessing and one with a pretrained image DNN) with specific models
        /// included in a package together with that extension method. It also contains three <see cref="ColumnCopyingEstimator"/>s
        /// to allow arbitrary column naming, as the ONNXEstimators require very specific naming based on the models.
        /// For an example, see Microsoft.ML.DnnImageFeaturizer.ResNet18 </param>
        /// <param name="inputColumn">inputColumn column name.</param>
        /// <param name="outputColumn">Output column name.</param>
        public DnnImageFeaturizerEstimator(IHostEnvironment env, Func<DnnImageFeaturizerInput, EstimatorChain<ColumnCopyingTransformer>> modelFactory, string inputColumn, string outputColumn)
        {
            _modelChain = modelFactory( new DnnImageFeaturizerInput(env, inputColumn, outputColumn, new DnnImageModelSelector()));
        }

        /// <summary>
        /// Note that OnnxEstimator which this is based on is a trivial estimator, so this does not do any actual training,
        /// just verifies the schema.
        /// </summary>
        public TransformerChain<ColumnCopyingTransformer> Fit(IDataView input)
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
                return new DnnImageFeaturizerEstimator(env, _modelFactory, inputNames[outCol.Input], outputNames[outCol]);
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
