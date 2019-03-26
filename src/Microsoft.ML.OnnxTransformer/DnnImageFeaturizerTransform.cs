// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms.Onnx
{
    /// <summary>
    /// This is a helper class that is required to use the <see cref="DnnImageFeaturizerEstimator"/>.
    /// Note that by default, it is not usable as it does not have any valid methods that return an <see cref="EstimatorChain{TLastTransformer}"/>
    /// that is used by the DnnImageFeaturizeEstimator.
    /// In order to use this, at least one model project with the corresponding extension methods must by included.
    /// </summary>
    /// <seealso cref="OnnxCatalog.DnnFeaturizeImage(TransformsCatalog, string, Func{DnnImageFeaturizerInput, EstimatorChain{ColumnCopyingTransformer}}, string)"/>
    public sealed class DnnImageModelSelector
    {
    }

    /// <summary>
    /// This is a helper class used to store all the inputs to an extension method on a DnnImageModelSelector required to return
    /// a chain of two <see cref="OnnxScoringEstimator"/>s.
    /// </summary>
    public sealed class DnnImageFeaturizerInput
    {
        [BestFriend]
        internal IHostEnvironment Environment { get; }
        public string InputColumn { get; }
        public DnnImageModelSelector ModelSelector { get; }
        public string OutputColumn { get; }

        internal DnnImageFeaturizerInput(string outputColumnName, string inputColumnName, IHostEnvironment env, DnnImageModelSelector modelSelector)
        {
            Environment = env;
            InputColumn = inputColumnName;
            OutputColumn = outputColumnName;
            ModelSelector = modelSelector;
        }
    }

    /// <summary>
    /// The Dnn Image Featurizer is just a wrapper around two <see cref="OnnxScoringEstimator"/>s and three <see cref="ColumnCopyingEstimator"/>
    /// with present pretrained DNN models. The ColumnsCopying are there to allow arbitrary column input and output names, as by default
    /// the ONNXTransformer requires the names of the columns to be identical to the names of the ONNX model nodes.
    /// Note that because of this, it only works on Windows machines as that is a constraint of the OnnxTransformer.
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
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        internal DnnImageFeaturizerEstimator(IHostEnvironment env, string outputColumnName, Func<DnnImageFeaturizerInput, EstimatorChain<ColumnCopyingTransformer>> modelFactory, string inputColumnName = null)
        {
            _modelChain = modelFactory(new DnnImageFeaturizerInput(outputColumnName, inputColumnName ?? outputColumnName, env, new DnnImageModelSelector()));
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
}
