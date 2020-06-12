// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms.Onnx
{
    // This is a helper class that is required to use the <see cref="DnnImageFeaturizerEstimator"/>.
    // Note that by default, it is not usable as it does not have any valid methods that return an <see cref="EstimatorChain{TLastTransformer}"/>
    // that is used by the DnnImageFeaturizeEstimator.
    // In order to use this, at least one model project with the corresponding extension methods must by included.
    /// <summary>
    /// Helper class for selecting a pre-trained DNN image featurization model to use in the <see cref="DnnImageFeaturizerEstimator"/>.
    /// </summary>
    /// <seealso cref="OnnxCatalog.DnnFeaturizeImage(TransformsCatalog, string, Func{DnnImageFeaturizerInput, EstimatorChain{ColumnCopyingTransformer}}, string)"/>
    public sealed class DnnImageModelSelector
    {
    }

    /// <summary>
    /// Helper class for storing all the inputs to an extension method on a <see cref="DnnImageModelSelector"/> required to return
    /// a chain of two <see cref="OnnxScoringEstimator"/>.
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
    /// Applies a pre-trained deep neural network (DNN) model to featurize input image data.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// ###  Estimator Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Does this estimator need to look at the data to train its parameters? | No |
    /// | Input column data type | Vector of <xref:System.Single> |
    /// | Output column data type | Vector of <xref:System.Single>, the size of the vector depends on the pre-trained DNN |
    /// | Exportable to ONNX | No |
    ///
    /// NuGet requirements:
    ///	- Microsoft.ML.OnnxTransformer
    /// - Microsoft.ML.OnnxRuntime.Gpu (only if GPU processing is used)
    /// - Each pre-trained DNN model has a separate NuGet that must be included if that model is used:
    ///   - Microsoft.ML.DnnImageFeaturizer.AlexNet
    ///   - Microsoft.ML.DnnImageFeaturizer.ResNet18
    ///   - Microsoft.ML.DnnImageFeaturizer.ResNet50
    ///   - Microsoft.ML.DnnImageFeaturizer.ResNet101
    ///
    /// The resulting transformer creates a new column, named as specified in the output column name parameters,
    /// where a pre-trained deep neural network is applied to the input image data.
    ///
    /// This estimator is a wrapper around a <xref:Microsoft.ML.Transforms.Onnx.OnnxScoringEstimator> with the current
    /// available DNN pre-trained models and several <xref:Microsoft.ML.Transforms.ColumnCopyingEstimator>.
    /// The <xref:Microsoft.ML.Transforms.ColumnCopyingEstimator> are needed to allow arbitrary column input and output
    /// names, since otherwise the <xref:Microsoft.ML.Transforms.Onnx.OnnxScoringEstimator> would require the column names to match
    /// the names of the ONNX model nodes.
    ///
    /// Any platform requirement for this estimator will follow the requirements on the <xref:Microsoft.ML.Transforms.Onnx.OnnxScoringEstimator>.
    ///
    /// Check the See Also section for links to usage examples.
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="OnnxCatalog.DnnFeaturizeImage(TransformsCatalog, string, Func{DnnImageFeaturizerInput, EstimatorChain{ColumnCopyingTransformer}}, string)"/>
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
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>. The column data is a known-sized vector of <see cref="System.Single"/>.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source. The column data must be a known-sized vector of <see cref="System.Single"/>.</param>
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
