// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using TorchSharp.Tensor;

namespace Microsoft.ML.Torch
{
    /// <summary>
    /// This class holds the information related to Torch model.
    /// It provides some convenient methods to query the model schema as well as
    /// creation of <see cref="TorchScorerEstimator"/> object.
    /// </summary>
    public sealed class TorchModel
    {
        internal readonly TorchModuleWrapper Module;
        private readonly IHostEnvironment _env;

        internal string ModelPath { get; }

        /// <summary>
        /// Instantiates <see cref="TorchModel"/>.
        /// </summary>
        /// <param name="env">An <see cref="IHostEnvironment"/> object.</param>
        /// <param name="module">The TorchSharp module object containing the model</param>
        /// <param name="modelPath">The file path to the model.</param>
        [BestFriend]
        internal TorchModel(IHostEnvironment env, TorchModuleWrapper module, string modelPath)
        {
            _env = env;
            Module = module;
            ModelPath = modelPath;
        }

        /// <summary>
        /// Scores a dataset using a pre-traiend <a href="https://www.pytorch.org/">Torch</a> model.
        /// </summary>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ScoreTorchModel](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Torch/AlexNet.cs)]
        /// ]]>
        /// </format>
        /// </example>
        internal TorchScorerEstimator ScoreTorchModel(string outputColumnName, long[][] shapes, string[] inputColumnNames = null)
        {
            var options = new TorchScorerEstimator.Options {
                OutputColumnName = outputColumnName,
                InputColumnNames = inputColumnNames ?? new[] { outputColumnName },
                InputShapes = shapes,
                ModelLocation = ModelPath
            };
            return new TorchScorerEstimator(_env, options, Module);
        }

        /// <summary>
        /// Scores a dataset using a pre-traiend <a href="https://www.pytorch.org/">Torch</a> model.
        /// </summary>
        /// <param name="outputColumnName">The name of the requested model output. The data type is a vector of <see cref="System.Single"/></param>
        /// <param name="shape">The shape of the input vector expected by the model.</param>
        /// <param name="inputColumnName"> The name of the model input. The data type is a vector of <see cref="System.Single"/>.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ScoreTorchModel](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Torch/AlexNet.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public TorchScorerEstimator ScoreTorchModel(string outputColumnName, long[] shape, string inputColumnName = null)
        {
            var options = new TorchScorerEstimator.Options
            {
                OutputColumnName = outputColumnName,
                InputColumnNames = inputColumnName != null ? new[] { inputColumnName } : new[] { outputColumnName },
                InputShapes = new[] { shape },
                ModelLocation = ModelPath
            };
            return new TorchScorerEstimator(_env, options, Module);
        }

        /// <summary>
        /// Scores a dataset using a <a href="https://www.pytorch.org/">Torch</a> model pre-trained using ML.NET.
        /// </summary>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ScoreTorchModel](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/TorchTransform.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public TorchTrainerEstimator TrainTorchModel(
            string outputColumnName,
            string inputColumnName,
            string labelColumnName,
            long[] shape,
            TorchSharp.NN.Optimizer optimizer,
            TorchSharp.NN.LossFunction.Loss loss,
            int epochs,
            int batchSize)
        {
            var options = new TorchTrainerEstimator.Options
            {
                OutputColumnName = outputColumnName,
                InputColumnNames = new[] { inputColumnName },
                InputShapes = new[] { shape },
                InputLabelColumnName = labelColumnName,
                Optimizer = optimizer,
                Loss = loss,
                Epochs = epochs,
                BatchSize = batchSize
            };
            return new TorchTrainerEstimator(_env, options, this);
        }
    }
}