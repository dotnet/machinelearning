// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.TorchSharp.Utils;
using TorchSharp;
using TorchSharp.Modules;

namespace Microsoft.ML.TorchSharp.NasBert.Modules.Layers
{
    internal sealed class FeedForwardLayer : Layer
    {
#pragma warning disable MSML_PrivateFieldName // Private field name not in: _camelCase format
        private readonly Sequential FullConnects;
        private readonly LayerNorm FinalLayerNorm;
#pragma warning restore MSML_PrivateFieldName // Private field name not in: _camelCase format


        public FeedForwardLayer(
            int embeddingDim = 768,
            int ffnEmbeddingDim = 3072,
            double dropoutRate = 0.1,
            double activationDropoutRate = 0.1,
            string activationFn = "relu",
            bool dynamicDropout = false)
            : base(nameof(FeedForwardLayer))
        {
            // Initialize parameters
            if (dynamicDropout)
            {
                dropoutRate = CalculateDropout(dropoutRate, embeddingDim,
                    SearchSpace.HiddenSizeChoices[SearchSpace.HiddenSizeChoices.Length - 1]);
                activationDropoutRate = CalculateDropout(activationDropoutRate, embeddingDim,
                    SearchSpace.HiddenSizeChoices[SearchSpace.HiddenSizeChoices.Length - 1]);
            }

            // Layer norm associated with the position wise feed-forward NN
            var fullConnected1 = torch.nn.Linear(embeddingDim, ffnEmbeddingDim);
            var activation = new ActivationFunction(activationFn);
            var activationDropoutLayer = torch.nn.Dropout(activationDropoutRate);
            var fullConnected2 = torch.nn.Linear(ffnEmbeddingDim, embeddingDim);
            var dropoutLayer = torch.nn.Dropout(dropoutRate);

            ModelUtils.InitNormal(fullConnected1.weight, mean: 0.0, std: 0.02);
            ModelUtils.InitZeros(fullConnected1.bias);
            ModelUtils.InitNormal(fullConnected2.weight, mean: 0.0, std: 0.02);
            ModelUtils.InitZeros(fullConnected2.bias);

            FullConnects = torch.nn.Sequential(
                ("fc1", fullConnected1),
                ("activation", activation),
                ("dropout1", activationDropoutLayer),
                ("fc2", fullConnected2),
                ("dropout2", dropoutLayer)
            );
            FinalLayerNorm = torch.nn.LayerNorm(new long[] { embeddingDim });

            RegisterComponents();
        }

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public override torch.Tensor forward(torch.Tensor x, Dictionary<string, object> param)
        {
            using var layerOutput = FullConnects.forward(x);
            using var layerOuptutIntermediate = layerOutput.add_(x);
            return FinalLayerNorm.forward(layerOutput);
        }

        public override void CloseLayerNormTraining() => FinalLayerNorm.eval();

        private static double CalculateDropout(double dropout, int sampleEmbeddingDim, int superEmbeddingDim)
            => dropout * sampleEmbeddingDim / superEmbeddingDim;
    }
}
