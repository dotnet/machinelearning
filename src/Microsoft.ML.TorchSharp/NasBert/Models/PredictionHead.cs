// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.TorchSharp.Utils;
using TorchSharp;
using TorchSharp.Modules;

namespace Microsoft.ML.TorchSharp.NasBert.Models
{
    internal sealed class PredictionHead : BaseHead
    {
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_PrivateFieldName:Private field name not in: _camelCase format", Justification = "Has to match TorchSharp model.")]
        private readonly Sequential Classifier;

        public PredictionHead(int inputDim, int numClasses, double dropoutRate)
            : base(nameof(PredictionHead))
        {
            var dropoutLayer = torch.nn.Dropout(dropoutRate);
            var dense = torch.nn.Linear(inputDim, numClasses);

            ModelUtils.InitXavierUniform(dense.weight);
            ModelUtils.InitZeros(dense.bias);

            Classifier = torch.nn.Sequential(
                ("dropout1", dropoutLayer),
                ("dense", dense)
            );

            RegisterComponents();
        }

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp")]
        public override torch.Tensor forward(torch.Tensor features)
        {
            // TODO: try whitening-like techniques
            // take <s> token (equiv. to [CLS])
            using var x = features[torch.TensorIndex.Colon, torch.TensorIndex.Single(0), torch.TensorIndex.Colon];
            return Classifier.forward(x);
        }
    }
}
