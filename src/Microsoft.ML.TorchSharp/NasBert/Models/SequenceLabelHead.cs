// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.TorchSharp.Utils;
using TorchSharp;
using TorchSharp.Modules;

namespace Microsoft.ML.TorchSharp.NasBert.Models
{
    internal sealed class SequenceLabelHead : BaseHead, torch.nn.IModule<torch.Tensor, torch.Tensor>
    {
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_PrivateFieldName:Private field name not in: _camelCase format", Justification = "Has to match TorchSharp model.")]
        private readonly Sequential Classifier;

        private bool _disposedValue;

        public SequenceLabelHead(int inputDim, int numLabels, double dropoutRate)
            : base(nameof(SequenceLabelHead))
        {
            var dropoutLayer = torch.nn.Dropout(dropoutRate);
            var dense = torch.nn.Linear(inputDim, numLabels);

            ModelUtils.InitXavierUniform(dense.weight);
            ModelUtils.InitZeros(dense.bias);

            Classifier = torch.nn.Sequential(
                ("dropout", dropoutLayer),
                ("dense", dense)
            );

            RegisterComponents();
        }

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp")]
        public torch.Tensor call(torch.Tensor features)
        {
            return Classifier.forward(features);
        }

        protected override void Dispose(bool disposing)
        {
            if (!_disposedValue)
            {
                if (disposing)
                {
                    Classifier.Dispose();
                    _disposedValue = true;
                }
            }

            base.Dispose(disposing);
        }
    }
}
