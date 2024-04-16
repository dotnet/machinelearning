// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using TorchSharp;
using TorchSharp.Modules;

namespace Microsoft.ML.TorchSharp.Roberta.Modules
{
    internal class Output : torch.nn.Module<torch.Tensor, torch.Tensor, torch.Tensor>
    {
#pragma warning disable MSML_GeneralName // This name should be PascalCased
        public readonly Linear dense;
        public readonly LayerNorm LayerNorm;
        public readonly Dropout Dropout;
        private bool _disposedValue;

        public Output(long ffnHiddenSize, long hiddenSize, double outputDropoutRate) : base(nameof(Output))
        {
            dense = torch.nn.Linear(ffnHiddenSize, hiddenSize, true);
            Dropout = torch.nn.Dropout(outputDropoutRate);
            LayerNorm = torch.nn.LayerNorm(new long[] { hiddenSize });
            RegisterComponents();
        }

        public override torch.Tensor forward(torch.Tensor hiddenStates, torch.Tensor inputTensor)
        {
            using var disposeScope = torch.NewDisposeScope();
            hiddenStates = dense.forward(hiddenStates);
            hiddenStates = Dropout.forward(hiddenStates);
            hiddenStates = LayerNorm.forward(hiddenStates + inputTensor);
            return hiddenStates.MoveToOuterDisposeScope();
        }

        protected override void Dispose(bool disposing)
        {
            if (!_disposedValue)
            {
                if (disposing)
                {
                    dense.Dispose();
                    LayerNorm.Dispose();
                    Dropout.Dispose();
                    _disposedValue = true;
                }
            }

            base.Dispose(disposing);
        }
    }
}
