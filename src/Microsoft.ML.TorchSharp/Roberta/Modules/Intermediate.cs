// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using TorchSharp;
using TorchSharp.Modules;

namespace Microsoft.ML.TorchSharp.Roberta.Modules
{
    internal class Intermediate : torch.nn.Module<torch.Tensor, torch.Tensor>
    {
#pragma warning disable MSML_GeneralName // This name should be PascalCased
        public readonly Linear dense;
        public readonly GELU gelu;
        private bool _disposedValue;

        public Intermediate(long hiddenSize, long ffnHiddenSize) : base(nameof(Intermediate))
        {
            dense = torch.nn.Linear(hiddenSize, ffnHiddenSize, true);
            gelu = torch.nn.GELU();
            RegisterComponents();
        }

        public override torch.Tensor forward(torch.Tensor t)
        {
            using var disposeScope = torch.NewDisposeScope();
            t = dense.forward(t);
            t = gelu.forward(t);
            return t.MoveToOuterDisposeScope();
        }

        protected override void Dispose(bool disposing)
        {
            if (!_disposedValue)
            {
                if (disposing)
                {
                    dense.Dispose();
                    gelu.Dispose();
                    _disposedValue = true;
                }
            }

            base.Dispose(disposing);
        }
    }
}
