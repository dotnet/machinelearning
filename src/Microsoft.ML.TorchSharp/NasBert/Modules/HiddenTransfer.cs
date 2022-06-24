// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.TorchSharp.Utils;
using TorchSharp;
using TorchSharp.Modules;

namespace Microsoft.ML.TorchSharp.NasBert.Modules
{
    internal abstract class HiddenTransfer : BaseModule
    {
        protected HiddenTransfer(string name) : base(name) { }

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public abstract torch.Tensor forward(torch.Tensor x, int hiddenSize, bool inputTransfer);
    }

    internal sealed class HiddenTransferDiscrete : HiddenTransfer
    {
#nullable enable
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_PrivateFieldName:Private field name not in: _camelCase format", Justification = "Have to match TorchSharp.")]
        private readonly Linear? InHiddenTransfer;
#nullable disable

        public HiddenTransferDiscrete(int hiddenSize1, int hiddenSize2) : base(nameof(HiddenTransferDiscrete))
        {
            if (hiddenSize1 != hiddenSize2)
            {
                InHiddenTransfer = torch.nn.Linear(hiddenSize1, hiddenSize2);

                ModelUtils.InitNormal(InHiddenTransfer.weight, mean: 0.0, std: 0.02);
                ModelUtils.InitZeros(InHiddenTransfer.bias);
            }

            RegisterComponents();
        }

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public override torch.Tensor forward(torch.Tensor x, int hiddenSize, bool inputTransfer)
        {
            return (!inputTransfer && InHiddenTransfer != null)
                ? InHiddenTransfer.forward(x)
                : x.alias();
        }
    }
}
