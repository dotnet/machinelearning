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
    internal sealed class ConvSeparable : BaseModule
    {
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_PrivateFieldName:Private field name not in: _camelCase format", Justification = "Have to match TorchSharp model.")]
        private readonly Sequential Conv;

        public ConvSeparable(int inChannels, int outChannels, int kernelSize, int padding, double dropout)
            : base(nameof(ConvSeparable))
        {
            // Weight shape: [InChannels, 1, KernelSize]
            var conv1 = torch.nn.Conv1d(inChannels, inChannels, kernelSize, padding: padding, groups: inChannels);
            // Weight shape: [OutChannels, InChannels, 1], Bias shape: [OutChannels]
            var conv2 = torch.nn.Conv1d(inChannels, outChannels, 1, padding: 0L, groups: 1, bias: true);

            var std = Math.Sqrt((4 * (1.0 - dropout)) / (kernelSize * inChannels));
            ModelUtils.InitNormal(conv1.weight, mean: 0, std: std);
            ModelUtils.InitNormal(conv2.weight, mean: 0, std: std);
            ModelUtils.InitConstant(conv2.bias, 0);

            Conv = torch.nn.Sequential(
                ("conv1", conv1),
                ("conv2", conv2)
            );

            RegisterComponents();
        }

        /// <summary>
        /// Input shape: [SeqLen, BatchSize, InChannel]
        /// Output shape: [SeqLen - KernelSize + 1, BatchSize, OutChannel]
        /// </summary>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public override torch.Tensor forward(torch.Tensor x)
        {
            using var x1 = x.permute(1, 2, 0);
            using var conv = Conv.forward(x1);
            return conv.permute(2, 0, 1);
        }
    }
}
