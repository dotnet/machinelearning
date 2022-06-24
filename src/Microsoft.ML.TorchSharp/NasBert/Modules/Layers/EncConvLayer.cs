// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.TorchSharp.Extensions;
using TorchSharp;
using TorchSharp.Modules;

namespace Microsoft.ML.TorchSharp.NasBert.Modules.Layers
{
    internal sealed class EncConvLayer : Layer
    {
#pragma warning disable MSML_PrivateFieldName // Private field name not in: _camelCase format
        private readonly Sequential Conv1;
        private readonly LayerNorm LayerNorm1;

        private readonly Sequential Conv2;
        private readonly LayerNorm LayerNorm2;
#pragma warning restore MSML_PrivateFieldName // Private field name not in: _camelCase format

        public EncConvLayer(int channel, int kernelSize, double dropoutRate, string activationFn,
            double activationDropoutRate) : base(nameof(EncConvLayer))
        {
            Conv1 = torch.nn.Sequential(
                ("conv", new ConvSeparable(channel, channel, kernelSize, kernelSize / 2, dropoutRate)),
                ("activation", new ActivationFunction(activationFn)),
                ("dropout", torch.nn.Dropout(activationDropoutRate))
            );
            LayerNorm1 = torch.nn.LayerNorm(new long[] { channel });

            Conv2 = torch.nn.Sequential(
                ("conv", new ConvSeparable(channel, channel, kernelSize, kernelSize / 2, dropoutRate)),
                ("activation", new ActivationFunction(activationFn)),
                ("dropout", torch.nn.Dropout(activationDropoutRate))
            );
            LayerNorm2 = torch.nn.LayerNorm(new long[] { channel });

            RegisterComponents();
        }

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public override torch.Tensor forward(torch.Tensor x, Dictionary<string, object> param = null)
        {
            if (!ParseArguments(param, out var selfAttentionPaddingMask))
            {
                throw new ArgumentException($"Invalid arguments: {param}.");
            }

            using var x1 = ForwardOneLayer(x, selfAttentionPaddingMask, Conv1, LayerNorm1);
            return ForwardOneLayer(x1, selfAttentionPaddingMask, Conv2, LayerNorm2);
        }

        private static torch.Tensor ForwardOneLayer(torch.Tensor input, torch.Tensor selfAttentionPaddingMask,
            torch.nn.Module convLayer, torch.nn.Module layerNorm)
        {
            using var disposeScope = torch.NewDisposeScope();

            torch.Tensor x = selfAttentionPaddingMask.IsNull()
                ? input.alias()
                : input.masked_fill(selfAttentionPaddingMask.T.unsqueeze(-1), 0);

            var conv = convLayer.forward(x);
            conv.add_(input);
            var norm = layerNorm.forward(conv);
            return norm.MoveToOuterDisposeScope();
        }

        public override void CloseLayerNormTraining()
        {
            LayerNorm1.eval();
            LayerNorm2.eval();
        }

        private static bool ParseArguments(IReadOnlyDictionary<string, object> param, out torch.Tensor selfAttentionPaddingMask)
        {
            selfAttentionPaddingMask = null;
            if (param == null) return false;

            if (param.ContainsKey(PaddingMaskKey)) selfAttentionPaddingMask = (torch.Tensor)param[PaddingMaskKey];
            return true;
        }
    }
}
