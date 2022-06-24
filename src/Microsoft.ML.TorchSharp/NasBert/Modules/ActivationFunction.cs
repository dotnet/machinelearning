// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using TorchSharp;

namespace Microsoft.ML.TorchSharp.NasBert.Modules
{

    internal sealed class ActivationFunction : BaseModule
    {
        private readonly torch.nn.Module _function;

        public ActivationFunction(string name) : base(name)
        {
            _function = name?.ToLower() switch
            {
                "relu" => torch.nn.ReLU(),
                "gelu" => torch.nn.GELU(),
                "gelu_fast" => new GeLUFast(),
                "tanh" => torch.nn.Tanh(),
                "linear" => torch.nn.Identity(),
                _ => throw new NotSupportedException($"Activation function {name} not supported.")
            };
        }

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public override torch.Tensor forward(torch.Tensor x)
        {
            return _function.forward(x);
        }

        public override string GetName()
        {
            return _function.GetName();
        }
    }

    /// <summary>
    /// See https://arxiv.org/pdf/1606.08415.pdf:
    /// y = 0.5x * (1 + tanh(sqrt(2/Pi) * (x + 0.044715 x^3)))
    /// </summary>
    public class GeLUFast : torch.nn.Module
    {
        private readonly double _alpha = Math.Sqrt(2 / Math.PI);
        private readonly double _beta = 0.044715;

        public GeLUFast() : base(nameof(GeLUFast)) { }

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Need to match TorchSharp.")]
        public override torch.Tensor forward(torch.Tensor x)
        {
            using var disposeScope = torch.NewDisposeScope();
            var x1 = torch.pow(x, 3).mul_(_beta).add_(x).mul_(_alpha);  // sqrt(2/Pi) * (x + 0.044715 x^3)
            var y = torch.nn.functional.tanh(x1).add_(1.0).mul_(0.5).mul_(x);
            return y.MoveToOuterDisposeScope();
        }
    }
}
