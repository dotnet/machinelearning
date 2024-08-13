// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Core;

#pragma warning disable MSML_GeneralName // This name should be PascalCased
internal class RMSNorm : torch.nn.Module<Tensor, Tensor>
#pragma warning restore MSML_GeneralName // This name should be PascalCased
{
    private readonly int _dim;
    private readonly float _eps;
#pragma warning disable MSML_PrivateFieldName // Private field name not in: _camelCase format
    private readonly Parameter weight;
#pragma warning restore MSML_PrivateFieldName // Private field name not in: _camelCase format

    public RMSNorm(
        int hiddenSize,
        float eps = 1e-6f,
        ScalarType dtype = ScalarType.Float32)
        : base(nameof(RMSNorm))
    {
        this._dim = hiddenSize;
        this._eps = eps;

        // the gamma scalar
        this.weight = torch.nn.Parameter(torch.ones(this._dim, dtype: dtype));
    }

    private Tensor Norm(Tensor x)
    {
        // (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        // rsqrt = 1 / sqrt
        var output = x * torch.rsqrt(x.pow(2).mean([-1L], keepdim: true) + this._eps);
        return output;
    }

#pragma warning disable MSML_GeneralName // This name should be PascalCased
    public override Tensor forward(Tensor input)
#pragma warning restore MSML_GeneralName // This name should be PascalCased
    {
        // needs higher precision for the norm so convert to float32
        // (B, Seq_Len, Dim)
        var normed = this.Norm(input.to_type(ScalarType.Float32)).type_as(input);
        // (B, Seq_Len, Dim) * (Dim) = (B, Seq_Len, Dim)
        var output = this.weight * normed;

        return output;
    }
}
