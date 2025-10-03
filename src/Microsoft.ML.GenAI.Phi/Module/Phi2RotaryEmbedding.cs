// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using TorchSharp;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Phi.Module;

internal class Phi2RotaryEmbedding : nn.Module<
    Tensor, // input
    int, // seq_len
    (
        Tensor, // cos
        Tensor // sin
    )>
{
    private readonly double _base;
    private readonly int _maxPositionEmbeddings;
    private readonly int _dim;

    public Phi2RotaryEmbedding(double baseValue, int maxPositionEmbeddings, int dim)
        : base(nameof(Phi2RotaryEmbedding))
    {
        _base = baseValue;
        _maxPositionEmbeddings = maxPositionEmbeddings;
        _dim = dim;
        var thetaNumerator = torch.arange(0, _dim, 2, dtype: ScalarType.Int64).to(torch.float32);
        this.register_buffer("inv_freq", torch.pow(baseValue, -1.0f * (thetaNumerator / dim)), persistent: false);
    }

    public int Dim => _dim;

#pragma warning disable MSML_GeneralName // This name should be PascalCased
    public override (Tensor, Tensor) forward(Tensor x, int seqLen)
#pragma warning restore MSML_GeneralName // This name should be PascalCased
    {
        // TODO
        // can be calculated once and cached
        var invFreq = this.get_buffer("inv_freq").to(x.device);
        var t = torch.arange(seqLen, dtype: invFreq.dtype, device: invFreq.device);
        var freqs = torch.outer(t, invFreq).to(torch.float32);
        var emb = torch.cat([freqs, freqs], dim: -1);

        var cos = torch.cos(emb);
        var sin = torch.sin(emb);

        return (cos[..seqLen].to_type(x.dtype), sin[..seqLen].to_type(x.dtype));
    }
}
