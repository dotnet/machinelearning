// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using TorchSharp;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Phi.Module;
internal class Phi3RotaryEmbeddingInput
{
    public Phi3RotaryEmbeddingInput(Tensor input, Tensor positionIds, int? seqLen = null)
    {
        Input = input;
        PositionIds = positionIds;
        SeqLen = seqLen;
    }

    public Tensor Input { get; set; }

    public Tensor PositionIds { get; set; }

    public int? SeqLen { get; set; }
}

internal class Phi3RotaryEmbeddingOutput
{
    public Phi3RotaryEmbeddingOutput(Tensor cos, Tensor sin)
    {
        Cos = cos;
        Sin = sin;
    }

    public Tensor Cos { get; set; }

    public Tensor Sin { get; set; }
}


internal class Phi3RotaryEmbedding : nn.Module<
    Phi3RotaryEmbeddingInput,
    Phi3RotaryEmbeddingOutput>
{
    private readonly double _base;
    private readonly int _maxPositionEmbeddings;
    private readonly int _dim;

    public Phi3RotaryEmbedding(double baseValue, int maxPositionEmbeddings, int dim)
        : base(nameof(Phi3RotaryEmbedding))
    {
        _base = baseValue;
        _maxPositionEmbeddings = maxPositionEmbeddings;
        _dim = dim;
        var thetaNumerator = torch.arange(0, _dim, 2, dtype: ScalarType.Int64).to(torch.float32);
        this.register_buffer("inv_freq", torch.pow(baseValue, -1.0f * (thetaNumerator / dim)), persistent: false);
    }

    public int Dim => _dim;

#pragma warning disable MSML_GeneralName // This name should be PascalCased
    public override Phi3RotaryEmbeddingOutput forward(Phi3RotaryEmbeddingInput input)
#pragma warning restore MSML_GeneralName // This name should be PascalCased
    {
        var x = input.Input;
        var positionIds = input.PositionIds;
        var seqLen = input.SeqLen;
        // TODO
        // can be calculated once and cached
        var invFreq = this.get_buffer("inv_freq").to(x.device);
        var invFreqExpanded = invFreq.unsqueeze(0).unsqueeze(-1);
        invFreqExpanded = invFreqExpanded.expand(new long[] { positionIds.shape[0], -1, 1 });
        var positionIdsExpanded = positionIds.unsqueeze(1).to(torch.float32);
        var freqs = invFreqExpanded * positionIdsExpanded;
        freqs = freqs.transpose(1, 2);
        var emb = torch.cat([freqs, freqs], dim: -1);

        var cos = torch.cos(emb);
        var sin = torch.sin(emb);

        return new(cos.to_type(x.dtype), sin.to_type(x.dtype));
    }
}
