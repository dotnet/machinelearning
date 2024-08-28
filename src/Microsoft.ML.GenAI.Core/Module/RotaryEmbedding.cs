// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Text.Json.Serialization;
using TorchSharp;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Core;

public class RopeScalingConfig
{
    public RopeScalingConfig()
    {
        this.Factor = 1.0f;
        this.LowFreqFactor = 1.0f;
        this.HighFreqFactor = 1.0f;
        this.OriginalMaxPositionEmbeddings = 8192;
        this.RopeType = "default";
    }

    [JsonPropertyName("factor")]
    public float Factor { get; set; }

    [JsonPropertyName("low_freq_factor")]
    public float LowFreqFactor { get; set; }

    [JsonPropertyName("high_freq_factor")]
    public float HighFreqFactor { get; set; }

    [JsonPropertyName("original_max_position_embeddings")]
    public int OriginalMaxPositionEmbeddings { get; set; }

    [JsonPropertyName("rope_type")]
    public string RopeType { get; set; }
}


internal class RotaryEmbeddingInput
{
    public RotaryEmbeddingInput(Tensor input, Tensor positionIds, int? seqLen = null)
    {
        Input = input;
        PositionIds = positionIds;
        SeqLen = seqLen;
    }

    public Tensor Input { get; set; }

    public Tensor PositionIds { get; set; }

    public int? SeqLen { get; set; }
}

internal class RotaryEmbeddingOutput
{
    public RotaryEmbeddingOutput(Tensor cos, Tensor sin)
    {
        Cos = cos;
        Sin = sin;
    }

    public Tensor Cos { get; set; }

    public Tensor Sin { get; set; }
}


internal class RotaryEmbedding : nn.Module<
    RotaryEmbeddingInput,
    RotaryEmbeddingOutput>
{
    private readonly double _base;
    private readonly int _maxPositionEmbeddings;
    private readonly int _dim;

    public RotaryEmbedding(double baseValue, int maxPositionEmbeddings, int dim)
        : this(baseValue, dim, new RopeScalingConfig() { RopeType = "default", OriginalMaxPositionEmbeddings = maxPositionEmbeddings })
    {
    }

    public RotaryEmbedding(double baseValue, int dim, RopeScalingConfig config)
        : base(nameof(RotaryEmbedding))
    {
        _base = baseValue;
        _maxPositionEmbeddings = config.OriginalMaxPositionEmbeddings;
        _dim = dim;

        if (config.RopeType == "default")
        {
            var thetaNumerator = torch.arange(0, _dim, 2, dtype: ScalarType.Int64).to(torch.float32);
            this.register_buffer("inv_freq", torch.pow(baseValue, -1.0f * (thetaNumerator / dim)), persistent: false);
        }
        else
        {
            throw new NotImplementedException("Rope type not implemented");
        }
    }

    public int Dim => _dim;

#pragma warning disable MSML_GeneralName // This name should be PascalCased
    public override RotaryEmbeddingOutput forward(RotaryEmbeddingInput input)
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
