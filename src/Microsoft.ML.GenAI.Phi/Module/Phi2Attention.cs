// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Diagnostics.Contracts;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Phi.Module;

internal class Phi2Attention : nn.Module<
    Tensor, // hidden_states
    Tensor, // position_ids
    Tensor?, // attention_mask
    int, // past_key_value_length
    bool, // output_attentions
    (
        Tensor, // hidden_states,
        Tensor?, // attentions,
        Tensor? // present_key_value
    )>
{
    private readonly int? _layerIdx;
    private readonly Phi2Config _config;
    private readonly double _attentionDropout;
    private readonly int _hiddenSize;
    private readonly int _numAttentionHeads;
    private readonly int _headDim;
    private readonly int _numKeyValueHeads;
    private readonly int _numKeyValueGroups;
    private readonly int _maxPositionEmbeddings;
    private readonly double _ropeTheta;
    private readonly double _partialRotaryFactor;
    private readonly bool _qkLayernorm;

    // we disable the warning for the private field name not in _camelCase format for all submodules fields
    // because their name will be used as keys to load the corresponding weights from the checkpoint
#pragma warning disable MSML_PrivateFieldName // Private field name not in: _camelCase format
    private readonly GenAILinear q_proj;
    private readonly GenAILinear k_proj;
    private readonly GenAILinear v_proj;
    private readonly GenAILinear dense;
    private readonly LayerNorm? q_layernorm;
    private readonly LayerNorm? k_layernorm;

    private readonly Phi2RotaryEmbedding phiRotaryEmbedding;

    // cache_k, cache_v
    private Tensor cache_k;
    private Tensor cache_v;
#pragma warning restore MSML_PrivateFieldName // Private field name not in: _camelCase format

    public Phi2Attention(Phi2Config config, int? layerIdx = null, int maxBatch = 2, int maxLength = 1024)
        : base(nameof(Phi2Attention))
    {
        this._layerIdx = layerIdx;
        this._config = config;
        this._attentionDropout = config.AttentionDropout;
        this._hiddenSize = config.HiddenSize;
        this._numAttentionHeads = config.NumAttentionHeads;
        this._headDim = this._hiddenSize / this._numAttentionHeads;
        this._numKeyValueHeads = config.NumKeyValueHeads ?? throw new ArgumentException("num_key_value_heads must be specified");
        this._numKeyValueGroups = this._numAttentionHeads / this._numKeyValueHeads;
        this._maxPositionEmbeddings = config.MaxPositionEmbeddings;
        this._ropeTheta = config.RopeTheta;
        this._partialRotaryFactor = config.PartialRotaryFactor;

        Contract.Assert(this._hiddenSize % (this._headDim * this._numAttentionHeads) == 0, "hidden_size must be divisible by num_attention_heads");
        this.q_proj = new GenAILinear(this._hiddenSize, this._numAttentionHeads * this._headDim, hasBias: true, dtype: config.Dtype);
        this.k_proj = new GenAILinear(this._hiddenSize, this._numKeyValueHeads * this._headDim, hasBias: true, dtype: config.Dtype);
        this.v_proj = new GenAILinear(this._hiddenSize, this._numKeyValueHeads * this._headDim, hasBias: true, dtype: config.Dtype);
        this.dense = new GenAILinear(this._numAttentionHeads * this._headDim, this._hiddenSize, hasBias: true, dtype: config.Dtype);

        this._qkLayernorm = config.QkLayernorm;
        if (this._qkLayernorm)
        {
            this.q_layernorm = nn.LayerNorm(this._hiddenSize / this._numAttentionHeads, eps: config.LayerNormEps, elementwise_affine: true, dtype: config.Dtype);
            this.k_layernorm = nn.LayerNorm(this._hiddenSize / this._numAttentionHeads, eps: config.LayerNormEps, elementwise_affine: true, dtype: config.Dtype);
        }

        this.RegisterComponents();
        this.phiRotaryEmbedding = new Phi2RotaryEmbedding(
            dim: (int)(this._partialRotaryFactor * this._headDim),
            maxPositionEmbeddings: this._maxPositionEmbeddings,
            baseValue: this._config.RopeTheta);
        this.cache_k = torch.zeros(maxBatch, this._numKeyValueHeads, maxLength, this._headDim, dtype: config.Dtype);
        this.cache_v = torch.zeros(maxBatch, this._numKeyValueHeads, maxLength, this._headDim, dtype: config.Dtype);
    }

#pragma warning disable MSML_GeneralName // This name should be PascalCased
    public override (Tensor, Tensor?, Tensor?) forward(
#pragma warning restore MSML_GeneralName // This name should be PascalCased
        Tensor hiddenStates,
        Tensor positionIds,
        Tensor? attentionMask = null,
        int pastKeyValueLength = 0,
        bool outputAttentions = false)
    {
        // move cache to the same device as hiddenStates
        if (this.cache_k.device != hiddenStates.device)
        {
            this.cache_k = this.cache_k.to(hiddenStates.device, disposeAfter: true).DetachFromDisposeScope();
            this.cache_v = this.cache_v.to(hiddenStates.device, disposeAfter: true).DetachFromDisposeScope();
        }

        using var disposeScope = torch.NewDisposeScope();
        var batchSize = (int)hiddenStates.shape[0];
        var seqLen = (int)hiddenStates.shape[1];

        var queryStates = this.q_proj.forward(hiddenStates);
        var keyStates = this.k_proj.forward(hiddenStates);
        var valueStates = this.v_proj.forward(hiddenStates);
        if (this._qkLayernorm)
        {
            queryStates = this.q_layernorm!.forward(queryStates);
            keyStates = this.k_layernorm!.forward(keyStates);
        }

        queryStates = queryStates.view(batchSize, seqLen, this._numAttentionHeads, this._headDim).transpose_(1, 2);
        keyStates = keyStates.view(batchSize, seqLen, this._numKeyValueHeads, this._headDim).transpose_(1, 2);
        valueStates = valueStates.view(batchSize, seqLen, this._numKeyValueHeads, this._headDim).transpose_(1, 2);
        var kvSeqLen = pastKeyValueLength == 0 ? (int)keyStates.shape[2] : pastKeyValueLength + (int)keyStates.shape[2];
        (var cos, var sin) = this.phiRotaryEmbedding.forward(valueStates, kvSeqLen);
        // split the last dim of queryStates and keyStates into rotary and non-rotary parts
        // shape: [batch_size, num_heads, seq_len, head_dim]
        // queryRot: [batch_size, num_heads, seq_len, :head_dim * partial_rotary_factor]
        // queryPass: [batch_size, num_heads, seq_len, head_dim * partial_rotary_factor:]
        var keyRot = keyStates[.., .., .., ..this.phiRotaryEmbedding.Dim];
        var keyPass = keyStates[.., .., .., this.phiRotaryEmbedding.Dim..];
        var queryRot = queryStates[.., .., .., ..this.phiRotaryEmbedding.Dim];
        var queryPass = queryStates[.., .., .., this.phiRotaryEmbedding.Dim..];
        (var qRot, var kRot) = Utils.ApplyRotaryPosEmb(queryRot, keyRot, cos, sin, positionIds);

        queryStates = torch.cat([qRot, queryPass], dim: -1);
        // update cache
        keyStates = torch.cat([kRot, keyPass], dim: -1);
        this.cache_k[..batchSize, .., pastKeyValueLength..kvSeqLen, ..] = keyStates;
        this.cache_v[..batchSize, .., pastKeyValueLength..kvSeqLen, ..] = valueStates;
        keyStates = this.cache_k[..batchSize, .., ..kvSeqLen, ..];
        valueStates = this.cache_v[..batchSize, .., ..kvSeqLen, ..];
        var keyStates2 = Utils.Phi2RepeatKV(keyStates, this._numKeyValueGroups).transpose(2, 3);
        var valueStates2 = Utils.Phi2RepeatKV(valueStates, this._numKeyValueGroups);
        // Queries and keys upcast to fp32 is required by Phi-2 to avoid overflow
        var attnWeights = torch.matmul(queryStates.to_type(float32), keyStates2.to_type(float32));
        attnWeights = attnWeights / Math.Sqrt(this._headDim);
        if (attentionMask is not null)
        {
            attnWeights = attnWeights + attentionMask;
        }
        attnWeights = nn.functional.softmax(attnWeights, dim: -1);
        attnWeights = nn.functional.dropout(attnWeights, p: this._attentionDropout);
        var attnOutput = torch.matmul(attnWeights, valueStates2.to_type(float32)).to_type(hiddenStates.dtype);
        attnOutput = attnOutput.transpose_(1, 2).contiguous();
        attnOutput = attnOutput.reshape(batchSize, seqLen, this._hiddenSize);
        var result = this.dense.forward(attnOutput);
        return (result.MoveToOuterDisposeScope(), null, null);
    }
}
