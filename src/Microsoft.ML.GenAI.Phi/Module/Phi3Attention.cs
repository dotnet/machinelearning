// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.GenAI.Core;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Phi.Module;

internal class Phi3AttentionInput
{
    public Phi3AttentionInput(
        Tensor hiddenStates,
        Tensor positionIds,
        Tensor? attentionMask = null,
        IKVCache? cache = null,
        bool outputAttentions = false)
    {
        this.HiddenStates = hiddenStates;
        this.AttentionMask = attentionMask;
        this.PositionIds = positionIds;
        this.Cache = cache;
        this.OutputAttentions = outputAttentions;
    }
    public Tensor HiddenStates { get; set; }

    public Tensor? AttentionMask { get; set; }

    public Tensor PositionIds { get; set; }

    public IKVCache? Cache { get; set; }

    public bool OutputAttentions { get; set; }
}

internal class Phi3AttentionOutput
{
    public Phi3AttentionOutput(
        Tensor hiddenStates,
        Tensor? attentions = null,
        IKVCache? cache = null)
    {
        this.HiddenStates = hiddenStates;
        this.Attentions = attentions;
        this.Cache = cache;
    }

    public Tensor HiddenStates { get; set; }

    public Tensor? Attentions { get; set; }

    public IKVCache? Cache { get; set; }
}

internal class Phi3Attention : nn.Module<Phi3AttentionInput, Phi3AttentionOutput>
{
    private readonly Phi3Config _config;
    private readonly int _layerIdx;
    private readonly double _attentionDropout;
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _numKeyValueHeads;
    private readonly int _numKeyValueGroups;
    private readonly int _maxPositionEmbeddings;
    private readonly int _originalMaxPositionEmbeddings;
    private readonly double _ropeTheta;
    private readonly Dictionary<string, object>? _ropeScaling;
#pragma warning disable MSML_PrivateFieldName // Private field name not in: _camelCase format
    private readonly QuantizedLinear o_proj;
    private readonly QuantizedLinear qkv_proj;
    private nn.Module<Phi3RotaryEmbeddingInput, Phi3RotaryEmbeddingOutput> rotary_emb = null!;
#pragma warning restore MSML_PrivateFieldName // Private field name not in: _camelCase format

    public Phi3Attention(Phi3Config config, int layerIdx)
        : base(nameof(Phi3Attention))
    {
        this._config = config;
        this._layerIdx = layerIdx;
        this._attentionDropout = config.AttentionDropout;
        this._hiddenSize = config.HiddenSize;
        this._numHeads = config.NumAttentionHeads;
        this._headDim = this._hiddenSize / this._numHeads;
        this._numKeyValueHeads = config.NumKeyValueHeads ?? throw new ArgumentException("num_key_value_heads must be specified");
        this._numKeyValueGroups = this._numHeads / this._numKeyValueHeads;
        this._maxPositionEmbeddings = config.MaxPositionEmbeddings;
        this._originalMaxPositionEmbeddings = config.OriginalMaxPositionEmbeddings;
        this._ropeTheta = config.RopeTheta;
        this._ropeScaling = config.RopeScaling;

        Contract.Assert(this._hiddenSize % (this._headDim * this._numHeads) == 0, "hidden_size must be divisible by num_heads");

        var opSize = this._numHeads * this._headDim + 2 * (this._numKeyValueHeads * this._headDim);
        this.o_proj = new QuantizedLinear(this._numHeads * this._headDim, this._hiddenSize, hasBias: false, dtype: config.DType);
        this.qkv_proj = new QuantizedLinear(this._hiddenSize, opSize, hasBias: false, dtype: config.DType);
        this.InitRope();
    }

    private void InitRope()
    {
        if (this._ropeScaling is null)
        {
            this.rotary_emb = new Phi3RotaryEmbedding(this._ropeTheta, this._maxPositionEmbeddings, this._headDim);
        }
        else
        {
            this.rotary_emb = new Phi3SuScaledRotaryEmbedding(this._headDim, this._config);
        }
    }

#pragma warning disable MSML_GeneralName // This name should be PascalCased
    public override Phi3AttentionOutput forward(Phi3AttentionInput input)
#pragma warning restore MSML_GeneralName // This name should be PascalCased
    {
        using (var _ = NewDisposeScope())
        {
            var hiddenStates = input.HiddenStates;
            var positionIds = input.PositionIds;
            var outputAttentions = input.OutputAttentions;
            var bsz = hiddenStates.shape[0];
            var qLen = hiddenStates.shape[1];

            var qkv = this.qkv_proj.forward(hiddenStates);
            var queryPos = this._numHeads * this._headDim;
            var queryStates = qkv[.., .., ..queryPos];
            var keyStates = qkv[.., .., queryPos..(queryPos + this._numKeyValueHeads * this._headDim)];
            var valueStates = qkv[.., .., (queryPos + this._numKeyValueHeads * this._headDim)..];
            queryStates = queryStates.view(bsz, qLen, this._numHeads, this._headDim).transpose(1, 2);
            keyStates = keyStates.view(bsz, qLen, this._numKeyValueHeads, this._headDim).transpose(1, 2);
            valueStates = valueStates.view(bsz, qLen, this._numKeyValueHeads, this._headDim).transpose(1, 2);

            var kvSeqLen = keyStates.IntShape()[^2];
            var pastKeyValue = input.Cache;
            if (pastKeyValue is not null)
            {
                kvSeqLen += pastKeyValue.GetUsableLength(kvSeqLen, this._layerIdx);
            }

            var embOutput = this.rotary_emb.forward(new Phi3RotaryEmbeddingInput(valueStates, positionIds, kvSeqLen));
            (var cos, var sin) = (embOutput.Cos, embOutput.Sin);

            (queryStates, keyStates) = Utils.ApplyRotaryPosEmb(queryStates, keyStates, cos, sin);

            if (pastKeyValue is not null)
            {
                (keyStates, valueStates) = pastKeyValue.UpdateKVCache(keyStates, valueStates, this._layerIdx);
            }

            // repeat k/v heads if n_kv_heads < n_heads
            keyStates = Utils.Phi3RepeatKV(keyStates, this._numKeyValueGroups);
            valueStates = Utils.Phi3RepeatKV(valueStates, this._numKeyValueGroups);

            var attnWeights = torch.matmul(queryStates, keyStates.transpose(2, 3));
            attnWeights = attnWeights / Math.Sqrt(this._headDim);

            // attnWeight's shape should be [bsz, this._numHeads, qLen, kvSeqLen]
            Contract.Assert(attnWeights.shape.Length == 4);
            Contract.Assert(attnWeights.shape[0] == bsz);
            Contract.Assert(attnWeights.shape[1] == this._numHeads);
            Contract.Assert(attnWeights.shape[2] == qLen);
            Contract.Assert(attnWeights.shape[3] == kvSeqLen);

            var attentionMask = input.AttentionMask;
            if (attentionMask is not null)
            {
                Contract.Assert(attentionMask.shape.Length == 4);
                Contract.Assert(attentionMask.shape[0] == bsz);
                Contract.Assert(attentionMask.shape[1] == 1);
                Contract.Assert(attentionMask.shape[2] == qLen);
                Contract.Assert(attentionMask.shape[3] == kvSeqLen);
                attnWeights = attnWeights + attentionMask;
            }

            // upscale attention to fp32 to avoid overflow
            attnWeights = nn.functional.softmax(attnWeights, dim: -1, dtype: ScalarType.Float32).to(valueStates.dtype);
            attnWeights = nn.functional.dropout(attnWeights, this._attentionDropout, this.training);

            var attnOutput = torch.matmul(attnWeights, valueStates);

            attnOutput = attnOutput.transpose(1, 2).contiguous();
            attnOutput = attnOutput.reshape(bsz, qLen, this._hiddenSize);

            attnOutput = this.o_proj.forward(attnOutput);

            return new(attnOutput.MoveToOuterDisposeScope(), outputAttentions ? attnWeights.MoveToOuterDisposeScope() : null, pastKeyValue);
        }
    }
}
