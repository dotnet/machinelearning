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
using Microsoft.ML.GenAI.Core.Extension;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Core;

internal class AttentionInput
{
    public AttentionInput(
        Tensor hiddenStates,
        Tensor positionIds,
        Tensor? attentionMask = null,
        IKVCache? cache = null,
        (Tensor, Tensor)? positionalEmbeddings = null, // cos, sin
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

    public (Tensor, Tensor)? PositionalEmbeddings { get; set; }

    public IKVCache? Cache { get; set; }

    public bool OutputAttentions { get; set; }
}

internal class AttentionOutput
{
    public AttentionOutput(
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

internal class Attention : nn.Module<AttentionInput, AttentionOutput>
{
    private readonly int _layerIdx;
    private readonly double _attentionDropout;
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _numKeyValueHeads;
    private readonly int _numKeyValueGroups;
    private readonly int _maxPositionEmbeddings;
    private readonly int _originalMaxPositionEmbeddings;
#pragma warning disable MSML_PrivateFieldName // Private field name not in: _camelCase format
    private readonly QuantizedLinear o_proj;
    private readonly QuantizedLinear? qkv_proj;
    private readonly QuantizedLinear? q_proj;
    private readonly QuantizedLinear? k_proj;
    private readonly QuantizedLinear? v_proj;
    private readonly nn.Module<RotaryEmbeddingInput, RotaryEmbeddingOutput> rotary_emb;
#pragma warning restore MSML_PrivateFieldName // Private field name not in: _camelCase format

    public Attention(
        double attentionDropout,
        int hiddenSize,
        int numHeads,
        int headDim,
        int numKeyValueHeads,
        int numKeyValueGroups,
        int maxPositionEmbeddings,
        int originalMaxPositionEmbeddings,
        int layerIdx,
        ScalarType dtype,
        nn.Module<RotaryEmbeddingInput, RotaryEmbeddingOutput> rotaryEmbedding,
        bool attentionBias = false,
        bool useQkvProj = true)
        : base(nameof(Attention))
    {
        this._layerIdx = layerIdx;
        this._attentionDropout = attentionDropout;
        this._hiddenSize = hiddenSize;
        this._numHeads = numHeads;
        this._headDim = headDim;
        this._numKeyValueHeads = numKeyValueHeads;
        this._numKeyValueGroups = numKeyValueGroups;
        this._maxPositionEmbeddings = maxPositionEmbeddings;
        this._originalMaxPositionEmbeddings = originalMaxPositionEmbeddings;

        Contract.Assert(this._hiddenSize % (this._headDim * this._numHeads) == 0, "hidden_size must be divisible by num_heads");

        this.o_proj = new QuantizedLinear(this._hiddenSize, this._hiddenSize, hasBias: attentionBias, dtype: dtype);
        if (useQkvProj)
        {
            var opSize = this._numHeads * this._headDim + 2 * (this._numKeyValueHeads * this._headDim);
            this.qkv_proj = new QuantizedLinear(this._hiddenSize, opSize, hasBias: attentionBias, dtype: dtype);
        }
        else
        {
            this.q_proj = new QuantizedLinear(this._hiddenSize, this._numHeads * this._headDim, hasBias: attentionBias, dtype: dtype);
            this.k_proj = new QuantizedLinear(this._hiddenSize, this._numKeyValueHeads * this._headDim, hasBias: attentionBias, dtype: dtype);
            this.v_proj = new QuantizedLinear(this._hiddenSize, this._numKeyValueHeads * this._headDim, hasBias: attentionBias, dtype: dtype);
        }

        this.rotary_emb = rotaryEmbedding;
    }

#pragma warning disable MSML_GeneralName // This name should be PascalCased
    public override AttentionOutput forward(AttentionInput input)
#pragma warning restore MSML_GeneralName // This name should be PascalCased
    {
        using (var _ = NewDisposeScope())
        {
            var hiddenStates = input.HiddenStates;
            var positionIds = input.PositionIds;
            var outputAttentions = input.OutputAttentions;
            var bsz = hiddenStates.shape[0];
            var qLen = hiddenStates.shape[1];

            Tensor queryStates;
            Tensor keyStates;
            Tensor valueStates;

            if (this.qkv_proj is not null)
            {
                var qkv = this.qkv_proj.forward(hiddenStates);
                var queryPos = this._numHeads * this._headDim;
                queryStates = qkv[.., .., ..queryPos];
                keyStates = qkv[.., .., queryPos..(queryPos + this._numKeyValueHeads * this._headDim)];
                valueStates = qkv[.., .., (queryPos + this._numKeyValueHeads * this._headDim)..];
            }
            else if (this.q_proj is not null && this.k_proj is not null && this.v_proj is not null)
            {
                queryStates = this.q_proj.forward(hiddenStates);
                keyStates = this.k_proj.forward(hiddenStates);
                valueStates = this.v_proj.forward(hiddenStates);
            }
            else
            {
                throw new InvalidOperationException("Invalid state, either qkv_proj or q_proj, k_proj, v_proj should be initialized");
            }

            queryStates = queryStates.view(bsz, qLen, this._numHeads, this._headDim).transpose(1, 2);
            keyStates = keyStates.view(bsz, qLen, this._numKeyValueHeads, this._headDim).transpose(1, 2);
            valueStates = valueStates.view(bsz, qLen, this._numKeyValueHeads, this._headDim).transpose(1, 2);
            var kvSeqLen = keyStates.IntShape()[^2];
            var pastKeyValue = input.Cache;
            if (pastKeyValue is not null)
            {
                kvSeqLen += pastKeyValue.GetUsableLength(kvSeqLen, this._layerIdx);
            }

            if (input.PositionalEmbeddings is (Tensor cos, Tensor sin))
            {
                (queryStates, keyStates) = Utils.ApplyRotaryPosEmb(queryStates, keyStates, cos, sin);
            }
            else
            {
                throw new NotImplementedException("Positional embeddings are not implemented");
                //var embOutput = this.rotary_emb.forward(new RotaryEmbeddingInput(valueStates, positionIds, kvSeqLen));
                //(cos, sin) = (embOutput.Cos, embOutput.Sin);

                //(queryStates, keyStates) = Utils.ApplyRotaryPosEmb(queryStates, keyStates, cos, sin);
            }

            if (pastKeyValue is not null)
            {
                (keyStates, valueStates) = pastKeyValue.UpdateKVCache(keyStates, valueStates, this._layerIdx);
            }

            // repeat k/v heads if n_kv_heads < n_heads
            keyStates = Utils.RepeatKV(keyStates, this._numKeyValueGroups);
            valueStates = Utils.RepeatKV(valueStates, this._numKeyValueGroups);

            // to fp32 to avoid overflow
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
                //Contract.Assert(attentionMask.shape[3] == kvSeqLen);
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
