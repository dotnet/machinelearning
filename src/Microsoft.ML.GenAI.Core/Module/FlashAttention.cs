// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using TorchSharp;
using TorchSharp.FlashAttention;
using Microsoft.ML.GenAI.Core.Extension;

namespace Microsoft.ML.GenAI.Core.Module;

internal class FlashAttention : nn.Module<AttentionInput, AttentionOutput>
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
#pragma warning restore MSML_PrivateFieldName // Private field name not in: _camelCase format

    public FlashAttention(
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
            (queryStates, keyStates) = Utils.ApplyRotaryPosEmb(queryStates, keyStates, input.PositionalEmbeddings.Cos, input.PositionalEmbeddings.Sin);

            if (pastKeyValue is not null)
            {
                (keyStates, valueStates) = pastKeyValue.UpdateKVCache(keyStates, valueStates, this._layerIdx);
            }


            // invoke flash attention api
            // todo: use varlen api to support batch > 1
            float dropoutRate = this.training ? (float)this._attentionDropout : 0.0f;

            // print q, k, v shape
            //Console.WriteLine($"queryStates shape: {queryStates.Peek("query")}");
            //Console.WriteLine($"keyStates shape: {keyStates.Peek("key")}");
            //Console.WriteLine($"valueStates shape: {valueStates.Peek("value")}");

            // shape of queryStates, keyStates, valueStates is [batch, nHead, sq_len, hidden_size / nHead]
            // transpose to [batch, sq_len, nHead, hidden_size / nHead]
            var output = FlashAttentionInterface.flash_attn_func(
                queryStates.transpose(1, 2),
                keyStates.transpose(1, 2),
                valueStates.transpose(1, 2),
                dropout_p: dropoutRate,
                causal: true);

            // shape of output.@out is [batch, nHead, sq_len, hidden_size / nHead]
            // convert to [batch, sq_len, hidden_size]

            output.@out = output.@out.reshape(bsz, qLen, -1).contiguous();
            var attnOutput = this.o_proj.forward(output.@out);

            return new(attnOutput.MoveToOuterDisposeScope(), outputAttentions ? output.@out.MoveToOuterDisposeScope() : null, pastKeyValue);
        }
    }
}
