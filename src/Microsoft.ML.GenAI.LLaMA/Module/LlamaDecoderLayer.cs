// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.GenAI.Core;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.LLaMA.Module;


internal class DecoderLayerInput
{
    public DecoderLayerInput(
        Tensor hiddenStates,
        Tensor attentionMask,
        Tensor positionIds,
        IKVCache? pastKeyValue = null,
        (Tensor, Tensor)? positionEmbeddings = null, // cos, sin
        bool outputAttentions = false)
    {
        this.HiddenStates = hiddenStates;
        this.AttentionMask = attentionMask;
        this.PositionIds = positionIds;
        this.PastKeyValue = pastKeyValue;
        this.OutputAttentions = outputAttentions;
    }

    public Tensor HiddenStates { get; set; }

    public Tensor AttentionMask { get; set; }

    public Tensor PositionIds { get; set; }

    public (Tensor, Tensor) PositionalEmbeddings { get; set; }

    public IKVCache? PastKeyValue { get; set; }

    public bool OutputAttentions { get; set; }
}

internal class DecoderLayerOutput
{
    public DecoderLayerOutput(
        Tensor hiddenStates,
        Tensor? attentions = null,
        IKVCache? pastKeyValue = null)
    {
        this.HiddenStates = hiddenStates;
        this.Attentions = attentions;
        this.PastKeyValue = pastKeyValue;
    }

    public Tensor HiddenStates { get; set; }

    public Tensor? Attentions { get; set; }

    public IKVCache? PastKeyValue { get; set; }
}
internal class LlamaDecoderLayer : nn.Module<DecoderLayerInput, DecoderLayerOutput>, IDynamicLoadModule
{
    private readonly LlamaConfig _llamaConfig;
    private readonly int _layerIndex;
    private readonly int _hiddenSize;

#pragma warning disable MSML_PrivateFieldName // Private field name not in: _camelCase format
    private readonly LlamaMLP mlp;
    private readonly Core.RMSNorm input_layernorm;
    private readonly Core.RMSNorm post_attention_layernorm;
    private readonly Attention self_attn;

    public Action<nn.Module>? LoadToDeviceFunc { get; set; }
    public Action<nn.Module>? UnloadFromDeviceFunc { get; set; }

#pragma warning restore MSML_PrivateFieldName // Private field name not in: _camelCase format

    public LlamaDecoderLayer(LlamaConfig config, int layerIndex)
        : base(nameof(LlamaDecoderLayer))
    {
        _llamaConfig = config;
        _layerIndex = layerIndex;
        _hiddenSize = config.HiddenSize;

        this.self_attn = CreateAttention(config, layerIndex);
        this.mlp = new LlamaMLP(config);
        this.input_layernorm = new Core.RMSNorm(this._hiddenSize, eps: config.RmsNormEps);
        this.post_attention_layernorm = new Core.RMSNorm(this._hiddenSize, eps: config.RmsNormEps);
    }

    private Attention CreateAttention(LlamaConfig config, int layerIndex)
    {
        var headDim = config.HiddenSize / config.NumAttentionHeads;
        return new Attention(
            attentionDropout: config.AttentionDropout,
            hiddenSize: config.HiddenSize,
            numHeads: config.NumAttentionHeads,
            headDim: headDim,
            numKeyValueHeads: config.NumKeyValueHeads,
            numKeyValueGroups: config.NumAttentionHeads / config.NumKeyValueHeads,
            maxPositionEmbeddings: config.MaxPositionEmbeddings,
            originalMaxPositionEmbeddings: config.MaxPositionEmbeddings,
            layerIdx: layerIndex,
            useQkvProj: false,
            dtype: config.DType,
            attentionBias: config.AttentionBias,
            rotaryEmbedding: config.RopeScaling switch
            {
                null => new RotaryEmbedding(config.RopeTheta, config.MaxPositionEmbeddings, headDim),
                _ => new RotaryEmbedding(config.RopeTheta, headDim, config.RopeScaling),
            });
    }

#pragma warning disable MSML_GeneralName // This name should be PascalCased
    public override DecoderLayerOutput forward(DecoderLayerInput input)
#pragma warning restore MSML_GeneralName // This name should be PascalCased
    {
        if (LoadToDeviceFunc != null)
        {
            LoadToDeviceFunc(this);
        }

        using var disposeScope = NewDisposeScope();
        var residual = input.HiddenStates;
        var hiddenStates = this.input_layernorm.forward(input.HiddenStates);

        var selfAttnInput = new AttentionInput(
            hiddenStates: hiddenStates,
            attentionMask: input.AttentionMask,
            positionIds: input.PositionIds,
            cache: input.PastKeyValue,
            outputAttentions: input.OutputAttentions);

        var selfAttnOutput = this.self_attn.forward(selfAttnInput);

        hiddenStates = residual + selfAttnOutput.HiddenStates;

        // Fully connected
        residual = hiddenStates;
        hiddenStates = this.post_attention_layernorm.forward(hiddenStates);
        hiddenStates = this.mlp.forward(hiddenStates);
        hiddenStates = residual + hiddenStates;

        return new DecoderLayerOutput(
            hiddenStates: hiddenStates,
            attentions: input.OutputAttentions ? selfAttnOutput.Attentions : null,
            pastKeyValue: selfAttnOutput.Cache);
    }
}
