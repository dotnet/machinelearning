// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.GenAI.Core;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Mistral.Module;

internal class DecoderLayerInput
{
    public DecoderLayerInput(
        Tensor hiddenStates,
        Tensor attentionMask,
        Tensor positionIds,
        RotaryEmbeddingOutput positionEmbeddings, // cos, sin
        IKVCache? pastKeyValue = null,
        bool outputAttentions = false)
    {
        this.HiddenStates = hiddenStates;
        this.AttentionMask = attentionMask;
        this.PositionIds = positionIds;
        this.PastKeyValue = pastKeyValue;
        this.OutputAttentions = outputAttentions;
        this.PositionalEmbeddings = positionEmbeddings;
    }

    public Tensor HiddenStates { get; set; }

    public Tensor AttentionMask { get; set; }

    public Tensor PositionIds { get; set; }

    public RotaryEmbeddingOutput PositionalEmbeddings { get; set; }

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
internal class MistralDecoderLayer : nn.Module<DecoderLayerInput, DecoderLayerOutput>, IDynamicLoadModule
{
    private readonly MistralConfig _llamaConfig;
    private readonly int _layerIndex;
    private readonly int _hiddenSize;

#pragma warning disable MSML_PrivateFieldName // Private field name not in: _camelCase format
    private readonly MistralMLP mlp;
    private readonly Core.RMSNorm input_layernorm;
    private readonly Core.RMSNorm post_attention_layernorm;
    private readonly Attention self_attn;

    public Action<nn.Module>? LoadToDeviceFunc { get; set; }
    public Action<nn.Module>? UnloadFromDeviceFunc { get; set; }

#pragma warning restore MSML_PrivateFieldName // Private field name not in: _camelCase format

    public MistralDecoderLayer(MistralConfig config, int layerIndex)
        : base(nameof(MistralDecoderLayer))
    {
        _llamaConfig = config;
        _layerIndex = layerIndex;
        _hiddenSize = config.HiddenSize;

        this.self_attn = CreateAttention(config, layerIndex);
        this.mlp = new MistralMLP(config);
        this.input_layernorm = new Core.RMSNorm(this._hiddenSize, eps: config.RmsNormEps, config.DType);
        this.post_attention_layernorm = new Core.RMSNorm(this._hiddenSize, eps: config.RmsNormEps, config.DType);
    }

    private Attention CreateAttention(MistralConfig config, int layerIndex)
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
            attentionBias: config.AttentionBias);
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
            positionalEmbeddings: input.PositionalEmbeddings,
            outputAttentions: input.OutputAttentions);

        var selfAttnOutput = this.self_attn.forward(selfAttnInput);

        hiddenStates = residual + selfAttnOutput.HiddenStates;

        // Fully connected
        residual = hiddenStates;
        hiddenStates = this.post_attention_layernorm.forward(hiddenStates);
        hiddenStates = this.mlp.forward(hiddenStates);
        hiddenStates = residual + hiddenStates;

        if (UnloadFromDeviceFunc != null)
        {
            UnloadFromDeviceFunc(this);
        }

        return new DecoderLayerOutput(
            hiddenStates: hiddenStates.MoveToOuterDisposeScope(),
            attentions: input.OutputAttentions ? selfAttnOutput.Attentions?.MoveToOuterDisposeScope() : null,
            pastKeyValue: selfAttnOutput.Cache);
    }
}
