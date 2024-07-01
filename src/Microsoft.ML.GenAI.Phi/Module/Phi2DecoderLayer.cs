// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Phi.Module;
public class Phi2DecoderLayer : nn.Module<
    Tensor, // hidden_states
    Tensor, // position_ids
    Tensor?, // attention_mask
    int, // past_key_value_length
    bool, // use_cache
    bool, // output_attentions
    (
        Tensor, // hidden_states,
        Tensor?, // attentions,
        Tensor? // present_key_value
    )>
{
    private readonly int? _layerIdx;

#pragma warning disable MSML_PrivateFieldName // Private field name not in: _camelCase format
    private readonly Phi2Attention self_attn;
    private readonly Phi2MLP mlp;
    private readonly LayerNorm input_layernorm;
    private readonly Dropout resid_dropout;
#pragma warning restore MSML_PrivateFieldName // Private field name not in: _camelCase format

    public Phi2DecoderLayer(Phi2Config config, int? layerIdx = null)
        : base(nameof(Phi2DecoderLayer))
    {
        this._layerIdx = layerIdx;
        this.self_attn = new Phi2Attention(config, layerIdx);
        this.mlp = new Phi2MLP(config);
        this.input_layernorm = nn.LayerNorm(config.HiddenSize, eps: config.LayerNormEps, dtype: config.Dtype);
        this.resid_dropout = nn.Dropout(config.ResidPdrop);
    }

#pragma warning disable MSML_GeneralName // This name should be PascalCased
    public override (Tensor, Tensor?, Tensor?) forward(
#pragma warning restore MSML_GeneralName // This name should be PascalCased
        Tensor hiddenStates,
        Tensor positionIds,
        Tensor? attentionMask = null,
        int pastKeyValueLength = 0,
        bool useCache = false,
        bool outputAttentions = false)
    {
        using var disposiableScope = torch.NewDisposeScope();
        var residual = hiddenStates;
        hiddenStates = this.input_layernorm.forward(hiddenStates);
        (var attnOutput, var attnWeights, var presentKeyValue) = this.self_attn.forward(
            hiddenStates: hiddenStates,
            positionIds: positionIds,
            attentionMask: attentionMask,
            pastKeyValueLength: pastKeyValueLength,
            outputAttentions: outputAttentions);
        var feedForwardHiddenStates = this.mlp.forward(hiddenStates);
        hiddenStates = residual + feedForwardHiddenStates + attnOutput;

        return (hiddenStates.MoveToOuterDisposeScope(), null, null);
    }
}
