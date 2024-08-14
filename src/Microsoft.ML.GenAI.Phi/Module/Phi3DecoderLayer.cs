// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.GenAI.Core;
using Microsoft.ML.GenAI.Core.Extension;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Phi.Module;

internal class Phi3DecoderLayerInput
{
    public Phi3DecoderLayerInput(
        Tensor hiddenStates,
        Tensor attentionMask,
        Tensor positionIds,
        IKVCache? pastKeyValue = null,
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

    public IKVCache? PastKeyValue { get; set; }

    public bool OutputAttentions { get; set; }
}

internal class Phi3DecoderLayerOutput
{
    public Phi3DecoderLayerOutput(
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

internal class Phi3DecoderLayer : nn.Module<Phi3DecoderLayerInput, Phi3DecoderLayerOutput>, IDynamicLoadModule
{
    private readonly Phi3Config _config;
#pragma warning disable MSML_PrivateFieldName // Private field name not in: _camelCase format
    private readonly nn.Module<AttentionInput, AttentionOutput> self_attn;
    private readonly Phi3MLP mlp;
    private readonly RMSNorm input_layernorm;
    private readonly Dropout resid_attn_dropout;
    private readonly Dropout resid_mlp_dropout;
    private readonly RMSNorm post_attention_layernorm;
#pragma warning restore MSML_PrivateFieldName // Private field name not in: _camelCase format

    public Phi3DecoderLayer(Phi3Config config, int layerIdx)
        : base(nameof(Phi3DecoderLayer))
    {
        this._config = config;
        if (config.AttnImplementation == "eager")
        {
            this.self_attn = Phi3Attention.FromConfig(config, layerIdx);
        }
        else
        {
            throw new NotImplementedException();
        }

        this.mlp = new Phi3MLP(config);
        this.input_layernorm = new RMSNorm(config.HiddenSize, config.RmsNormEps, config.DType);

        this.resid_attn_dropout = nn.Dropout(config.ResidPdrop);
        this.resid_mlp_dropout = nn.Dropout(config.ResidPdrop);
        this.post_attention_layernorm = new RMSNorm(config.HiddenSize, config.RmsNormEps, config.DType);
    }

    public Action<nn.Module>? LoadToDeviceFunc { get; set; }

    public Action<nn.Module>? UnloadFromDeviceFunc { get; set; }

#pragma warning disable MSML_GeneralName // This name should be PascalCased
    public override Phi3DecoderLayerOutput forward(Phi3DecoderLayerInput input)
#pragma warning restore MSML_GeneralName // This name should be PascalCased
    {
        if (LoadToDeviceFunc != null)
        {
            LoadToDeviceFunc(this);
        }
        using var disposeScope = NewDisposeScope();
        var hiddenStates = input.HiddenStates;
        var residual = input.HiddenStates;
        hiddenStates = this.input_layernorm.forward(hiddenStates);

        var attentionInput = new AttentionInput(hiddenStates, input.PositionIds, input.AttentionMask, input.PastKeyValue, input.OutputAttentions);
        var output = this.self_attn.forward(attentionInput);
        var attnOutputs = output.HiddenStates;
        var selfAttnWeights = output.Attentions;
        var presentKeyValue = output.Cache;
        hiddenStates = residual + this.resid_attn_dropout.forward(attnOutputs);
        residual = hiddenStates;
        hiddenStates = this.post_attention_layernorm.forward(hiddenStates);
        hiddenStates = this.mlp.forward(hiddenStates);
        hiddenStates = residual + this.resid_mlp_dropout.forward(hiddenStates);

        if (UnloadFromDeviceFunc != null)
        {
            UnloadFromDeviceFunc(this);
        }
        return new Phi3DecoderLayerOutput(hiddenStates.MoveToOuterDisposeScope(), selfAttnWeights?.MoveToOuterDisposeScope(), presentKeyValue);
    }
}
