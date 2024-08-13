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

internal class Phi3Attention
{
    public static Attention FromConfig(Phi3Config config, int layerIdx)
    {
        var headDim = config.HiddenSize / config.NumAttentionHeads;
        return new Attention(
            attentionDropout: config.AttentionDropout,
            hiddenSize: config.HiddenSize,
            numHeads: config.NumAttentionHeads,
            headDim: headDim,
            numKeyValueHeads: config.NumKeyValueHeads ?? throw new ArgumentException("num_key_value_heads must be specified"),
            numKeyValueGroups: config.NumAttentionHeads / config.NumKeyValueHeads ?? throw new ArgumentException("num_key_value_heads must be specified"),
            maxPositionEmbeddings: config.MaxPositionEmbeddings,
            originalMaxPositionEmbeddings: config.OriginalMaxPositionEmbeddings,
            layerIdx: layerIdx,
            useQkvProj: true,
            dtype: config.DType,
            rotaryEmbedding: config.RopeScaling switch
            {
                null => new RotaryEmbedding(config.RopeTheta, config.MaxPositionEmbeddings, headDim),
                _ => new Phi3SuScaledRotaryEmbedding(headDim, config),
            });
    }
}
