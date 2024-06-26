// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Core;

public class CasualLMModelInput
{
    public CasualLMModelInput(
        Tensor inputIds,
        Tensor? attentionMask = null,
        Tensor? positionIds = null,
        int pastKeyValuesLength = 0,
        Tensor? inputsEmbeds = null,
        bool useCache = false,
        bool outputAttentions = false,
        bool outputHiddenStates = false)
    {
        this.InputIds = inputIds;
        this.AttentionMask = attentionMask;
        this.PositionIds = positionIds;
        this.PastKeyValuesLength = pastKeyValuesLength;
        this.InputEmbeddings = inputsEmbeds;
        this.UseCache = useCache;
        this.OutputAttentions = outputAttentions;
        this.OutputHiddenStates = outputHiddenStates;
    }

    public Tensor InputIds { get; set; }

    public Tensor? AttentionMask { get; set; }

    public Tensor? PositionIds { get; set; }

    public IKVCache? OverrideCache { get; set; }

    public int PastKeyValuesLength { get; set; }

    public Tensor? InputEmbeddings { get; set; }

    public bool UseCache { get; set; }

    public bool OutputAttentions { get; set; }

    public bool OutputHiddenStates { get; set; }
}
