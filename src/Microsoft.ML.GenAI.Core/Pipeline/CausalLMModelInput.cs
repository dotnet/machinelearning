// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Core;

public class CausalLMModelInput
{
    internal static class Defaults
    {
        internal const Tensor? AttentionMask = null;
        internal const Tensor? PositionIds = null;
        internal const int PastKeyValuesLength = 0;
        internal const Tensor? InputsEmbeds = null;
        internal const bool UseCache = true;
        internal const bool OutputAttentions = false;
        internal const bool OutputHiddenStates = false;
        internal const Tensor? Labels = null;
    }
    public CausalLMModelInput(
        Tensor inputIds,
        Tensor? attentionMask = Defaults.AttentionMask,
        Tensor? positionIds = Defaults.PositionIds,
        int pastKeyValuesLength = Defaults.PastKeyValuesLength,
        Tensor? inputsEmbeds = Defaults.InputsEmbeds,
        Tensor? labels = Defaults.Labels,
        bool useCache = Defaults.UseCache,
        bool outputAttentions = Defaults.OutputAttentions,
        bool outputHiddenStates = Defaults.OutputHiddenStates)
    {
        this.InputIds = inputIds;
        this.AttentionMask = attentionMask;
        this.PositionIds = positionIds;
        this.PastKeyValuesLength = pastKeyValuesLength;
        this.InputEmbeddings = inputsEmbeds;
        this.UseCache = useCache;
        this.OutputAttentions = outputAttentions;
        this.OutputHiddenStates = outputHiddenStates;
        this.Labels = labels;
    }

    public Tensor InputIds { get; set; }

    public Tensor? AttentionMask { get; set; }

    public Tensor? PositionIds { get; set; }

    public IKVCache? OverrideCache { get; set; }

    public int PastKeyValuesLength { get; set; }

    public Tensor? InputEmbeddings { get; set; }

    /// <summary>
    /// Shape: [batch_size, sequence_length]
    /// DTypes: int64
    /// Labels for computing the causal language modeling loss.
    /// Indices should be in [0, config.vocab_size - 1] or [-100] for padding/masking.
    /// </summary>
    public Tensor? Labels { get; set; }

    public bool UseCache { get; set; }

    public bool OutputAttentions { get; set; }

    public bool OutputHiddenStates { get; set; }
}
