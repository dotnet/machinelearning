// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Core;

public class CausalLMModelOutput
{
    internal static class Defaults
    {
        internal const Tensor? Logits = null;
        internal const Tensor[]? AllHiddenStates = null;
        internal const Tensor[]? Attentions = null;
        internal const IKVCache? Cache = null;
        internal const Tensor? Loss = null;
    }
    public CausalLMModelOutput(
        Tensor lastHiddenState,
        Tensor? logits = Defaults.Logits,
        Tensor[]? allHiddenStates = Defaults.AllHiddenStates,
        Tensor[]? attentions = Defaults.Attentions,
        IKVCache? cache = Defaults.Cache,
        Tensor? loss = Defaults.Loss)
    {
        this.LastHiddenState = lastHiddenState;
        this.AllHiddenStates = allHiddenStates;
        this.Logits = logits;
        this.Attentions = attentions;
        this.Cache = cache;
        this.Loss = loss;
    }

    /// <summary>
    /// Shape: [1,]
    /// Available when label is provided in the input.
    /// </summary>
    public Tensor? Loss { get; set; }

    public Tensor? Logits { get; set; }

    public Tensor LastHiddenState { get; set; }

    public Tensor[]? AllHiddenStates { get; set; }

    public Tensor[]? Attentions { get; set; }

    public IKVCache? Cache { get; set; }
}
