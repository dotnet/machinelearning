// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Core;

public class CasualLMModelOutput
{
    public CasualLMModelOutput(
        Tensor lastHiddenState,
        Tensor logits,
        Tensor[]? allHiddenStates = null,
        Tensor[]? attentions = null,
        IKVCache? cache = null)
    {
        this.LastHiddenState = lastHiddenState;
        this.AllHiddenStates = allHiddenStates;
        this.Logits = logits;
        this.Attentions = attentions;
        this.Cache = cache;
    }

    public Tensor Logits { get; set; }

    public Tensor LastHiddenState { get; set; }

    public Tensor[]? AllHiddenStates { get; set; }

    public Tensor[]? Attentions { get; set; }

    public IKVCache? Cache { get; set; }
}
