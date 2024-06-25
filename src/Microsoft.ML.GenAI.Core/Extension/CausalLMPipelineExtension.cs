// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using TorchSharp;

namespace Microsoft.ML.GenAI.Core.Extension;

public static class CausalLMPipelineExtension
{
    public static string? Generate(
        this CausalLMPipeline pipeline,
        string prompt,
        int maxLen = 128,
        float temperature = 0.7f,
        float topP = 0.9f,
        string[]? stopSequences = null,
        int eosId = 0,
        string device = "cpu",
        bool bos = true,
        bool eos = false,
        bool echo = false)
    {
        using var newScope = NewDisposeScope();
        var inputIds = pipeline.Tokenizer.EncodeToIds(prompt);
        var inputTensor = torch.tensor(inputIds.ToArray(), dtype: ScalarType.Int64, device: device).unsqueeze(0);
        var attentionMask = torch.ones_like(inputTensor);

        // set up stop token ids
        // stop token ids: [[eosId], [stopSequence1], [stopSequence2], ...]
        // when causal language model generates tokens, it will stop when it generates any token in stopSequences
        List<int[]> stopTokenIds = [[eosId]];
        if (stopSequences != null)
        {
            stopTokenIds.AddRange(stopSequences.Select(x => pipeline.Tokenizer.EncodeToIds(x).ToArray()));
        }

        (var token, var _) = pipeline.Generate(inputTensor, attentionMask, temperature: temperature, maxLen: maxLen, topP: topP, stopTokenSequence: stopTokenIds.ToArray(), echo: echo);

        var tokenIds = token[0].to_type(ScalarType.Int32).data<int>().ToArray();

        return pipeline.Tokenizer.Decode(tokenIds);
    }
}
