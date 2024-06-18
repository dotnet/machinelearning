// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Tokenizers;
using TorchSharp;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Core;

public class CausalLMPipeline<TTokenizer, TModel> : CausalLMPipeline
    where TTokenizer : Tokenizer
    where TModel : nn.Module<CasualLMModelInput, CasualLMModelOutput>
{
    public CausalLMPipeline(
        TTokenizer tokenizer,
        TModel model,
        string device = "cpu")
        : base(tokenizer, model, device)
    {
    }
}

public class CausalLMPipeline
{
    public CausalLMPipeline(
        Tokenizer tokenizer,
        nn.Module<CasualLMModelInput, CasualLMModelOutput> model,
        string device = "cpu")
    {
        this.Tokenizer = tokenizer;
        this.Model = model;
        this.Device = device;
    }

    public Tokenizer Tokenizer { get; }

    public nn.Module<CasualLMModelInput, CasualLMModelOutput> Model { get; }

    public Device Device { get; }

    public virtual (
        Tensor, // output token ids [batch_size, sequence_length]
        Tensor // output logits [batch_size, sequence_length, vocab_size]
    ) Generate(
        Tensor inputIds, // input token ids [batch_size, sequence_length]
        Tensor attentionMask, // attention mask [batch_size, sequence_length]
        int[][] stopTokenSequence,
        float temperature = 0.7f,
        float topP = 0.9f,
        int maxLen = 128,
        bool echo = false)
    {
        using var newScope = NewDisposeScope();
        var batch = inputIds.shape[0];
        var device = inputIds.device;
        var promptLength = (int)inputIds.shape[1];
        var totalLen = promptLength + maxLen;

        using (var noGrad = torch.no_grad())
        {
            var prevPos = 0;
            var eosReached = torch.tensor(new bool[batch], device: device);
            torch.Tensor? logits = default;
            var cache = new DynamicKVCache();
            if (promptLength == totalLen)
            {
                var input = new CasualLMModelInput(inputIds, attentionMask, pastKeyValuesLength: 0)
                {
                    OverrideCache = cache,
                };
                var output = this.Model.forward(input);
                logits = output.Logits;
            }
            for (var curPos = promptLength; curPos != totalLen; curPos++)
            {
                var input = new CasualLMModelInput(inputIds[.., prevPos..curPos], attentionMask[.., prevPos..curPos], pastKeyValuesLength: prevPos)
                {
                    OverrideCache = cache,
                };
                var output = this.Model.forward(input);
                logits = output.Logits;
                torch.Tensor nextToken;
                if (temperature > 0)
                {
                    var probs = torch.softmax(logits[.., -1] / temperature, dim: -1);
                    nextToken = this.SampleTopP(probs, topP);
                }
                else
                {
                    nextToken = torch.argmax(logits[.., -1], dim: -1);
                }

                nextToken = nextToken.reshape(-1);
                inputIds = torch.cat([inputIds, nextToken.unsqueeze(1)], dim: -1);
                attentionMask = torch.cat([attentionMask, attentionMask.new_ones(attentionMask.shape[0], 1)], dim: -1);
                foreach (var stopSequence in stopTokenSequence)
                {
                    // determine if the last n tokens are the stop sequence
                    var lastN = inputIds[.., ^stopSequence.Length..];
                    var lastNMatch = lastN == torch.tensor(stopSequence, device: device);
                    eosReached |= lastNMatch.all(dim: -1);
                }
                if (eosReached.all().item<bool>())
                {
                    break;
                }

                // pBar.Tick(curPos, message);
                var nextTokenIds = nextToken.to_type(ScalarType.Int32).data<int>().ToArray();
                var nextTokenStr = this.Tokenizer.Decode(nextTokenIds);

                prevPos = curPos;
            }

            if (echo)
            {
                // return entire inputIds and logits
                return (inputIds.MoveToOuterDisposeScope(), logits!.MoveToOuterDisposeScope());
            }
            else
            {
                // return [batch_size, promptLength..] and [batch_size, promptLength.., vocab_size]
                return (inputIds[.., promptLength..].MoveToOuterDisposeScope(), logits![.., promptLength..].MoveToOuterDisposeScope());
            }
        }
    }

    protected torch.Tensor SampleTopP(torch.Tensor logits, float topP)
    {
        (var probsSort, var probsIndex) = torch.sort(logits, dim: -1, descending: true);
        var cumSum = torch.cumsum(probsSort, dim: -1);
        var mask = cumSum - probsSort > topP;
        probsSort[mask] = 0f;
        probsSort /= probsSort.sum(dim: -1, keepdim: true);
        var nextToken = torch.multinomial(probsSort, num_samples: 1);
        nextToken = torch.gather(probsIndex, dim: -1, index: nextToken);
        return nextToken;
    }
}
