// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.GenAI.Core.Extension;
using Microsoft.ML.Tokenizers;
using TorchSharp;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Core;

public interface ICausalLMPipeline<out TTokenizer, out TModel> : ICausalLMPipeline
    where TTokenizer : Tokenizer
    where TModel : nn.Module<CasualLMModelInput, CasualLMModelOutput>
{
    TTokenizer Tokenizer { get; }

    TModel Model { get; }
}

public interface ICausalLMPipeline
{
    string Generate(
        string prompt,
        int maxLen = CausalLMPipeline.Defaults.MaxLen,
        float temperature = CausalLMPipeline.Defaults.Temperature,
        float topP = CausalLMPipeline.Defaults.TopP,
        string[]? stopSequences = CausalLMPipeline.Defaults.StopSequence);

    IEnumerable<string> GenerateStreaming(
        string prompt,
        int maxLen = CausalLMPipeline.Defaults.MaxLen,
        float temperature = CausalLMPipeline.Defaults.Temperature,
        float topP = CausalLMPipeline.Defaults.TopP,
        string[]? stopSequences = CausalLMPipeline.Defaults.StopSequence);

    (Tensor, Tensor) Generate(
        Tensor inputIds,
        Tensor attentionMask,
        int[][] stopTokenSequence,
        float temperature = CausalLMPipeline.Defaults.Temperature,
        float topP = CausalLMPipeline.Defaults.TopP,
        int maxLen = CausalLMPipeline.Defaults.MaxLen);

    IEnumerable<(Tensor, Tensor)> GenerateStreaming(
        Tensor inputIds,
        Tensor attentionMask,
        int[][] stopTokenSequence,
        float temperature = CausalLMPipeline.Defaults.Temperature,
        float topP = CausalLMPipeline.Defaults.TopP,
        int maxLen = CausalLMPipeline.Defaults.MaxLen);
}

public class CausalLMPipeline<TTokenizer, TModel> : CausalLMPipeline, ICausalLMPipeline<TTokenizer, TModel>
    where TTokenizer : Tokenizer
    where TModel : nn.Module<CasualLMModelInput, CasualLMModelOutput>
{
    public CausalLMPipeline(
        TTokenizer tokenizer,
        TModel model,
        string device = Defaults.Device)
        : base(tokenizer, model, device)
    {
    }

    public new TTokenizer Tokenizer { get => (TTokenizer)base.Tokenizer; }

    public new TModel Model { get => (TModel)base.Model; }
}

public class CausalLMPipeline : ICausalLMPipeline
{
    internal static class Defaults
    {
        internal const string Device = "cpu";
        internal const float Temperature = 0.7F;
        internal const float TopP = 0.9F;
        internal const int MaxLen = 128;
        internal const string[]? StopSequence = null;
    }

    public CausalLMPipeline(
        Tokenizer tokenizer,
        nn.Module<CasualLMModelInput, CasualLMModelOutput> model,
        string device = Defaults.Device)
    {
        this.Tokenizer = tokenizer;
        this.Model = model;
        this.Device = device;
    }

    /// <summary>
    /// For moq purpose
    /// </summary>
    protected private CausalLMPipeline()
    {
        this.Tokenizer = default!;
        this.Model = default!;
        this.Device = default!;
    }

    public Tokenizer Tokenizer { get; }

    public nn.Module<CasualLMModelInput, CasualLMModelOutput> Model { get; }

    public Device Device { get; }

    public IEnumerable<(
        Tensor, // output token ids [batch_size, 1]
        Tensor  // output logits [batch_size, 1, vocab_size]
    )> GenerateStreaming(
        Tensor inputIds,
        Tensor attentionMask,
        int[][] stopTokenSequence,
        float temperature = Defaults.Temperature,
        float topP = Defaults.TopP,
        int maxLen = Defaults.MaxLen)
    {
        using var scope = NewDisposeScope();
        using var noGrad = torch.no_grad();
        var batch = inputIds.shape[0];
        var device = inputIds.device;
        var promptLength = (int)inputIds.shape[1];
        var totalLen = promptLength + maxLen;

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
            logits = output.Logits?.MoveToOtherDisposeScope(inputIds) ?? throw new InvalidOperationException("Logits is null");
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
            inputIds = torch.cat([inputIds, nextToken.unsqueeze(1)], dim: -1).MoveToOtherDisposeScope(inputIds);
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

            yield return (nextToken.MoveToOuterDisposeScope(), logits[.., ^1].MoveToOuterDisposeScope());
            prevPos = curPos;
        }
    }

    public virtual (
        Tensor, // output token ids [batch_size, sequence_length]
        Tensor // output logits [batch_size, sequence_length, vocab_size]
    ) Generate(
        Tensor inputIds, // input token ids [batch_size, sequence_length]
        Tensor attentionMask, // attention mask [batch_size, sequence_length]
        int[][] stopTokenSequence,
        float temperature = 0.7f,
        float topP = 0.9f,
        int maxLen = 128)
    {
        using var scope = NewDisposeScope();
        Tensor? logits = null;
        foreach (var (token, _logits) in this.GenerateStreaming(inputIds, attentionMask, stopTokenSequence, temperature, topP, maxLen))
        {
            inputIds = torch.cat([inputIds, token.unsqueeze(1)], dim: -1).MoveToOtherDisposeScope(inputIds);
            if (logits is null)
            {
                logits = _logits;
            }
            else
            {
                logits = torch.cat([logits, _logits], dim: -1).MoveToOtherDisposeScope(inputIds);
            }
        }

        return (inputIds, logits ?? throw new InvalidOperationException("Logits is null"));
    }

    public virtual string Generate(
        string prompt,
        int maxLen = 128,
        float temperature = 0.7f,
        float topP = 0.9f,
        string[]? stopSequences = null)
    {
        var chunks = new List<string>();

        foreach (var chunk in this.GenerateStreaming(prompt, maxLen, temperature, topP, stopSequences))
        {
            chunks.Add(chunk);
        }

        return string.Join(string.Empty, chunks);
    }


    public virtual IEnumerable<string> GenerateStreaming(
        string prompt,
        int maxLen = 128,
        float temperature = 0.7F,
        float topP = 0.9F,
        string[]? stopSequences = Defaults.StopSequence)
    {
        using var newScope = NewDisposeScope();
        var inputIds = this.Tokenizer.EncodeToIds(prompt);
        var inputTensor = torch.tensor(inputIds.ToArray(), dtype: ScalarType.Int64, device: this.Device).unsqueeze(0);
        var attentionMask = torch.ones_like(inputTensor, device: this.Device);
        // set up stop token ids
        // stop token ids: [[eosId], [stopSequence1], [stopSequence2], ...]
        // when causal language model generates tokens, it will stop when it generates any token in stopSequences
        List<int[]> stopTokenIds = [[]];
        if (stopSequences != null)
        {
            stopTokenIds.AddRange(stopSequences.Select(x =>
            {
                var tokens = this.Tokenizer.EncodeToTokens(x, out var _, false, false);

                return tokens
                // Skip the first _ token automatically added by tokenizer
                .Where(t => t.Offset != (0, 0))
                .Select(t => t.Id)
                .ToArray();
            }));
        }

        stopTokenIds = stopTokenIds.Where(ids => ids.Count() > 0).ToList();

        foreach (var (token, _) in this.GenerateStreaming(inputTensor, attentionMask, stopTokenIds.ToArray(), temperature: temperature, maxLen: maxLen))
        {
            var tokenIds = token[0].to_type(ScalarType.Int32).data<int>().ToArray();
            var duplicateTokenString = this.Tokenizer.Decode(tokenIds.Concat(tokenIds)) ?? throw new InvalidOperationException("Failed to decode token ids");
            var tokenString = this.Tokenizer.Decode(tokenIds) ?? throw new InvalidOperationException("Failed to decode token ids");
            // replace the first occurrence of the token with the duplicate token
            tokenString = duplicateTokenString.Substring(tokenString.Length);

            yield return tokenString;
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
