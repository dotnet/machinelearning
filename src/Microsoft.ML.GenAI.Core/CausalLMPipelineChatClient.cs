// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.AI;
using Microsoft.ML.Tokenizers;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Core;

public abstract class CausalLMPipelineChatClient<TTokenizer, TCausalLMModel> : IChatClient
    where TTokenizer : Tokenizer
    where TCausalLMModel : nn.Module<CausalLMModelInput, CausalLMModelOutput>
{
    private readonly ICausalLMPipeline<TTokenizer, TCausalLMModel> _pipeline;
    private readonly IMEAIChatTemplateBuilder _chatTemplateBuilder;
    private readonly ChatClientMetadata _metadata;

    public CausalLMPipelineChatClient(
        ICausalLMPipeline<TTokenizer, TCausalLMModel> pipeline,
        IMEAIChatTemplateBuilder chatTemplateBuilder,
        ChatClientMetadata? metadata = null)
    {
        var classNameWithType = $"{nameof(CausalLMPipelineChatClient<TTokenizer, TCausalLMModel>)}<{typeof(TTokenizer).Name}, {typeof(TCausalLMModel).Name}>";
        _metadata = new ChatClientMetadata(providerName: classNameWithType, defaultModelId: typeof(TCausalLMModel).Name);
        _chatTemplateBuilder = chatTemplateBuilder;
        _pipeline = pipeline;
    }

    public virtual Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        var prompt = _chatTemplateBuilder.BuildPrompt(messages, options);
        var stopSequences = options?.StopSequences ?? Array.Empty<string>();

        var output = _pipeline.Generate(
            prompt,
            maxLen: options?.MaxOutputTokens ?? 1024,
            temperature: options?.Temperature ?? 0.7f,
            stopSequences: stopSequences.ToArray()) ?? throw new InvalidOperationException("Failed to generate a reply.");

        var chatMessage = new ChatMessage(ChatRole.Assistant, output);
        return Task.FromResult(new ChatResponse([chatMessage])
        {
            CreatedAt = DateTime.UtcNow,
            FinishReason = ChatFinishReason.Stop,
            ResponseId = Guid.NewGuid().ToString("N"),
        });
    }

#pragma warning disable CS1998 // Async method lacks 'await' operators and will run synchronously
    public virtual async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
#pragma warning restore CS1998 // Async method lacks 'await' operators and will run synchronously
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var prompt = _chatTemplateBuilder.BuildPrompt(messages, options);
        var stopSequences = options?.StopSequences ?? Array.Empty<string>();

        string responseId = Guid.NewGuid().ToString("N");
        foreach (var output in _pipeline.GenerateStreaming(
            prompt,
            maxLen: options?.MaxOutputTokens ?? 1024,
            temperature: options?.Temperature ?? 0.7f,
            stopSequences: stopSequences.ToArray()))
        {
            yield return new(ChatRole.Assistant, output)
            {
                CreatedAt = DateTime.UtcNow,
                ResponseId = responseId,
            };
        }
    }

    public virtual void Dispose()
    {
    }

    public virtual object? GetService(Type serviceType, object? serviceKey = null) =>
        serviceKey is not null ? null :
        serviceType == typeof(ChatClientMetadata) ? _metadata :
        serviceType.IsAssignableFrom(GetType()) ? this :
        null;
}
