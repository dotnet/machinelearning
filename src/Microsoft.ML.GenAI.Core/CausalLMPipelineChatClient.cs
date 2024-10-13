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
using Microsoft.ML.GenAI.Core;
using Microsoft.ML.Tokenizers;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Core;

public abstract class CausalLMPipelineChatClient<TTokenizer, TCausalLMModel> : IChatClient
    where TTokenizer : Tokenizer
    where TCausalLMModel : nn.Module<CausalLMModelInput, CausalLMModelOutput>
{
    private readonly ICausalLMPipeline<TTokenizer, TCausalLMModel> _pipeline;
    private readonly ChatClientMetadata _metadata;
    private readonly IMEAIChatTemplateBuilder _chatTemplateBuilder;

    public CausalLMPipelineChatClient(
        ICausalLMPipeline<TTokenizer, TCausalLMModel> pipeline,
        IMEAIChatTemplateBuilder chatTemplateBuilder,
        ChatClientMetadata? metadata = null)
    {
        metadata ??= new ChatClientMetadata(modelId: nameof(CausalLMPipelineChatClient<TTokenizer, TCausalLMModel>));
        _chatTemplateBuilder = chatTemplateBuilder;
        _pipeline = pipeline;
        _metadata = metadata;
    }

    public ChatClientMetadata Metadata => _metadata;



    public virtual Task<ChatCompletion> CompleteAsync(IList<ChatMessage> chatMessages, ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        var prompt = _chatTemplateBuilder.BuildPrompt(chatMessages, options);
        var stopSequences = options?.StopSequences ?? Array.Empty<string>();

        var output = _pipeline.Generate(
            prompt,
            maxLen: options?.MaxOutputTokens ?? 1024,
            temperature: options?.Temperature ?? 0.7f,
            stopSequences: stopSequences.ToArray()) ?? throw new InvalidOperationException("Failed to generate a reply.");

        var chatMessage = new ChatMessage(ChatRole.Assistant, output);
        return Task.FromResult(new ChatCompletion([chatMessage])
        {
            CreatedAt = DateTime.UtcNow,
            FinishReason = ChatFinishReason.Stop,
        });
    }

#pragma warning disable CS1998 // Async method lacks 'await' operators and will run synchronously
    public virtual async IAsyncEnumerable<StreamingChatCompletionUpdate> CompleteStreamingAsync(
#pragma warning restore CS1998 // Async method lacks 'await' operators and will run synchronously
        IList<ChatMessage> chatMessages,
        ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var prompt = _chatTemplateBuilder.BuildPrompt(chatMessages, options);
        var stopSequences = options?.StopSequences ?? Array.Empty<string>();

        foreach (var output in _pipeline.GenerateStreaming(
            prompt,
            maxLen: options?.MaxOutputTokens ?? 1024,
            temperature: options?.Temperature ?? 0.7f,
            stopSequences: stopSequences.ToArray()))
        {
            yield return new StreamingChatCompletionUpdate
            {
                Role = ChatRole.Assistant,
                Text = output,
                CreatedAt = DateTime.UtcNow,
            };
        }
    }

    public virtual void Dispose()
    {
    }

    public virtual TService? GetService<TService>(object? key = null) where TService : class
    {
        return null;
    }
}
