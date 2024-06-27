// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Runtime.CompilerServices;
using System.Text;
using Microsoft.ML.GenAI.Core;
using Microsoft.ML.Tokenizers;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.TextGeneration;

namespace Microsoft.ML.GenAI.Phi;

public class Phi3CausalLMChatCompletionService : IChatCompletionService
{
    private readonly ICausalLMPipeline<Tokenizer, Phi3ForCasualLM> _pipeline;
    private readonly Phi3CausalLMTextGenerationService _textGenerationService;
    private const char NewLine = '\n'; // has to be \n, \r\n will cause wanky result.

    public Phi3CausalLMChatCompletionService(ICausalLMPipeline<Tokenizer, Phi3ForCasualLM> pipeline)
    {
        _pipeline = pipeline;
        _textGenerationService = new Phi3CausalLMTextGenerationService(pipeline);
    }

    public IReadOnlyDictionary<string, object?> Attributes => _textGenerationService.Attributes;

    public async Task<IReadOnlyList<ChatMessageContent>> GetChatMessageContentsAsync(
        ChatHistory chatHistory,
        PromptExecutionSettings? executionSettings = null,
        Kernel? kernel = null,
        CancellationToken cancellationToken = default)
    {
        // build prompt from chat history
        var sb = new StringBuilder();

        foreach (var message in chatHistory)
        {
            foreach (var item in message.Items)
            {
                if (item is not TextContent textContent)
                {
                    throw new NotSupportedException($"Only text content is supported, but got {item.GetType().Name}");
                }

                var prompt = message.Role switch
                {
                    _ when message.Role == AuthorRole.System => $"<|system|>{NewLine}{textContent}<|end|>{NewLine}",
                    _ when message.Role == AuthorRole.User => $"<|user|>{NewLine}{textContent}<|end|>{NewLine}",
                    _ when message.Role == AuthorRole.Assistant => $"<|assistant|>{NewLine}{textContent}<|end|>{NewLine}",
                    _ => throw new NotSupportedException($"Unsupported role {message.Role}")
                };

                sb.Append(prompt);
            }
        }

        sb.Append("<|assistant|>");
        var reply = await _textGenerationService.GetTextContentAsync(sb.ToString(), executionSettings, kernel, cancellationToken);
        return [new ChatMessageContent(AuthorRole.Assistant, reply.Text)];
    }

    public async IAsyncEnumerable<StreamingChatMessageContent> GetStreamingChatMessageContentsAsync(
        ChatHistory chatHistory,
        PromptExecutionSettings? executionSettings = null,
        Kernel? kernel = null,
        [EnumeratorCancellation]
        CancellationToken cancellationToken = default)
    {
        // CausalLMPipeline doesn't support streaming output yet
        // here we simply implement this api using the synchronous version

        var response = await GetChatMessageContentsAsync(chatHistory, executionSettings, kernel, cancellationToken);

        foreach (var item in response)
        {
            yield return new StreamingChatMessageContent(item.Role, item.Content);
        }
    }
}
