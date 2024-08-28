// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Runtime.CompilerServices;
using Microsoft.ML.GenAI.Core;
using Microsoft.ML.Tokenizers;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

namespace Microsoft.ML.GenAI.LLaMA;

public class LlamaChatCompletionService : IChatCompletionService
{
    private readonly ICausalLMPipeline<Tokenizer, LlamaForCausalLM> _pipeline;
    private readonly LlamaTextCompletionService _textGenerationService;
    private readonly ISemanticKernelChatTemplateBuilder _templateBuilder;

    /// <summary>
    /// Create a new instance of <see cref="LlamaChatCompletionService"/>.
    /// </summary>
    /// <param name="pipeline">pipeline</param>
    /// <param name="templateBuilder">The template builder to use for generating chat prompts, if not provided, <see cref="Llama3_1ChatTemplateBuilder.Instance"/> will be used.</param>
    public LlamaChatCompletionService(ICausalLMPipeline<Tokenizer, LlamaForCausalLM> pipeline, ISemanticKernelChatTemplateBuilder? templateBuilder = null)
    {
        _pipeline = pipeline;
        _textGenerationService = new LlamaTextCompletionService(pipeline);
        _templateBuilder = templateBuilder ?? Llama3_1ChatTemplateBuilder.Instance;
    }

    public IReadOnlyDictionary<string, object?> Attributes => _textGenerationService.Attributes;

    public async Task<IReadOnlyList<ChatMessageContent>> GetChatMessageContentsAsync(ChatHistory chatHistory, PromptExecutionSettings? executionSettings = null, Kernel? kernel = null, CancellationToken cancellationToken = default)
    {
        var prompt = _templateBuilder.BuildPrompt(chatHistory);
        var replies = await _textGenerationService.GetTextContentsAsync(prompt, executionSettings, kernel, cancellationToken);

        return replies.Select(reply => new ChatMessageContent(AuthorRole.Assistant, reply.Text)).ToList();
    }

    public async IAsyncEnumerable<StreamingChatMessageContent> GetStreamingChatMessageContentsAsync(
        ChatHistory chatHistory,
        PromptExecutionSettings? executionSettings = null,
        Kernel? kernel = null,
        [EnumeratorCancellation]
        CancellationToken cancellationToken = default)
    {
        var prompt = _templateBuilder.BuildPrompt(chatHistory);

        await foreach (var reply in _textGenerationService.GetStreamingTextContentsAsync(prompt, executionSettings, kernel, cancellationToken))
        {
            yield return new StreamingChatMessageContent(AuthorRole.Assistant, reply.Text);
        }
    }
}
