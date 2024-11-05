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
    private readonly ISemanticKernelChatTemplateBuilder _templateBuilder;

    public Phi3CausalLMChatCompletionService(
        ICausalLMPipeline<Tokenizer, Phi3ForCasualLM> pipeline,
        ISemanticKernelChatTemplateBuilder? templateBuilder = null)
    {
        _pipeline = pipeline;
        _textGenerationService = new Phi3CausalLMTextGenerationService(pipeline);
        _templateBuilder = templateBuilder ?? Phi3ChatTemplateBuilder.Instance;
    }

    public IReadOnlyDictionary<string, object?> Attributes => _textGenerationService.Attributes;

    public async Task<IReadOnlyList<ChatMessageContent>> GetChatMessageContentsAsync(
        ChatHistory chatHistory,
        PromptExecutionSettings? executionSettings = null,
        Kernel? kernel = null,
        CancellationToken cancellationToken = default)
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
