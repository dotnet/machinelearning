// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Runtime.CompilerServices;
using Microsoft.Extensions.AI;
using Microsoft.ML.GenAI.Core;
using Microsoft.ML.Tokenizers;

namespace Microsoft.ML.GenAI.LLaMA;

public class Llama3CausalLMChatClient : CausalLMPipelineChatClient<Tokenizer, LlamaForCausalLM>
{
    private readonly string _eotToken = "<|eot_id|>";

    public Llama3CausalLMChatClient(
        ICausalLMPipeline<Tokenizer, LlamaForCausalLM> pipeline,
        IMEAIChatTemplateBuilder? chatTemplateBuilder = null,
        ChatClientMetadata? metadata = null)
        : base(
            pipeline,
            chatTemplateBuilder ?? Llama3_1ChatTemplateBuilder.Instance,
            metadata ?? new ChatClientMetadata(defaultModelId: nameof(Llama3CausalLMChatClient)))
    {
    }

    public override Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new ChatOptions();

        if (options.StopSequences != null)
        {
            options.StopSequences.Add(_eotToken);
        }
        else
        {
            options.StopSequences = new List<string> { _eotToken };
        }

        return base.GetResponseAsync(messages, options, cancellationToken);
    }

    public override IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new ChatOptions();
        options.StopSequences ??= [];
        options.StopSequences.Add(_eotToken);

        return base.GetStreamingResponseAsync(messages, options, cancellationToken);
    }
}
