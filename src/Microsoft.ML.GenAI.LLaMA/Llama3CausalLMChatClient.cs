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
            metadata ?? new ChatClientMetadata(modelId: nameof(Llama3CausalLMChatClient)))
    {
    }

    public override Task<ChatCompletion> CompleteAsync(
        IList<ChatMessage> chatMessages,
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

        return base.CompleteAsync(chatMessages, options, cancellationToken);
    }

#pragma warning disable CS1998 // Async method lacks 'await' operators and will run synchronously
    public override async IAsyncEnumerable<StreamingChatCompletionUpdate> CompleteStreamingAsync(
#pragma warning restore CS1998 // Async method lacks 'await' operators and will run synchronously
        IList<ChatMessage> chatMessages,
        ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
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

        return base.CompleteStreamingAsync(chatMessages, options, cancellationToken);
    }
}
