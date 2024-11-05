// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using AutoGen.Core;
using Microsoft.ML.GenAI.Core;
using Microsoft.ML.Tokenizers;

namespace Microsoft.ML.GenAI.Phi;

public class Phi3Agent : IStreamingAgent
{
    private const char Newline = '\n';
    private readonly ICausalLMPipeline<Tokenizer, Phi3ForCasualLM> _pipeline;
    private readonly string? _systemMessage;
    private readonly IAutoGenChatTemplateBuilder _templateBuilder;

    public Phi3Agent(
        ICausalLMPipeline<Tokenizer, Phi3ForCasualLM> pipeline,
        string name,
        string? systemMessage = "you are a helpful assistant",
        IAutoGenChatTemplateBuilder? templateBuilder = null)
    {
        this.Name = name;
        this._pipeline = pipeline;
        this._systemMessage = systemMessage;
        this._templateBuilder = templateBuilder ?? Phi3ChatTemplateBuilder.Instance;
    }

    public string Name { get; }

    public Task<IMessage> GenerateReplyAsync(IEnumerable<IMessage> messages, GenerateReplyOptions? options = null, CancellationToken cancellationToken = default)
    {
        if (_systemMessage != null)
        {
            var systemMessage = new TextMessage(Role.System, _systemMessage, from: this.Name);
            messages = messages.Prepend(systemMessage);
        }

        var input = _templateBuilder.BuildPrompt(messages);
        var maxLen = options?.MaxToken ?? 1024;
        var temperature = options?.Temperature ?? 0.7f;
        var stopTokenSequence = options?.StopSequence ?? [];
        stopTokenSequence = stopTokenSequence.Append("<|end|>").ToArray();

        var output = _pipeline.Generate(
            input,
            maxLen: maxLen,
            temperature: temperature,
            stopSequences: stopTokenSequence) ?? throw new InvalidOperationException("Failed to generate a reply.");

        return Task.FromResult<IMessage>(new TextMessage(Role.Assistant, output, from: this.Name));
    }

#pragma warning disable CS1998 // Async method lacks 'await' operators and will run synchronously
    public async IAsyncEnumerable<IMessage> GenerateStreamingReplyAsync(
#pragma warning restore CS1998 // Async method lacks 'await' operators and will run synchronously
        IEnumerable<IMessage> messages,
        GenerateReplyOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (_systemMessage != null)
        {
            var systemMessage = new TextMessage(Role.System, _systemMessage, from: this.Name);
            messages = messages.Prepend(systemMessage);
        }

        var input = _templateBuilder.BuildPrompt(messages);
        var maxLen = options?.MaxToken ?? 1024;
        var temperature = options?.Temperature ?? 0.7f;
        var stopTokenSequence = options?.StopSequence ?? [];
        stopTokenSequence = stopTokenSequence.Append("<|end|>").ToArray();

        foreach (var output in _pipeline.GenerateStreaming(
            input,
            maxLen: maxLen,
            temperature: temperature,
            stopSequences: stopTokenSequence))
        {
            yield return new TextMessageUpdate(Role.Assistant, output, from: this.Name);
        }
    }
}
