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

    public Phi3Agent(
        ICausalLMPipeline<Tokenizer, Phi3ForCasualLM> pipeline,
        string name,
        string? systemMessage = "you are a helpful assistant")
    {
        this.Name = name;
        this._pipeline = pipeline;
        this._systemMessage = systemMessage;
    }

    public string Name { get; }

    public Task<IMessage> GenerateReplyAsync(IEnumerable<IMessage> messages, GenerateReplyOptions? options = null, CancellationToken cancellationToken = default)
    {
        var input = BuildPrompt(messages);
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
        var input = BuildPrompt(messages);
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

    private string BuildPrompt(IEnumerable<IMessage> messages)
    {
        var availableRoles = new[] { Role.System, Role.User, Role.Assistant };
        if (messages.Any(m => m.GetContent() is null))
        {
            throw new InvalidOperationException("Please provide a message with content.");
        }

        if (messages.Any(m => m.GetRole() is null || availableRoles.Contains(m.GetRole()!.Value) == false))
        {
            throw new InvalidOperationException("Please provide a message with a valid role. The valid roles are System, User, and Assistant.");
        }

        // construct template based on instruction from
        // https://huggingface.co/microsoft/Phi-3-mini-128k-instruct#chat-format

        var sb = new StringBuilder();
        if (_systemMessage is not null)
        {
            sb.Append($"<|system|>{Newline}{_systemMessage}<|end|>{Newline}");
        }
        foreach (var message in messages)
        {
            var role = message.GetRole()!.Value;
            var content = message.GetContent()!;
            sb.Append(message switch
            {
                _ when message.GetRole() == Role.System => $"<|system|>{Newline}{content}<|end|>{Newline}",
                _ when message.GetRole() == Role.User => $"<|user|>{Newline}{content}<|end|>{Newline}",
                _ when message.GetRole() == Role.Assistant => $"<|assistant|>{Newline}{content}<|end|>{Newline}",
                _ => throw new InvalidOperationException("Invalid role.")
            });
        }

        sb.Append("<|assistant|>");
        var input = sb.ToString();

        return input;
    }
}
