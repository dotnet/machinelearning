// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AutoGen.Core;
using Microsoft.Extensions.AI;
using Microsoft.ML.GenAI.Core;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using TextContent = Microsoft.SemanticKernel.TextContent;

namespace Microsoft.ML.GenAI.Phi;

public class Phi3ChatTemplateBuilder : IChatTemplateBuilder, IMEAIChatTemplateBuilder
{
    private const char Newline = '\n';

    public static Phi3ChatTemplateBuilder Instance => new Phi3ChatTemplateBuilder();

    public string BuildPrompt(IEnumerable<IMessage> messages, IEnumerable<FunctionContract>? tools = null)
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

    public string BuildPrompt(ChatHistory chatHistory)
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
                    _ when message.Role == AuthorRole.System => $"<|system|>{Newline}{textContent}<|end|>{Newline}",
                    _ when message.Role == AuthorRole.User => $"<|user|>{Newline}{textContent}<|end|>{Newline}",
                    _ when message.Role == AuthorRole.Assistant => $"<|assistant|>{Newline}{textContent}<|end|>{Newline}",
                    _ => throw new NotSupportedException($"Unsupported role {message.Role}")
                };

                sb.Append(prompt);
            }
        }

        sb.Append("<|assistant|>");

        return sb.ToString();
    }

    public string BuildPrompt(IList<ChatMessage> messages, ChatOptions? options = null)
    {
        var availableRoles = new[] { ChatRole.System, ChatRole.User, ChatRole.Assistant };
        if (messages.Any(m => m.Text is null))
        {
            throw new InvalidOperationException("Please provide a message with content.");
        }

        if (messages.Any(m => availableRoles.Any(availableRole => availableRole == m.Role) == false))
        {
            throw new InvalidOperationException("Please provide a message with a valid role. The valid roles are System, User, and Assistant.");
        }

        // construct template based on instruction from
        // https://huggingface.co/microsoft/Phi-3-mini-128k-instruct#chat-format

        var sb = new StringBuilder();
        foreach (var message in messages)
        {
            var role = message.Role.Value;
            var content = message.Text;
            sb.Append(message switch
            {
                _ when message.Role == ChatRole.System => $"<|system|>{Newline}{content}<|end|>{Newline}",
                _ when message.Role == ChatRole.User => $"<|user|>{Newline}{content}<|end|>{Newline}",
                _ when message.Role == ChatRole.Assistant => $"<|assistant|>{Newline}{content}<|end|>{Newline}",
                _ => throw new InvalidOperationException("Invalid role.")
            });
        }

        sb.Append("<|assistant|>");
        var input = sb.ToString();

        return input;
    }
}
