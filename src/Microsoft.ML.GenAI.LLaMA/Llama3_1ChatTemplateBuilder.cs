// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Text;
using AutoGen.Core;
using Microsoft.ML.GenAI.Core;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

namespace Microsoft.ML.GenAI.LLaMA;
#pragma warning disable MSML_GeneralName // This name should be PascalCased
public class Llama3_1ChatTemplateBuilder : IChatTemplateBuilder
#pragma warning restore MSML_GeneralName // This name should be PascalCased
{
    private const char Newline = '\n';

    public string BuildPrompt(IEnumerable<IMessage> messages)
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
        // https://github.com/meta-llama/llama3/blob/11817d47e1ba7a4959b025eb1ca308572e0e3963/llama/generation.py#L280

        var sb = new StringBuilder();
        sb.Append("<|begin_of_text|>");
        foreach (var message in messages)
        {
            var role = message.GetRole()!.Value;
            var content = message.GetContent()!;
            sb.Append(message switch
            {
                _ when message.GetRole() == Role.System => $"<|start_header_id|>system<|end_header_id|>{Newline}{content.Trim()}<|eot_id|>{Newline}",
                _ when message.GetRole() == Role.User => $"<|start_header_id|>user<|end_header_id|>{Newline}{content.Trim()}<|eot_id|>{Newline}",
                _ when message.GetRole() == Role.Assistant => $"<|start_header_id|>assistant<|end_header_id|>{Newline}{content.Trim()}<|eot_id|>{Newline}",
                _ => throw new InvalidOperationException("Invalid role.")
            });
        }

        sb.Append($"<|start_header_id|>assistant<|end_header_id|>{Newline}");
        var input = sb.ToString();

        return input;
    }

    public string BuildPrompt(ChatHistory chatHistory)
    {
        // build prompt from chat history
        var sb = new StringBuilder();

        sb.Append("<|begin_of_text|>");
        foreach (var message in chatHistory)
        {
            foreach (var item in message.Items)
            {
                if (item is not TextContent textContent)
                {
                    throw new NotSupportedException($"Only text content is supported, but got {item.GetType().Name}");
                }

                var text = textContent.Text?.Trim() ?? string.Empty;

                var prompt = message.Role switch
                {
                    _ when message.Role == AuthorRole.System => $"<|start_header_id|>system<|end_header_id|>{Newline}{text}<|eot_id|>{Newline}",
                    _ when message.Role == AuthorRole.User => $"<|start_header_id|>user<|end_header_id|>{Newline}{text}<|eot_id|>{Newline}",
                    _ when message.Role == AuthorRole.Assistant => $"<|start_header_id|>assistant<|end_header_id|>{Newline}{text}<|eot_id|>{Newline}",
                    _ => throw new NotSupportedException($"Unsupported role {message.Role}")
                };

                sb.Append(prompt);
            }
        }

        sb.Append($"<|start_header_id|>assistant<|end_header_id|>{Newline}");

        return sb.ToString();
    }

    public static Llama3_1ChatTemplateBuilder Instance { get; } = new Llama3_1ChatTemplateBuilder();
}
