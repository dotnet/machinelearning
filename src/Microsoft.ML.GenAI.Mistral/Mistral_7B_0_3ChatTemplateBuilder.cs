// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Text;
using AutoGen.Core;
using Microsoft.ML.GenAI.Core;
using Microsoft.SemanticKernel.ChatCompletion;

namespace Microsoft.ML.GenAI.Mistral;

/// <summary>
/// the chat template builder for Mistral 7B v0.3
/// </summary>
#pragma warning disable MSML_GeneralName // This name should be PascalCased
public class Mistral_7B_0_3ChatTemplateBuilder : IChatTemplateBuilder
#pragma warning restore MSML_GeneralName // This name should be PascalCased
{
    private const char Newline = '\n';

    public static Mistral_7B_0_3ChatTemplateBuilder Instance { get; } = new Mistral_7B_0_3ChatTemplateBuilder();

    public string BuildPrompt(IEnumerable<IMessage> messages, IEnumerable<FunctionContract>? tools = null)
    {
        // can only contain at most one system message
        if (messages.Where(m => m.GetRole() == Role.System).Count() > 1)
        {
            throw new InvalidOperationException("Please provide at most one system message.");
        }

        var systemMessage = messages.FirstOrDefault(m => m.GetRole() == Role.System)?.GetContent();

        // split the messages into two sequences by the last user message
        // e.g [user, assistant, user, assistant, user] -> [[user, assistant, user, assistant], [user]]

        var firstSequence = messages.Take(messages.ToList().FindLastIndex(m => m.GetRole() == Role.User) + 1);
        var secondSequence = messages.Skip(messages.ToList().FindLastIndex(m => m.GetRole() == Role.User) + 1);

        var sb = new StringBuilder();
        sb.Append("<s>");
        foreach (var message in firstSequence)
        {
            // skip system
            if (message.GetRole() == Role.System)
            {
                continue;
            }

            var role = message.GetRole()!.Value;
            var content = message.GetContent()!;
            sb.Append(message switch
            {
                _ when message.GetRole() == Role.User => $"[INST]{content.Trim()}[/INST]",
                _ when message.GetRole() == Role.Assistant => $"{content.Trim()}</s>",
                _ => throw new InvalidOperationException("Invalid role.")
            });
        }

        foreach (var message in secondSequence)
        {
            var role = message.GetRole()!.Value;
            var content = message.GetContent()!;
            sb.Append(message switch
            {
                _ when message.GetRole() == Role.User && !string.IsNullOrEmpty(systemMessage) => $"[INST] {systemMessage} {Newline}{Newline}{content.Trim()}[/INST]",
                _ when message.GetRole() == Role.User => $"[INST]{content.Trim()}[/INST]",
                _ when message.GetRole() == Role.Assistant => $"{content.Trim()}</s>",
                _ => throw new InvalidOperationException("Invalid role.")
            });
        }

        return sb.ToString();
    }

    public string BuildPrompt(ChatHistory chatHistory)
    {
        throw new NotImplementedException();
    }
}
