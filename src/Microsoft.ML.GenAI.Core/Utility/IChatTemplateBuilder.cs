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
using Microsoft.SemanticKernel.ChatCompletion;

namespace Microsoft.ML.GenAI.Core;

public interface ISemanticKernelChatTemplateBuilder
{
    string BuildPrompt(ChatHistory chatHistory);
}

public interface IAutoGenChatTemplateBuilder
{
    string BuildPrompt(IEnumerable<IMessage> messages, IEnumerable<FunctionContract>? tools = null);
}

public interface IMEAIChatTemplateBuilder
{
    /// <summary>
    /// Build a prompt from a list of messages.
    /// </summary>
    /// <param name="messages">the list of <see cref="ChatMessage"/> to be rendered</param>
    /// <param name="options"></param>
    /// <param name="appendAssistantTag">true if append assistant tag at the end of prompt.</param>
    /// <returns></returns>
    string BuildPrompt(IList<ChatMessage> messages, ChatOptions? options = null, bool appendAssistantTag = true);
}

public interface IChatTemplateBuilder : IAutoGenChatTemplateBuilder, ISemanticKernelChatTemplateBuilder
{
}
