// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Text;
using ApprovalTests;
using ApprovalTests.Namers;
using ApprovalTests.Reporters;
using AutoGen.Core;
using Microsoft.Extensions.AI;
using Microsoft.ML.GenAI.Core.Extension;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using TorchSharp;
using Xunit;

namespace Microsoft.ML.GenAI.LLaMA.Tests;

[Collection("NoParallelization")]
public class LLaMA3_1Tests
{
    public LLaMA3_1Tests()
    {
        if (Environment.GetEnvironmentVariable("HELIX_CORRELATION_ID") != null)
        {
            Approvals.UseAssemblyLocationForApprovedFiles();
        }

        torch.set_default_device("meta");
    }

    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public void Llama_3_1_8b_ShapeTest()
    {
        var model = new LlamaForCausalLM(LlamaConfig.Llama3_1_8B_Instruct, "meta");
        var stateDictStr = model.PeekShape();
        Approvals.Verify(stateDictStr);
    }

    [WindowsOnlyFact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public void Llama_3_1_70b_ShapeTest()
    {
        var model = new LlamaForCausalLM(LlamaConfig.Llama3_1_70B_Instruct, "meta");
        var stateDictStr = model.PeekShape();
        Approvals.Verify(stateDictStr);
    }

    [WindowsOnlyFact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public void Llama_3_1_405b_ShapeTest()
    {
        var model = new LlamaForCausalLM(LlamaConfig.Llama3_1_405B_Instruct, "meta");
        var stateDictStr = model.PeekShape();
        Approvals.Verify(stateDictStr);
    }

    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public void TokenizerTest()
    {
        var modelWeightFolder = Path.Join("Llama-3.1");
        var tokenizer = LlamaTokenizerHelper.FromPretrained(modelWeightFolder);

        var messages = new string[]
        {
            "Can you provide ways to eat combinations of bananas and dragonfruits?",
            "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey.",
            "What about solving an 2x + 3 = 7 equation?",
            """
            <|begin_of_text|>Hello World<|end_of_text|>
            """
        };

        var sb = new StringBuilder();
        foreach (var message in messages)
        {
            var tokenizeIds = tokenizer.EncodeToIds(message, true, false);
            var decodeToString = tokenizer.Decode(tokenizeIds);
            sb.AppendLine(decodeToString);
            var tokenizedStr = string.Join(", ", tokenizeIds.Select(x => x.ToString()));

            sb.AppendLine(tokenizedStr);
        }
        Approvals.Verify(sb.ToString());
    }

    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public void ItBuildChatTemplateFromAutoGenChatHistory()
    {
        var chatHistory = new List<IMessage>
        {
            new TextMessage(Role.System, "You are a helpful AI assistant."),
            new TextMessage(Role.User, "Hello?"),
            new TextMessage(Role.Assistant, "World!"),
        };

        var prompt = Llama3_1ChatTemplateBuilder.Instance.BuildPrompt(chatHistory);

        Approvals.Verify(prompt);
    }

    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public void ItBuildChatTemplateFromSemanticKernelChatHistory()
    {
        var chatHistory = new ChatHistory
        {
            new ChatMessageContent(AuthorRole.System, "You are a helpful AI assistant."),
            new ChatMessageContent(AuthorRole.User, "Hello?"),
            new ChatMessageContent(AuthorRole.Assistant, "World!"),
        };

        var prompt = Llama3_1ChatTemplateBuilder.Instance.BuildPrompt(chatHistory);

        Approvals.Verify(prompt);
    }

    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public void ItBuildChatTemplateFromMEAIChatHistory()
    {
        var chatHistory = new[]
        {
            new ChatMessage(ChatRole.System, "You are a helpful AI assistant."),
            new ChatMessage(ChatRole.User, "Hello?"),
            new ChatMessage(ChatRole.Assistant, "World!"),
        };

        var prompt = Llama3_1ChatTemplateBuilder.Instance.BuildPrompt(chatHistory);

        Approvals.Verify(prompt);
    }
}
