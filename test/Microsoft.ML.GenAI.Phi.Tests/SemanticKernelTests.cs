// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using FluentAssertions;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML.GenAI.Core;
using Microsoft.ML.GenAI.Phi.Extension;
using Microsoft.ML.Tokenizers;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Moq;
using Xunit;

namespace Microsoft.ML.GenAI.Phi.Tests;

public class SemanticKernelTests
{
    [Fact]
    public async Task ItAddPhi3CausalLMChatCompletionServiceTestAsync()
    {
        var pipeline = Mock.Of<ICausalLMPipeline<Tokenizer, Phi3ForCausalLM>>();
        // mock generate api
        Mock.Get(pipeline).Setup(p => p.Generate(
            It.IsAny<string>(), // prompt
            It.IsAny<int>(),    // max length
            It.IsAny<float>(),  // temperature 
            It.IsAny<float>(),  // top_p
            It.IsAny<string[]>()))   // stop sequence
            .Callback((string prompt, int maxLen, float temperature, float topP, string[] stopSequences) =>
            {
                // check prompt
                prompt.Should().Be("<|system|>\nyou are a helpful assistant<|end|>\n<|user|>\nhey<|end|>\n<|assistant|>");
            })
            .Returns((string prompt, int maxLen, float temperature, float topP, string[] stopSequences) => "hello");

        var kernel = Kernel.CreateBuilder()
            .AddGenAIChatCompletion(pipeline)
            .Build();

        var chatService = kernel.Services.GetRequiredService<IChatCompletionService>();

        var chatHistory = new ChatHistory();
        chatHistory.AddSystemMessage("you are a helpful assistant");
        chatHistory.AddUserMessage("hey");
        var responses = await chatService.GetChatMessageContentsAsync(chatHistory);
        responses.Count().Should().Be(1);
        var response = responses.First();
        response.Role.Should().Be(AuthorRole.Assistant);
        response.Items.Count().Should().Be(1);
        var textContent = response.Items.First() as TextContent;
        textContent!.Text.Should().Be("hello");
    }

    [Fact]
    public async Task ItAddPhi3CausalLMTextGenerationServiceTestAsync()
    {
        var pipeline = Mock.Of<ICausalLMPipeline<Tokenizer, Phi3ForCausalLM>>();
        // mock generate api
        Mock.Get(pipeline).Setup(p => p.Generate(
            It.IsAny<string>(), // prompt
            It.IsAny<int>(),    // max length
            It.IsAny<float>(),  // temperature 
            It.IsAny<float>(),  // top_p
            It.IsAny<string[]>()))   // stop sequence
            .Callback((string prompt, int maxLen, float temperature, float topP, string[] stopSequences) =>
            {
                // check prompt
                prompt.Should().Be("test");
            })
            .Returns((string prompt, int maxLen, float temperature, float topP, string[] stopSequences) => "hello");

        var kernel = Kernel.CreateBuilder()
            .AddGenAITextGeneration(pipeline)
            .Build();

        var response = await kernel.InvokePromptAsync("test");
    }
}
