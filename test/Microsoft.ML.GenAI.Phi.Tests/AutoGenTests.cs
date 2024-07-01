// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using AutoGen.Core;
using FluentAssertions;
using Microsoft.ML.GenAI.Core;
using Microsoft.ML.Tokenizers;
using Moq;
using Xunit;

namespace Microsoft.ML.GenAI.Phi.Tests;

public class AutoGenTests
{
    [Fact]
    public async Task ItGenerateTextReply()
    {
        var pipeline = Mock.Of<ICausalLMPipeline<Tokenizer, Phi3ForCasualLM>>();
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

        var agent = new Phi3Agent(pipeline, "assistant");
        var reply = await agent.SendAsync("hey");

        reply.GetContent().Should().Be("hello");
        reply.From.Should().Be(agent.Name);
    }
}
