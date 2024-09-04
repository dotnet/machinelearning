// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Text;
using ApprovalTests;
using ApprovalTests.Namers;
using ApprovalTests.Reporters;
using AutoGen.Core;
using Microsoft.ML.GenAI.Core.Extension;
using TorchSharp;
using Xunit;

namespace Microsoft.ML.GenAI.Mistral.Tests;

[Collection("NoParallelization")]
public class Mistral_7B_Instruct_V0_3Tests
{
    public Mistral_7B_Instruct_V0_3Tests()
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
    public void Mistral_7B_Instruct_V0_3_ShapeTest()
    {
        var model = new MistralForCausalLM(MistralConfig.Mistral_7B_Instruct_v0_3);
        var stateDictStr = model.PeekShape();
        Approvals.Verify(stateDictStr);
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

        var prompt = Mistral_7B_0_3ChatTemplateBuilder.Instance.BuildPrompt(chatHistory);

        Approvals.Verify(prompt);
    }

    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public void ItBuildChatTemplateWithToolsFromAutoGenChatHistory()
    {
        var getWeatherTool = new FunctionContract
        {
            Name = "get_current_weather",
            Namespace = "weather",
            Description = "Get the current weather",
            Parameters = [
                new FunctionParameterContract
                {
                    Name = "location",
                    ParameterType = typeof(string),
                    Description = "The city and state, e.g. San Francisco, CA",
                    IsRequired = true
                }
                ]
        };

        var getWeatherToolCall = new ToolCall("get_current_weather", "{\"location\": \"Seattle, WA\"}") { ToolCallId = "9Ae3bDc2F" };
        var getWeatherToolCallResult = new ToolCall("get_current_weather", "{\"temperature\": 22.0}", "sunny") { ToolCallId = "9Ae3bDc2F" };
        var toolCallMessage = new ToolCallMessage([getWeatherToolCall]);
        var toolCallResultMessage = new ToolCallResultMessage([getWeatherToolCallResult]);
        var aggregateToolCallMessage = new ToolCallAggregateMessage(toolCallMessage, toolCallResultMessage);

        var chatHistory = new List<IMessage>
        {
            new TextMessage(Role.System, "You are a helpful AI assistant."),
            new TextMessage(Role.User, "What's the weather in Seattle?"),
            toolCallMessage,
            toolCallResultMessage,
            new TextMessage(Role.Assistant, "The current temperature in Seattle is 22.0 degrees celsius."),

            // test tool call aggregate message for immediate tool call execution
            new TextMessage(Role.User, "What's the weather in New York?"),
            aggregateToolCallMessage,
            new TextMessage(Role.Assistant, "The current temperature in New York is 22.0 degrees celsius."),

            new TextMessage(Role.User, "What's the weather in Paris?"),
        };

        var prompt = Mistral_7B_0_3ChatTemplateBuilder.Instance.BuildPrompt(chatHistory, [getWeatherTool]);

        Approvals.Verify(prompt);
    }

    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public void TokenizerTest()
    {
        var modelWeightFolder = Path.Join("C:\\Users\\xiaoyuz\\source\\repos\\Mistral-7B-Instruct-v0.3");
        var tokenizer = MistralTokenizerHelper.FromPretrained(modelWeightFolder);

        var messages = new string[]
        {
            // system : You are a helpful assistant that can answer questions about the weather.
            // tool: [get-weather-tool-call]
            // user : What's the weather like in Paris?
            // assistant: // get-weather-tool-call
            // tool: get-weather-tool-call-result
            // assistant: The current temperature in Paris is 22.0 degrees celsius.
            """
            <s>[AVAILABLE_TOOLS][{"type": "function", "function": {"name": "get_current_weather", "description": "Get the current weather", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "format": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The temperature unit to use. Infer this from the users location."}}, "required": ["location", "format"]}}}][/AVAILABLE_TOOLS][INST] What's the weather like in Paris?[/INST][TOOL_CALLS] [{"name": "get_current_temperature", "arguments": {"location": "Paris, France", "unit": "celsius"}, "id": "9Ae3bDc2F"}]</s>[TOOL_RESULTS] {"content": 22.0, "call_id": "9Ae3bDc2F"}[/TOOL_RESULTS] The current temperature in Paris is 22.0 degrees celsius.</s>
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
}
