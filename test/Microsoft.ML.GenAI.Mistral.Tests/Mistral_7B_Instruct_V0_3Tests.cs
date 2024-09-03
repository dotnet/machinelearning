using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ApprovalTests.Namers;
using ApprovalTests.Reporters;
using ApprovalTests;
using Xunit;
using TorchSharp;
using Microsoft.ML.GenAI.Core.Extension;
using AutoGen.Core;

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
