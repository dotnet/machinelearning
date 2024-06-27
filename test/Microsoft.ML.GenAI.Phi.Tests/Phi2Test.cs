using ApprovalTests.Namers;
using ApprovalTests.Reporters;
using ApprovalTests;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using Xunit;
using TorchSharp;
using FluentAssertions;
using Microsoft.ML.TestFramework;
using Xunit.Abstractions;
using Microsoft.ML.Tokenizers;
using Microsoft.ML.GenAI.Core.Extension;
using System.Text.Json;
using Microsoft.ML.GenAI.Phi.Module;
namespace Microsoft.ML.GenAI.Phi.Tests;

public class Phi2Test : BaseTestClass
{
    public Phi2Test(ITestOutputHelper output) : base(output)
    {
    }

    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public void LoadSafeTensorShapeTest()
    {
        torch.set_default_device("meta");
        var modelWeightFolder = "C:\\Users\\xiaoyuz\\source\\repos\\phi-2";
        var configName = "config.json";
        var config = Path.Join(modelWeightFolder, configName);
        var modelConfig = JsonSerializer.Deserialize<Phi2Config>(File.ReadAllText(config)) ?? throw new ArgumentNullException(nameof(config));
        var model = new Phi2ForCasualLM(modelConfig);
        var stateDictStr = model.PeekShape();
        Approvals.Verify(stateDictStr);
    }

    //[Fact]
    //[UseReporter(typeof(DiffReporter))]
    //[UseApprovalSubdirectory("Approvals")]
    //public async Task ForwardTest()
    //{
    //    // create dummy input id with 128 length and attention mask
    //    var device = "cuda";
    //    var inputIds = torch.arange(128, dtype: ScalarType.Int64, device: device).unsqueeze(0);
    //    var attentionMask = torch.ones(1, 128, device: device);
    //    var modelWeightFolder = "C:\\Users\\xiaoyuz\\source\\repos\\phi-2";
    //    var model = Phi2ForCasualLM.FromPretrained(modelWeightFolder, torchDtype: ScalarType.BFloat16, checkPointName: "model.safetensors.index.json", device: "cuda");
    //    var input = new CasualLMModelInput(inputIds, attentionMask, past_key_values_length: 0);
    //    var output = model.forward(input);
    //    var outputTokenIds = output.last_hidden_state;
    //    var outputLogits = output.logits;

    //    var outputTokenIdsStr = outputTokenIds.Peek("output");
    //    var outputLogitsStr = outputLogits.Peek("logits");

    //    var sb = new StringBuilder();
    //    sb.AppendLine(outputTokenIdsStr);
    //    sb.AppendLine(outputLogitsStr);

    //    Approvals.Verify(sb.ToString());
    //}

    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public void TokenizerTest()
    {
        var modelWeightFolder = "C:\\Users\\xiaoyuz\\source\\repos\\phi-2";
        var tokenizer = Tokenizer.CreatePhi2(modelWeightFolder, addBeginOfSentence: true);
        tokenizer.EndOfSentenceId.Should().Be(50256);
        tokenizer.BeginningOfSentenceId.Should().Be(50256);
        var messages = new string[]
        {
            "Can you provide ways to eat combinations of bananas and dragonfruits?",
            "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey.",
            "What about solving an 2x + 3 = 7 equation?"
        };
        var sb = new StringBuilder();
        foreach (var message in messages)
        {
            var tokenized = tokenizer.EncodeToIds(message, true, false);
            var tokenizedStr = string.Join(", ", tokenized.Select(x => x.ToString()));

            sb.AppendLine(tokenizedStr);
        }
        Approvals.Verify(sb.ToString());
    }
}
