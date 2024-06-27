// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Text;
using System.Text.Json;
using ApprovalTests;
using ApprovalTests.Namers;
using ApprovalTests.Reporters;
using FluentAssertions;
using Microsoft.ML.GenAI.Core.Extension;
using Microsoft.ML.TestFramework;
using TorchSharp;
using Xunit;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Phi.Tests;

public class Phi3Tests : BaseTestClass
{
    public Phi3Tests(ITestOutputHelper output) : base(output)
    {
        torch.set_default_device("meta");
    }

    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public void Phi3Mini4KShapeTest()
    {
        var modelWeightFolder = "C:\\Users\\xiaoyuz\\source\\repos\\Phi-3-mini-4k-instruct";
        var config = Path.Join(modelWeightFolder, "config.json");
        var modelConfig = JsonSerializer.Deserialize<Phi3Config>(File.ReadAllText(config)) ?? throw new ArgumentNullException(nameof(config));
        var model = new Phi3ForCasualLM(modelConfig);
        var stateDictStr = model.PeekShape();
        Approvals.Verify(stateDictStr);
    }

    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public void Phi3Medium4KShapeTest()
    {
        var modelWeightFolder = "C:\\Users\\xiaoyuz\\source\\repos\\Phi-3-medium-4k-instruct";
        var config = Path.Join(modelWeightFolder, "config.json");
        var modelConfig = JsonSerializer.Deserialize<Phi3Config>(File.ReadAllText(config)) ?? throw new ArgumentNullException(nameof(config));
        var model = new Phi3ForCasualLM(modelConfig);
        var stateDictStr = model.PeekShape();
        Approvals.Verify(stateDictStr);
    }


    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public void Phi3Medium128KShapeTest()
    {
        var modelWeightFolder = "C:\\Users\\xiaoyuz\\source\\repos\\Phi-3-medium-128k-instruct";
        var config = Path.Join(modelWeightFolder, "config.json");
        var modelConfig = JsonSerializer.Deserialize<Phi3Config>(File.ReadAllText(config)) ?? throw new ArgumentNullException(nameof(config));
        var model = new Phi3ForCasualLM(modelConfig);
        var stateDictStr = model.PeekShape();
        Approvals.Verify(stateDictStr);
    }

    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public void Phi3Mini128KShapeTest()
    {
        var modelWeightFolder = "C:\\Users\\xiaoyuz\\source\\repos\\Phi-3-mini-128k-instruct";
        var config = Path.Join(modelWeightFolder, "config.json");
        var modelConfig = JsonSerializer.Deserialize<Phi3Config>(File.ReadAllText(config)) ?? throw new ArgumentNullException(nameof(config));
        var model = new Phi3ForCasualLM(modelConfig);
        var stateDictStr = model.PeekShape();
        Approvals.Verify(stateDictStr);
    }

    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public void Phi3Mini128KLayerSizeTest()
    {
        var dtype = ScalarType.BFloat16;
        var modelWeightFolder = "C:\\Users\\xiaoyuz\\source\\repos\\Phi-3-mini-128k-instruct";
        var config = Path.Join(modelWeightFolder, "config.json");
        var modelConfig = JsonSerializer.Deserialize<Phi3Config>(File.ReadAllText(config)) ?? throw new ArgumentNullException(nameof(config));
        modelConfig.DType = dtype;
        var model = new Phi3ForCasualLM(modelConfig);

        var size = model.GetSizeForEachDynamicLayerInBytes();
        // convert size to MB
        var sizeInMB = size.ToDictionary(x => x.Key, x => x.Value * 1.0f / 1024 / 1024);

        var json = JsonSerializer.Serialize(sizeInMB, new JsonSerializerOptions { WriteIndented = true });
        Approvals.Verify(json);
    }

    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public void TokenizerTest()
    {
        var modelWeightFolder = "C:\\Users\\xiaoyuz\\source\\repos\\Phi-3-mini-4k-instruct";
        var tokenizer = LLama2Tokenizer.FromPretrained(modelWeightFolder);
        tokenizer.BosId.Should().Be(1);
        tokenizer.EosId.Should().Be(2);
        var messages = new string[]
        {
            "Can you provide ways to eat combinations of bananas and dragonfruits?",
            "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey.",
            "What about solving an 2x + 3 = 7 equation?",
            "\nCount to 3\n",
            "<|user|>\nCount to 3<|end|>\n<|assistant|>",
        };
        var sb = new StringBuilder();
        foreach (var message in messages)
        {
            var tokenized = tokenizer.Encode(message, true, false);
            var tokenizedStr = string.Join(", ", tokenized.Select(x => x.ToString()));

            sb.AppendLine(tokenizedStr);
        }
        Approvals.Verify(sb.ToString());
    }
}
