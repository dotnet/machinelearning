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
using TorchSharp;
using Xunit;

namespace Microsoft.ML.GenAI.Phi.Tests;

[Collection("NoParallelization")]
public class Phi3Tests
{
    public Phi3Tests()
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
    public void Phi3Mini4KShapeTest()
    {
        var model = new Phi3ForCasualLM(Phi3Config.Phi3Mini4kInstruct);
        var stateDictStr = model.PeekShape();
        Approvals.Verify(stateDictStr);
    }

    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public void Phi3Mini4KInt8QuantizeShapeTest()
    {
        var model = new Phi3ForCasualLM(Phi3Config.Phi3Mini4kInstruct);
        model.ToInt8QuantizeModule();
        var size = model.GetSizeInBytes();
        var stateDictStr = model.PeekShape();
        var sizeInGB = size / 1024 / 1024 / 1024;
        sizeInGB.Should().Be(3);
        Approvals.Verify(stateDictStr);
    }

    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public void Phi3Mini4KInt4QuantizeShapeTest()
    {
        var model = new Phi3ForCasualLM(Phi3Config.Phi3Mini4kInstruct);
        model.ToInt4QuantizeModule();
        var size = model.GetSizeInBytes();
        var stateDictStr = model.PeekShape();
        var sizeInGB = size / 1024 / 1024 / 1024;
        sizeInGB.Should().Be(2);
        Approvals.Verify(stateDictStr);
    }

    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public void Phi3Medium4KShapeTest()
    {
        var model = new Phi3ForCasualLM(Phi3Config.Phi3Medium4kInstruct);
        var stateDictStr = model.PeekShape();
        Approvals.Verify(stateDictStr);
    }


    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public void Phi3Medium128KShapeTest()
    {
        var model = new Phi3ForCasualLM(Phi3Config.Phi3Medium128kInstruct);
        var stateDictStr = model.PeekShape();
        Approvals.Verify(stateDictStr);
    }

    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public void Phi3Mini128KShapeTest()
    {
        var model = new Phi3ForCasualLM(Phi3Config.Phi3Mini128kInstruct);
        var stateDictStr = model.PeekShape();
        Approvals.Verify(stateDictStr);
    }

    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public void Phi3Mini128KLayerSizeTest()
    {
        var model = new Phi3ForCasualLM(Phi3Config.Phi3Mini128kInstruct);
        var size = model.GetSizeForEachDynamicLayerInBytes();
        // convert size to MB
        var sizeInMB = size.ToDictionary(x => x.Key, x => x.Value / 1024 / 1024);

        var json = JsonSerializer.Serialize(sizeInMB, new JsonSerializerOptions { WriteIndented = true });
        Approvals.Verify(json);
    }

    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public void TokenizerTest()
    {
        var modelWeightFolder = Path.Join("Llama");
        var modelPath = Path.Join(modelWeightFolder, "tokenizer.model");
        var tokenizer = Phi3TokenizerHelper.FromPretrained(modelPath);
        tokenizer.BeginningOfSentenceId.Should().Be(1);
        tokenizer.EndOfSentenceId.Should().Be(2);

        // test <|end|>
        var endIds = tokenizer.EncodeToIds("<|end|>", addBeginningOfSentence: false, addEndOfSentence: false, considerPreTokenization: false, considerNormalization: false);
        endIds.Should().BeEquivalentTo(new int[] { 32007 });

        var messages = new string[]
        {
            "Can you provide ways to eat combinations of bananas and dragonfruits?",
            "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey.",
            "What about solving an 2x + 3 = 7 equation?",
            "\nCount to 3\n",
            "<|user|>",
            "<|end|>",
            "<|assistant|>",
            "<|user|>\nCount to 3<|end|>\n<|assistant|>",
        };
        var sb = new StringBuilder();
        foreach (var message in messages)
        {
            var tokenizeIds = tokenizer.EncodeToIds(message, true, false, considerPreTokenization: true);
            var decodeToString = tokenizer.Decode(tokenizeIds, considerSpecialTokens: true);
            sb.AppendLine(decodeToString);
            var tokenizedStr = string.Join(", ", tokenizeIds.Select(x => x.ToString()));

            sb.AppendLine(tokenizedStr);
        }
        Approvals.Verify(sb.ToString());
    }
}
