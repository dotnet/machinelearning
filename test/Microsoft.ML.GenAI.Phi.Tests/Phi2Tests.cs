// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Text;
using ApprovalTests;
using ApprovalTests.Namers;
using ApprovalTests.Reporters;
using FluentAssertions;
using Microsoft.ML.GenAI.Core.Extension;
using Microsoft.ML.Tokenizers;
using TorchSharp;
using Xunit;
namespace Microsoft.ML.GenAI.Phi.Tests;

[Collection("NoParallelization")]
public class Phi2Tests
{
    public Phi2Tests()
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
    public void LoadSafeTensorShapeTest()
    {
        var model = new Phi2ForCasualLM(Phi2Config.Phi2);
        var stateDictStr = model.PeekShape();
        Approvals.Verify(stateDictStr);
    }

    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public void TokenizerTest()
    {
        var modelWeightFolder = Path.Join("Phi-2");
        var tokenizer = Phi2TokenizerHelper.Create(modelWeightFolder, addBeginOfSentence: true);
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
