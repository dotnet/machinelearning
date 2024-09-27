// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using ApprovalTests;
using ApprovalTests.Namers;
using ApprovalTests.Reporters;
using Microsoft.ML.GenAI.Core.Extension;
using TorchSharp;
using Xunit;

namespace Microsoft.ML.GenAI.LLaMA.Tests;

[Collection("NoParallelization")]
public class LLaMA3_2Tests
{
    public LLaMA3_2Tests()
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
    public void Llama_3_2_1b_ShapeTest()
    {
        var model = new LlamaForCausalLM(LlamaConfig.Llama3_2_1B_Instruct);
        var stateDictStr = model.PeekShape();
        Approvals.Verify(stateDictStr);
    }

    [WindowsOnlyFact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public void Llama_3_2_3b_ShapeTest()
    {
        var model = new LlamaForCausalLM(LlamaConfig.Llama_3_2_3B_Instruct);
        var stateDictStr = model.PeekShape();
        Approvals.Verify(stateDictStr);
    }
}
