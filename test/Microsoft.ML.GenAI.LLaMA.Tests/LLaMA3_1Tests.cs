// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ApprovalTests;
using ApprovalTests.Namers;
using ApprovalTests.Reporters;
using TorchSharp;
using Xunit;
using Microsoft.ML.GenAI.Core.Extension;

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
        var model = new LlamaForCausalLM(LlamaConfig.Llama3_1_8B_Instruct);
        var stateDictStr = model.PeekShape();
        Approvals.Verify(stateDictStr);
    }

}
