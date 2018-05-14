// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.TestFramework;
using System.IO;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public class CSharpCodeGen : BaseTestClass
    {
        public CSharpCodeGen(ITestOutputHelper output) : base(output)
        {
        }

        //[Fact(Skip = "Temporary solution(Windows ONLY) to regenerate codegenerated CSharpAPI.cs")]
        [Fact]
        public void GenerateCSharpAPI()
        {
            var cSharpAPIPath = Path.Combine(RootDir, @"src\\Microsoft.ML\\CSharpApi.cs");
            Runtime.Tools.Maml.Main(new[] { $"? generator=cs{{csFilename={cSharpAPIPath}}}" });
        }
    }
}
