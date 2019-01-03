// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.RunTests;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public class CSharpCodeGen : BaseTestBaseline
    {
        public CSharpCodeGen(ITestOutputHelper output) : base(output)
        {
        }

        [Fact(Skip = "Execute this test if you want to regenerate CSharpApi file")]
        public void RegenerateCSharpApi()
        {
            var basePath = GetDataPath("../../src/Microsoft.ML.Legacy/CSharpApi.cs");
            Tools.Maml.Main(new[] { $"? generator=cs{{csFilename={basePath}}}" });
        }

        [ConditionalFact(typeof(BaseTestBaseline), nameof(LessThanNetCore30OrNotNetCore))]
        public void TestGeneratedCSharpAPI()
        {
            var dataPath = GetOutputPath("Api.cs");
            Tools.Maml.Main(new[] { $"? generator=cs{{csFilename={dataPath}}}" });

            var basePath = GetDataPath("../../src/Microsoft.ML.Legacy/CSharpApi.cs");
            using (StreamReader baseline = OpenReader(basePath))
            using (StreamReader result = OpenReader(dataPath))
            {
                for (; ; )
                {
                    string line1 = baseline.ReadLine();
                    string line2 = result.ReadLine();

                    if (line1 == null && line2 == null)
                        break;
                    Assert.Equal(line1, line2);
                }
            }
        }
    }
}
