// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.RunTests;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public class BinaryLoaderSaverTests : TestDataPipeBase
    {
        public BinaryLoaderSaverTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void OldKeyTypeCodecTest()
        {
            // Checks that we can load IDataViews defined with unknown cardinality KeyType.
            // schema-codec-test.idv was generated with the following command before simplifying the KeyType:
            // dotnet MML.dll savedata loader=text{col=A:U4[0-2]:0 col=B:U4[0-5]:0 col=C:U1[0-10]:0 col=D:U2[0-*]:0 col=E:U4[0-*]:0 col=F:U8[0-*]:0} dout=codectest.idv
            var data = ML.Data.LoadFromBinary(GetDataPath("schema-codec-test.idv"));

            var outputPath = GetOutputPath("BinaryLoaderSaver", "OldKeyTypeCodecTest.txt");
            using (var ch = Env.Start("save"))
            {
                using (var fs = File.Create(outputPath))
                    ML.Data.SaveAsText(data, fs, headerRow: false);
            }
            CheckEquality("BinaryLoaderSaver", "OldKeyTypeCodecTest.txt");
            Done();
        }
    }
}
