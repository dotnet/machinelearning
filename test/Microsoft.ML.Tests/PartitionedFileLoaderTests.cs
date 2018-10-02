// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.RunTests;
using System.IO;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public class PartitionedFileLoaderTests : TestDataPipeBase
    {
        public PartitionedFileLoaderTests(ITestOutputHelper output)
            : base(output)
        {

        }

        [Fact]
        public void PartitionedNamedDirectories()
        {
            string basePath = GetDataPath("Partitioned", "Named");
            string pathData = Path.Combine(basePath, "...", "*.csv");

            TestCore(pathData, false,
                new[] {
                    "loader=Part{bp=" + basePath + " loader=Text{header+ sep=comma col=L0:TX:0}}"
                });

            Done();
        }

        [Fact]
        public void PartitionedUnnamedDirectories()
        {
            string basePath = GetDataPath("Partitioned", "Unnamed"); ;
            string pathData = Path.Combine(basePath, "...", "*.csv");

            TestCore(pathData, false,
                new[] {
                    "loader=Part{parser=SmplPP{col=Month:I4:1} path+ bp=" + basePath + " loader=Text{header+ sep=comma col=L0:I4:1}}"
                });

            // Test again with global parser data type.
            TestCore(pathData, false,
                new[] {
                    "loader=Part{parser=SmplPP{type=I4 col=Month:1} path+ bp=" + basePath + " loader=Text{header+ sep=comma col=L0:I4:1}}"
                });

            Done();
        }
    }
}
