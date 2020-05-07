// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.ML.Data;
using Microsoft.ML.TestFrameworkCommon.Attributes;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.RunTests
{
    public class PartitionedFileLoaderTests : TestDataPipeBase
    {
        protected override void Initialize()
        {
            base.Initialize();
            Env.ComponentCatalog.RegisterAssembly(typeof(ParquetLoader).Assembly);
        }

        public PartitionedFileLoaderTests(ITestOutputHelper output)
            : base(output)
        {

        }

        [Theory]
        [IterationData(iterations: 20)]
        [Trait("Category", "RunSpecificTest")]
        public void CompletePartitionedNamedDirectories(int iterations)
        {
            Output.WriteLine($"{iterations} - th");

            int timeout = 20 * 60 * 1000;

            var runTask = Task.Run(PartitionedNamedDirectories);
            var timeoutTask = Task.Delay(timeout + iterations);
            var finishedTask = Task.WhenAny(timeoutTask, runTask).Result;
            if (finishedTask == timeoutTask)
            {
                Console.WriteLine("PartitionedNamedDirectories test Hanging: fail to complete in 20 minutes");
                Environment.FailFast("Fail here to take memory dump");
            }
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

        [Theory]
        [IterationData(iterations: 20)]
        [Trait("Category", "RunSpecificTest")]
        public void CompletePartitionedUnnamedDirectories(int iterations)
        {
            Output.WriteLine($"{iterations} - th");

            int timeout = 20 * 60 * 1000;

            var runTask = Task.Run(PartitionedUnnamedDirectories);
            var timeoutTask = Task.Delay(timeout + iterations);
            var finishedTask = Task.WhenAny(timeoutTask, runTask).Result;
            if (finishedTask == timeoutTask)
            {
                Console.WriteLine("PartitionedUnnamedDirectories test Hanging: fail to complete in 20 minutes");
                Environment.FailFast("Fail here to take memory dump");
            }
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
