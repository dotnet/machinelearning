// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Threading.Tasks;
using Microsoft.ML.Data;
using Microsoft.ML.TestFrameworkCommon.Attributes;
using Xunit;

namespace Microsoft.ML.RunTests
{
    public sealed partial class TestParquet : TestDataPipeBase
    {
        protected override void Initialize()
        {
            base.Initialize();
            Env.ComponentCatalog.RegisterAssembly(typeof(ParquetLoader).Assembly);
        }

        [Theory]
        [IterationData(iterations: 20)]
        [Trait("Category", "RunSpecificTest")]
        public void CompleteTransposerTest(int iterations)
        {
            Output.WriteLine($"{iterations} - th");

            int timeout = 20 * 60 * 1000;

            var runTask = Task.Run(TestParquetPrimitiveDataTypes);
            var timeoutTask = Task.Delay(timeout + iterations);
            var finishedTask = Task.WhenAny(timeoutTask, runTask).Result;
            if (finishedTask == timeoutTask)
            {
                Console.WriteLine("TestParquetPrimitiveDataTypes test Hanging: fail to complete in 20 minutes");
                Environment.FailFast("Fail here to take memory dump");
            }
        }

        [Fact]
        public void TestParquetPrimitiveDataTypes()
        {
            string pathData = GetDataPath(@"Parquet", "alltypes.parquet");
            TestCore(pathData, false, new[] { "loader=Parquet{bigIntDates=+}" } );
            Done();
        }

        [Theory]
        [IterationData(iterations: 20)]
        [Trait("Category", "RunSpecificTest")]
        public void CompleteTestParquetNull(int iterations)
        {
            Output.WriteLine($"{iterations} - th");

            int timeout = 20 * 60 * 1000;

            var runTask = Task.Run(TestParquetNull);
            var timeoutTask = Task.Delay(timeout + iterations);
            var finishedTask = Task.WhenAny(timeoutTask, runTask).Result;
            if (finishedTask == timeoutTask)
            {
                Console.WriteLine("TestParquetNull test Hanging: fail to complete in 20 minutes");
                Environment.FailFast("Fail here to take memory dump");
            }
        }

        [Fact]
        public void TestParquetNull()
        {
            string pathData = GetDataPath(@"Parquet", "test-null.parquet");
            var ex = Assert.Throws<InvalidOperationException>(() => TestCore(pathData, false, new[] { "loader=Parquet{bigIntDates=+}" }, forceDense: true));
            Assert.Equal("Nullable object must have a value.", ex.Message);
            Done();
        }
    }
}
