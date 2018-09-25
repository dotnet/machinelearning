// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using System;
using Xunit;

namespace Microsoft.ML.Runtime.RunTests
{
    public sealed partial class TestParquet : TestDataPipeBase
    {
        protected override void InitializeCore()
        {
            base.InitializeCore();
            Env.ComponentCatalog.RegisterAssembly(typeof(ParquetLoader).Assembly);
        }

        [Fact]
        public void TestParquetPrimitiveDataTypes()
        {
            string pathData = GetDataPath(@"Parquet", "alltypes.parquet");
            TestCore(pathData, false, new[] { "loader=Parquet{bigIntDates=+}" } );
            Done();
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
