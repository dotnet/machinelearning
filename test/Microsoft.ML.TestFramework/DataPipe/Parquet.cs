// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.TextAnalytics;
using Xunit;
using System.Runtime.InteropServices;

namespace Microsoft.ML.Runtime.RunTests
{
    public sealed partial class TestParquet : TestDataPipeBase
    {

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
            TestCore(pathData, false, new[] { "loader=Parquet{bigIntDates=+}" }, forceDense: true);
            Done();
        }
    }
}
