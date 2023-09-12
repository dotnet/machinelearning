// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BenchmarkDotNet.Attributes;

namespace Microsoft.Data.Analysis.PerformanceTests
{
    public class PerformanceTests
    {
        private const int ItemsCount = 1000000;
        private Int32DataFrameColumn _column1;
        private Int32DataFrameColumn _column2;

        [GlobalSetup]
        public void SetUp()
        {
            var values = Enumerable.Range(0, ItemsCount);
            _column1 = new Int32DataFrameColumn("Column1", values);
            _column2 = new Int32DataFrameColumn("Column2", values);
        }

        [Benchmark]
        public void Sum()
        {
            var column = _column1 + _column2;
        }
    }
}
