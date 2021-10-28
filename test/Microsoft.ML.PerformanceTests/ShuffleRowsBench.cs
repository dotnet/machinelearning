// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Attributes;
using Microsoft.ML.Data;
using Microsoft.ML.PerformanceTests.Harness;

namespace Microsoft.ML.PerformanceTests
{
    [CIBenchmark]
    public class ShuffleRowsBench : BenchmarkBase
    {
        private TrainRow[] _rows;
        private MLContext _context;

        [GlobalSetup]
        public void Setup()
        {
            _rows = new TrainRow[10_000];
            for (var i = 0; i < _rows.Length; i++)
            {
                _rows[i] = new TrainRow() { Sample = i.ToString(), Week = i, Label = i / 2 };
            }

            _context = new MLContext();
        }

        [Benchmark]
        public void ShuffleRows()
        {
            IDataView data = _context.Data.LoadFromEnumerable(_rows);

            IDataView shuffledData = _context.Data.ShuffleRows(data, seed: 0);

            foreach (string sample in shuffledData.GetColumn<string>("Sample"))
            {
            }
        }

        private class TrainRow
        {
            [ColumnName("Sample")]
            public string Sample;

            [ColumnName("Week")]
            public float Week;

            [ColumnName("Label")]
            public float Label;
        }
    }
}
