// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using BenchmarkDotNet.Attributes;
using Microsoft.ML.Benchmarks.Harness;
using Microsoft.ML.Data;

namespace Microsoft.ML.Benchmarks
{
    [CIBenchmark]
    public class CacheDataViewBench
    {
        private const int Length = 100000;

        // Global.
        private IDataView _cacheDataView;
        private DataViewRowCursor _cursor;
        private ValueGetter<int> _seekerGetter;
        private ValueGetter<int> _cursorGetter;
        private DataViewSchema.Column _col;

        private RowSeeker _seeker;
        private long[] _positions;

        [GlobalSetup(Targets = new[] { nameof(CacheWithCursor), nameof(CacheWithSeeker) })]
        public void Setup()
        {
            var ctx = new MLContext();
            var builder = new ArrayDataViewBuilder(ctx);
            int[] values = new int[Length];
            for (int i = 0; i < values.Length; ++i)
                values[i] = i;
            builder.AddColumn("A", NumberDataViewType.Int32, values);
            var dv = builder.GetDataView();
            var cacheDv = ctx.Data.Cache(dv);

            var col = cacheDv.Schema.GetColumnOrNull("A").Value;
            // First do one pass through.
            using (var cursor = cacheDv.GetRowCursor(col))
            {
                var getter = cursor.GetGetter<int>(col);
                int val = 0;
                int count = 0;
                while (cursor.MoveNext())
                {
                    getter(ref val);
                    if (val != cursor.Position)
                        throw new Exception($"Unexpected value {val} at {cursor.Position}");
                    count++;
                }
                if (count != Length)
                    throw new Exception($"Expected {Length} values in cache but only saw {count}");
            }
            _cacheDataView = cacheDv;

            // Only needed for seeker, but may as well set it.
            _positions = new long[Length];
            var rand = new Random(0);
            for (int i = 0; i < _positions.Length; ++i)
                _positions[i] = rand.Next(Length);

            _col = _cacheDataView.Schema["A"];
            _seeker = ((IRowSeekable)_cacheDataView).GetSeeker(colIndex => colIndex == _col.Index);
            _seekerGetter = _seeker.GetGetter<int>(_col);
        }

        [Benchmark]
        public void CacheWithCursor()
        {
           // This setup takes very less time to execute as compared to the actual _cursorGetter.
            // The most preferable position for this setup will be in GlobalSetup.
            _cursor = _cacheDataView.GetRowCursor(_col);
            _cursorGetter = _cursor.GetGetter<int>(_col);

            int val = 0;
            while (_cursor.MoveNext())
                _cursorGetter(ref val); 
        }

        [Benchmark]
        public void CacheWithSeeker()
        {
            int val = 0;
            foreach (long pos in _positions)
            {
                _seeker.MoveTo(pos);
                _seekerGetter(ref val);
            }
        }
    }
}
