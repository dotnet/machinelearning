// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Attributes;
using Microsoft.ML.Runtime.Data;
using System;

namespace Microsoft.ML.Benchmarks
{
    public class CacheDataViewBench
    {
        private const int Length = 100000;

        // Global.
        private IDataView _cacheDataView;
        // Per iteration.
        private RowCursor _cursor;
        private ValueGetter<int> _getter;

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
            builder.AddColumn("A", NumberType.I4, values);
            var dv = builder.GetDataView();
            var cacheDv = ctx.Data.Cache(dv);

            var col = cacheDv.Schema.GetColumnOrNull("A").Value;
            // First do one pass through.
            using (var cursor = cacheDv.GetRowCursor(colIndex => colIndex == col.Index))
            {
                var getter = cursor.GetGetter<int>(col.Index);
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
        }

        [IterationSetup(Target = nameof(CacheWithCursor))]
        public void CacheWithCursorSetup()
        {
            var col = _cacheDataView.Schema.GetColumnOrNull("A").Value;
            _cursor = _cacheDataView.GetRowCursor(colIndex => colIndex == col.Index);
            _getter = _cursor.GetGetter<int>(col.Index);
        }

        [Benchmark]
        public void CacheWithCursor()
        {
            int val = 0;
            while (_cursor.MoveNext())
                _getter(ref val);
        }

        [IterationSetup(Target = nameof(CacheWithSeeker))]
        public void CacheWithSeekerSetup()
        {
            var col = _cacheDataView.Schema.GetColumnOrNull("A").Value;
            _seeker = ((IRowSeekable)_cacheDataView).GetSeeker(colIndex => colIndex == col.Index);
            _getter = _seeker.GetGetter<int>(col.Index);
        }

        [Benchmark]
        public void CacheWithSeeker()
        {
            int val = 0;
            foreach (long pos in _positions)
            {
                _seeker.MoveTo(pos);
                _getter(ref val);
            }
        }
    }
}
