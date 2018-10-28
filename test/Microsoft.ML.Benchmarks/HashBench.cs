// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using BenchmarkDotNet.Attributes;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Learners;
using System.Linq;

namespace Microsoft.ML.Benchmarks
{
    public class HashBench
    {
        private sealed class Counted : ICounted
        {
            public long Position { get; set; }

            public long Batch => 0;

            public ValueGetter<UInt128> GetIdGetter()
                => (ref UInt128 val) => val = new UInt128((ulong)Position, 0);
        }

        private const int Count = 100_000;

        private readonly IHostEnvironment _env = new LocalEnvironment();

        private Counted _counted;
        private ValueGetter<uint> _getter;
        private ValueGetter<VBuffer<uint>> _vecGetter;

        private void InitMap<T>(T val, ColumnType type, int hashBits = 20)
        {
            var col = RowColumnUtils.GetColumn("Foo", type, ref val);
            _counted = new Counted();
            var inRow = RowColumnUtils.GetRow(_counted, col);
            // One million features is a nice, typical number.
            var info = new HashTransformer.ColumnInfo("Foo", "Bar", hashBits: hashBits);
            var xf = new HashTransformer(_env, new[] { info });
            var mapper = xf.GetRowToRowMapper(inRow.Schema);
            mapper.Schema.TryGetColumnIndex("Bar", out int outCol);
            var outRow = mapper.GetRow(inRow, c => c == outCol, out var _);
            if (type.IsVector)
                _vecGetter = outRow.GetGetter<VBuffer<uint>>(outCol);
            else
                _getter = outRow.GetGetter<uint>(outCol);
        }

        /// <summary>
        /// All the scalar mappers have the same output type.
        /// </summary>
        private void RunScalar()
        {
            uint val = default;
            for (int i = 0; i < Count; ++i)
            {
                _getter(ref val);
                ++_counted.Position;
            }
        }

        private void InitDenseVecMap<T>(T[] vals, PrimitiveType itemType, int hashBits = 20)
        {
            var vbuf = new VBuffer<T>(vals.Length, vals);
            InitMap(vbuf, new VectorType(itemType, vals.Length), hashBits);
        }

        /// <summary>
        /// All the vector mappers have the same output type.
        /// </summary>
        private void RunVector()
        {
            VBuffer<uint> val = default;
            for (int i = 0; i < Count; ++i)
            {
                _vecGetter(ref val);
                ++_counted.Position;
            }
        }

        [GlobalSetup(Target = nameof(HashScalarString))]
        public void SetupHashScalarString()
        {
            InitMap("Hello".AsMemory(), TextType.Instance);
        }

        [Benchmark]
        public void HashScalarString()
        {
            RunScalar();
        }

        [GlobalSetup(Target = nameof(HashScalarFloat))]
        public void SetupHashScalarFloat()
        {
            InitMap(5.0f, NumberType.R4);
        }

        [Benchmark]
        public void HashScalarFloat()
        {
            RunScalar();
        }

        [GlobalSetup(Target = nameof(HashScalarDouble))]
        public void SetupHashScalarDouble()
        {
            InitMap(5.0, NumberType.R8);
        }

        [Benchmark]
        public void HashScalarDouble()
        {
            RunScalar();
        }

        [GlobalSetup(Target = nameof(HashScalarKey))]
        public void SetupHashScalarKey()
        {
            InitMap(6u, new KeyType(DataKind.U4, 0, 100));
        }

        [Benchmark]
        public void HashScalarKey()
        {
            RunScalar();
        }



        [GlobalSetup(Target = nameof(HashVectorString))]
        public void SetupHashVectorString()
        {
            var tokens = "Hello my friend, stay awhile and listen! ".Split().Select(token => token.AsMemory()).ToArray();
            InitDenseVecMap(tokens, TextType.Instance);
        }

        [Benchmark]
        public void HashVectorString()
        {
            RunVector();
        }

        [GlobalSetup(Target = nameof(HashVectorFloat))]
        public void SetupHashVectorFloat()
        {
            InitDenseVecMap(new[] { 1f, 2f, 3f, 4f, 5f }, NumberType.R4);
        }

        [Benchmark]
        public void HashVectorFloat()
        {
            RunVector();
        }

        [GlobalSetup(Target = nameof(HashVectorDouble))]
        public void SetupHashVectorDouble()
        {
            InitDenseVecMap(new[] { 1d, 2d, 3d, 4d, 5d }, NumberType.R8);
        }

        [Benchmark]
        public void HashVectorDouble()
        {
            RunVector();
        }

        [GlobalSetup(Target = nameof(HashVectorKey))]
        public void SetupHashVectorKey()
        {
            InitDenseVecMap(new[] { 1u, 2u, 0u, 4u, 5u }, new KeyType(DataKind.U4, 0, 100));
        }

        [Benchmark]
        public void HashVectorKey()
        {
            RunVector();
        }
    }
}
