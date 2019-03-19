// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using BenchmarkDotNet.Attributes;
using Microsoft.ML.Benchmarks.Harness;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

namespace Microsoft.ML.Benchmarks
{
    [CIBenchmark]
    public class HashBench
    {
        private sealed class RowImpl : DataViewRow
        {
            public long PositionValue;

            public override DataViewSchema Schema { get; }
            public override long Position => PositionValue;
            public override long Batch => 0;
            public override ValueGetter<DataViewRowId> GetIdGetter()
                => (ref DataViewRowId val) => val = new DataViewRowId((ulong)Position, 0);

            private readonly Delegate _getter;

            /// <summary>
            /// Returns whether the given column is active in this row.
            /// </summary>
            public override bool IsColumnActive(DataViewSchema.Column column)
            {
                if (column.Index != 0)
                    throw new Exception();
                return true;
            }

            /// <summary>
            /// Returns a value getter delegate to fetch the valueof column with the given columnIndex, from the row.
            /// This throws if the column is not active in this row, or if the type
            /// <typeparamref name="TValue"/> differs from this column's type.
            /// </summary>
            /// <typeparam name="TValue"> is the output column's content type.</typeparam>
            /// <param name="column"> is the index of a output column whose getter should be returned.</param>
            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
            {
                if (column.Index != 0)
                    throw new Exception();
                if (_getter is ValueGetter<TValue> typedGetter)
                    return typedGetter;
                throw new Exception();
            }

            public static RowImpl Create<T>(DataViewType type, ValueGetter<T> getter)
            {
                if (type.RawType != typeof(T))
                    throw new Exception();
                return new RowImpl(type, getter);
            }

            private RowImpl(DataViewType type, Delegate getter)
            {
                var builder = new DataViewSchema.Builder();
                builder.AddColumn("Foo", type, null);
                Schema = builder.ToSchema();
                _getter = getter;
            }
        }

        private const int Count = 100_000;

        private readonly IHostEnvironment _env = new MLContext();

        private RowImpl _inRow;
        private ValueGetter<uint> _getter;
        private ValueGetter<VBuffer<uint>> _vecGetter;

        private void InitMap<T>(T val, DataViewType type, int numberOfBits = 20, ValueGetter<T> getter = null)
        {
            if (getter == null)
                getter = (ref T dst) => dst = val;
            _inRow = RowImpl.Create(type, getter);
            // One million features is a nice, typical number.
            var info = new HashingEstimator.ColumnOptions("Bar", "Foo", numberOfBits: numberOfBits);
            var xf = new HashingTransformer(_env, new[] { info });
            var mapper = ((ITransformer)xf).GetRowToRowMapper(_inRow.Schema);
            var column = mapper.OutputSchema["Bar"];
            var outRow = mapper.GetRow(_inRow, column);
            if (type is VectorType)
                _vecGetter = outRow.GetGetter<VBuffer<uint>>(column);
            else
                _getter = outRow.GetGetter<uint>(column);
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
                ++_inRow.PositionValue;
            }
        }

        private void InitDenseVecMap<T>(T[] vals, PrimitiveDataViewType itemType, int numberOfBits = 20)
        {
            var vbuf = new VBuffer<T>(vals.Length, vals);
            InitMap(vbuf, new VectorType(itemType, vals.Length), numberOfBits, vbuf.CopyTo);
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
                ++_inRow.PositionValue;
            }
        }

        [GlobalSetup(Target = nameof(HashScalarString))]
        public void SetupHashScalarString()
        {
            InitMap("Hello".AsMemory(), TextDataViewType.Instance);
        }

        [Benchmark]
        public void HashScalarString()
        {
            RunScalar();
        }

        [GlobalSetup(Target = nameof(HashScalarFloat))]
        public void SetupHashScalarFloat()
        {
            InitMap(5.0f, NumberDataViewType.Single);
        }

        [Benchmark]
        public void HashScalarFloat()
        {
            RunScalar();
        }

        [GlobalSetup(Target = nameof(HashScalarDouble))]
        public void SetupHashScalarDouble()
        {
            InitMap(5.0, NumberDataViewType.Double);
        }

        [Benchmark]
        public void HashScalarDouble()
        {
            RunScalar();
        }

        [GlobalSetup(Target = nameof(HashScalarKey))]
        public void SetupHashScalarKey()
        {
            InitMap(6u, new KeyType(typeof(uint), 100));
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
            InitDenseVecMap(tokens, TextDataViewType.Instance);
        }

        [Benchmark]
        public void HashVectorString()
        {
            RunVector();
        }

        [GlobalSetup(Target = nameof(HashVectorFloat))]
        public void SetupHashVectorFloat()
        {
            InitDenseVecMap(new[] { 1f, 2f, 3f, 4f, 5f }, NumberDataViewType.Single);
        }

        [Benchmark]
        public void HashVectorFloat()
        {
            RunVector();
        }

        [GlobalSetup(Target = nameof(HashVectorDouble))]
        public void SetupHashVectorDouble()
        {
            InitDenseVecMap(new[] { 1d, 2d, 3d, 4d, 5d }, NumberDataViewType.Double);
        }

        [Benchmark]
        public void HashVectorDouble()
        {
            RunVector();
        }

        [GlobalSetup(Target = nameof(HashVectorKey))]
        public void SetupHashVectorKey()
        {
            InitDenseVecMap(new[] { 1u, 2u, 0u, 4u, 5u }, new KeyType(typeof(uint), 100));
        }

        [Benchmark]
        public void HashVectorKey()
        {
            RunVector();
        }
    }
}
