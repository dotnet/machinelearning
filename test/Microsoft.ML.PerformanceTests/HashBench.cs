// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using BenchmarkDotNet.Attributes;
using Microsoft.ML.Data;
using Microsoft.ML.PerformanceTests.Harness;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

namespace Microsoft.ML.PerformanceTests
{
    [CIBenchmark]
    public class HashBench : BenchmarkBase
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

        private readonly MLContext _env = new MLContext(1);

        private RowImpl _inRow;
        private ValueGetter<uint> _getter;
        private ValueGetter<VBuffer<uint>> _vecGetter;

        private void InitMapMurmurHashV2<T>(T val, DataViewType type, int numberOfBits = 20, ValueGetter<T> getter = null)
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
            if (type is VectorDataViewType)
                _vecGetter = outRow.GetGetter<VBuffer<uint>>(column);
            else
                _getter = outRow.GetGetter<uint>(column);
        }

        private void InitMapMurmurHashV1<T>(T val, DataViewType type, ValueGetter<T> getter = null)
        {
            if (getter == null)
                getter = (ref T dst) => dst = val;
            _inRow = RowImpl.Create(type, getter);

            var modelPath = GetBenchmarkDataPathAndEnsureData("backcompat/MurmurHashV1.zip");
            var estimator = _env.Model.Load(modelPath, out var schema);
            var mapper = ((ITransformer)estimator).GetRowToRowMapper(_inRow.Schema);
            var column = mapper.OutputSchema["Bar"];
            var outRow = mapper.GetRow(_inRow, column);
            if (type is VectorDataViewType)
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
            InitMapMurmurHashV2(vbuf, new VectorDataViewType(itemType, vals.Length), numberOfBits, vbuf.CopyTo);
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
            InitMapMurmurHashV2("Hello".AsMemory(), TextDataViewType.Instance);
        }

        [Benchmark]
        public void HashScalarString()
        {
            RunScalar();
        }

        [GlobalSetup(Target = nameof(HashScalarFloat))]
        public void SetupHashScalarFloat()
        {
            InitMapMurmurHashV2(5.0f, NumberDataViewType.Single);
        }

        [Benchmark]
        public void HashScalarFloat()
        {
            RunScalar();
        }

        [GlobalSetup(Target = nameof(HashScalarDouble))]
        public void SetupHashScalarDouble()
        {
            InitMapMurmurHashV2(5.0, NumberDataViewType.Double);
        }

        [Benchmark]
        public void HashScalarDouble()
        {
            RunScalar();
        }

        [GlobalSetup(Target = nameof(HashScalarKey))]
        public void SetupHashScalarKey()
        {
            InitMapMurmurHashV2(6u, new KeyDataViewType(typeof(uint), 100));
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
            InitDenseVecMap(new[] { 1u, 2u, 0u, 4u, 5u }, new KeyDataViewType(typeof(uint), 100));
        }

        [Benchmark]
        public void HashVectorKey()
        {
            RunVector();
        }

        //The benchmarks below use a model that uses MurmurHash Version 1
        [GlobalSetup(Target = nameof(HashScalarStringV1))]
        public void SetupHashScalarStringV1()
        {
            InitMapMurmurHashV1("Hello".AsMemory(), TextDataViewType.Instance);
        }

        [Benchmark]
        public void HashScalarStringV1()
        {
            RunScalar();
        }

        [GlobalSetup(Target = nameof(HashScalarKeyV1))]
        public void SetupHashScalarKeyV1()
        {
            InitMapMurmurHashV1(6u, new KeyDataViewType(typeof(uint), 100));
        }

        [Benchmark]
        public void HashScalarKeyV1()
        {
            RunScalar();
        }

        [GlobalSetup(Target = nameof(HashScalarFloatV1))]
        public void SetupHashScalarFloatV1()
        {
            InitMapMurmurHashV1(5.0f, NumberDataViewType.Single);
        }

        [Benchmark]
        public void HashScalarFloatV1()
        {
            RunScalar();
        }

        // * Summary *

        // BenchmarkDotNet=v0.12.0, OS=Windows 10.0.18363
        // Intel Core i7-7700 CPU 3.60GHz(Kaby Lake), 1 CPU, 8 logical and 4 physical cores
        // .NET Core SDK = 3.1.200

        // [Host]     : .NET Core 2.1.16 (CoreCLR 4.6.28516.03, CoreFX 4.6.28516.10), X64 RyuJIT
        // Job-FUEYRN : .NET Core 2.1.16 (CoreCLR 4.6.28516.03, CoreFX 4.6.28516.10), X64 RyuJIT

        // Arguments=/p:Configuration=Release Toolchain = netcoreapp2.1  MaxIterationCount=20
        // WarmupCount=1

        // |              Method |        Mean |     Error |    StdDev | Extra Metric |
        // |------------------- |------------:|----------:|----------:|-------------:|
        // |   HashScalarString |  3,232.4 us |  49.93 us |  46.71 us |            - |
        // |    HashScalarFloat |    675.3 us |   3.22 us |   2.86 us |            - |
        // |   HashScalarDouble |    824.2 us |   3.32 us |   3.10 us |            - |
        // |      HashScalarKey |    570.8 us |   2.77 us |   2.46 us |            - |
        // |   HashVectorString | 25,937.6 us | 621.62 us | 715.85 us |            - |
        // |    HashVectorFloat | 13,825.1 us |  74.14 us |  69.35 us |            - |
        // |   HashVectorDouble | 14,616.9 us |  43.29 us |  36.15 us |            - |
        // |      HashVectorKey | 13,096.8 us |  56.62 us |  52.96 us |            - |
        // | HashScalarStringV1 |  3,154.5 us |  27.44 us |  25.67 us |            - |
        // |   HashScalarUIntV1 |    546.2 us |   2.35 us |   2.08 us |            - |
        // |  HashScalarFloatV1 |    670.7 us |   3.60 us |   3.36 us |            - |
    }
}
