// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.Tools;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Conversions;
using System;
using System.IO;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public class HashTests : TestDataPipeBase
    {
        public HashTests(ITestOutputHelper output) : base(output)
        {
        }

        private class TestClass
        {
            public float A;
            public float B;
            public float C;
        }

        private class TestMeta
        {
            [VectorType(2)]
            public float[] A;
            public float B;
            [VectorType(2)]
            public double[] C;
            public double D;
        }

        [Fact]
        public void HashWorkout()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };

            var dataView = ComponentCreation.CreateDataView(Env, data);
            var pipe = new HashEstimator(Env, new[]{
                    new HashTransformer.ColumnInfo("A", "HashA", hashBits:4, invertHash:-1),
                    new HashTransformer.ColumnInfo("B", "HashB", hashBits:3, ordered:true),
                    new HashTransformer.ColumnInfo("C", "HashC", seed:42),
                    new HashTransformer.ColumnInfo("A", "HashD"),
                });

            TestEstimatorCore(pipe, dataView);
            Done();
        }

        [Fact]
        public void TestMetadata()
        {

            var data = new[] {
                new TestMeta() { A=new float[2] { 3.5f, 2.5f}, B=1, C= new double[2] { 5.1f, 6.1f}, D= 7},
                new TestMeta() { A=new float[2] { 3.5f, 2.5f}, B=1, C= new double[2] { 5.1f, 6.1f}, D= 7},
                new TestMeta() { A=new float[2] { 3.5f, 2.5f}, B=1, C= new double[2] { 5.1f, 6.1f}, D= 7}};


            var dataView = ComponentCreation.CreateDataView(Env, data);
            var pipe = new HashEstimator(Env, new[] {
                new HashTransformer.ColumnInfo("A", "HashA", invertHash:1, hashBits:10),
                new HashTransformer.ColumnInfo("A", "HashAUnlim", invertHash:-1, hashBits:10),
                new HashTransformer.ColumnInfo("A", "HashAUnlimOrdered", invertHash:-1, hashBits:10, ordered:true)
            });
            var result = pipe.Fit(dataView).Transform(dataView);
            ValidateMetadata(result);
            Done();
        }

        private void ValidateMetadata(IDataView result)
        {

            Assert.True(result.Schema.TryGetColumnIndex("HashA", out int HashA));
            Assert.True(result.Schema.TryGetColumnIndex("HashAUnlim", out int HashAUnlim));
            Assert.True(result.Schema.TryGetColumnIndex("HashAUnlimOrdered", out int HashAUnlimOrdered));
            VBuffer<ReadOnlyMemory<char>> keys = default;
            var types = result.Schema.GetMetadataTypes(HashA);
            Assert.Equal(types.Select(x => x.Key), new string[1] { MetadataUtils.Kinds.KeyValues });
            result.Schema.GetMetadata(MetadataUtils.Kinds.KeyValues, HashA, ref keys);
            Assert.True(keys.Length == 1024);
            //REVIEW: This is weird. I specified invertHash to 1 so I expect only one value to be in key values, but i got two.
            Assert.Equal(keys.Items().Select(x => x.Value.ToString()), new string[2] { "2.5", "3.5" });

            types = result.Schema.GetMetadataTypes(HashAUnlim);
            Assert.Equal(types.Select(x => x.Key), new string[1] { MetadataUtils.Kinds.KeyValues });
            result.Schema.GetMetadata(MetadataUtils.Kinds.KeyValues, HashA, ref keys);
            Assert.True(keys.Length == 1024);
            Assert.Equal(keys.Items().Select(x => x.Value.ToString()), new string[2] { "2.5", "3.5" });

            types = result.Schema.GetMetadataTypes(HashAUnlimOrdered);
            Assert.Equal(types.Select(x => x.Key), new string[1] { MetadataUtils.Kinds.KeyValues });
            result.Schema.GetMetadata(MetadataUtils.Kinds.KeyValues, HashA, ref keys);
            Assert.True(keys.Length == 1024);
            Assert.Equal(keys.Items().Select(x => x.Value.ToString()), new string[2] { "2.5", "3.5" });
        }

        [Fact]
        public void TestCommandLine()
        {
            Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0} xf=Hash{col=B:A} in=f:\2.txt" }), (int)0);
        }

        [Fact]
        public void TestOldSavingAndLoading()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var dataView = ComponentCreation.CreateDataView(Env, data);
            var pipe = new HashEstimator(Env, new[]{
                    new HashTransformer.ColumnInfo("A", "HashA", hashBits:4, invertHash:-1),
                    new HashTransformer.ColumnInfo("B", "HashB", hashBits:3, ordered:true),
                    new HashTransformer.ColumnInfo("C", "HashC", seed:42),
                    new HashTransformer.ColumnInfo("A", "HashD"),
            });
            var result = pipe.Fit(dataView).Transform(dataView);
            var resultRoles = new RoleMappedData(result);
            using (var ms = new MemoryStream())
            {
                TrainUtils.SaveModel(Env, Env.Start("saving"), ms, null, resultRoles);
                ms.Position = 0;
                var loadedView = ModelFileUtils.LoadTransforms(Env, dataView, ms);
            }
        }

        private sealed class Counted : ICounted
        {
            public long Position => 0;
            public long Batch => 0;
            public ValueGetter<UInt128> GetIdGetter() => (ref UInt128 val) => val = default;
        }

        private void HashTestCore<T>(T val, PrimitiveType type, uint expected, uint expectedOrdered, uint expectedOrdered3)
        {
            const int bits = 10;

            var col = RowColumnUtils.GetColumn("Foo", type, ref val);
            var inRow = RowColumnUtils.GetRow(new Counted(), col);

            // First do an unordered hash.
            var info = new HashTransformer.ColumnInfo("Foo", "Bar", hashBits: bits);
            var xf = new HashTransformer(Env, new[] { info });
            var mapper = xf.GetRowToRowMapper(inRow.Schema);
            mapper.Schema.TryGetColumnIndex("Bar", out int outCol);
            var outRow = mapper.GetRow(inRow, c => c == outCol, out var _);

            var getter = outRow.GetGetter<uint>(outCol);
            uint result = 0;
            getter(ref result);
            Assert.Equal(expected, result);

            // Next do an ordered hash.
            info = new HashTransformer.ColumnInfo("Foo", "Bar", hashBits: bits, ordered: true);
            xf = new HashTransformer(Env, new[] { info });
            mapper = xf.GetRowToRowMapper(inRow.Schema);
            mapper.Schema.TryGetColumnIndex("Bar", out outCol);
            outRow = mapper.GetRow(inRow, c => c == outCol, out var _);

            getter = outRow.GetGetter<uint>(outCol);
            getter(ref result);
            Assert.Equal(expectedOrdered, result);

            // Next build up a vector to make sure that hashing is consistent between scalar values
            // at least in the first position, and in the unordered case, the last position.
            const int vecLen = 5;
            var denseVec = new VBuffer<T>(vecLen, Utils.CreateArray(vecLen, val));
            col = RowColumnUtils.GetColumn("Foo", new VectorType(type, vecLen), ref denseVec);
            inRow = RowColumnUtils.GetRow(new Counted(), col);

            info = new HashTransformer.ColumnInfo("Foo", "Bar", hashBits: bits, ordered: false);
            xf = new HashTransformer(Env, new[] { info });
            mapper = xf.GetRowToRowMapper(inRow.Schema);
            mapper.Schema.TryGetColumnIndex("Bar", out outCol);
            outRow = mapper.GetRow(inRow, c => c == outCol, out var _);

            var vecGetter = outRow.GetGetter<VBuffer<uint>>(outCol);
            VBuffer<uint> vecResult = default;
            vecGetter(ref vecResult);

            Assert.Equal(vecLen, vecResult.Length);
            // They all should equal this in this case.
            Assert.All(vecResult.DenseValues(), v => Assert.Equal(expected, v));

            // Now do ordered with the dense vector.
            info = new HashTransformer.ColumnInfo("Foo", "Bar", hashBits: bits, ordered: true);
            xf = new HashTransformer(Env, new[] { info });
            mapper = xf.GetRowToRowMapper(inRow.Schema);
            mapper.Schema.TryGetColumnIndex("Bar", out outCol);
            outRow = mapper.GetRow(inRow, c => c == outCol, out var _);
            vecGetter = outRow.GetGetter<VBuffer<uint>>(outCol);
            vecGetter(ref vecResult);

            Assert.Equal(vecLen, vecResult.Length);
            Assert.Equal(expectedOrdered, vecResult.GetItemOrDefault(0));
            Assert.Equal(expectedOrdered3, vecResult.GetItemOrDefault(3));
            Assert.All(vecResult.DenseValues(), v => Assert.True((v == 0) == (expectedOrdered == 0)));

            // Let's now do a sparse vector.
            var sparseVec = new VBuffer<T>(10, 3, Utils.CreateArray(3, val), new[] { 0, 3, 7 });
            col = RowColumnUtils.GetColumn("Foo", new VectorType(type, vecLen), ref sparseVec);
            inRow = RowColumnUtils.GetRow(new Counted(), col);

            info = new HashTransformer.ColumnInfo("Foo", "Bar", hashBits: bits, ordered: false);
            xf = new HashTransformer(Env, new[] { info });
            mapper = xf.GetRowToRowMapper(inRow.Schema);
            mapper.Schema.TryGetColumnIndex("Bar", out outCol);
            outRow = mapper.GetRow(inRow, c => c == outCol, out var _);
            vecGetter = outRow.GetGetter<VBuffer<uint>>(outCol);
            vecGetter(ref vecResult);

            Assert.Equal(10, vecResult.Length);
            Assert.Equal(expected, vecResult.GetItemOrDefault(0));
            Assert.Equal(expected, vecResult.GetItemOrDefault(3));
            Assert.Equal(expected, vecResult.GetItemOrDefault(7));

            info = new HashTransformer.ColumnInfo("Foo", "Bar", hashBits: bits, ordered: true);
            xf = new HashTransformer(Env, new[] { info });
            mapper = xf.GetRowToRowMapper(inRow.Schema);
            mapper.Schema.TryGetColumnIndex("Bar", out outCol);
            outRow = mapper.GetRow(inRow, c => c == outCol, out var _);
            vecGetter = outRow.GetGetter<VBuffer<uint>>(outCol);
            vecGetter(ref vecResult);

            Assert.Equal(10, vecResult.Length);
            Assert.Equal(expectedOrdered, vecResult.GetItemOrDefault(0));
            Assert.Equal(expectedOrdered3, vecResult.GetItemOrDefault(3));
        }

        private void HashTestPositiveIntegerCore(ulong value, uint expected, uint expectedOrdered, uint expectedOrdered3)
        {
            uint eKey = value == 0 ? 0 : expected;
            uint eoKey = value == 0 ? 0 : expectedOrdered;
            uint e3Key = value == 0 ? 0 : expectedOrdered3;

            if (value <= byte.MaxValue)
            {
                HashTestCore((byte)value, NumberType.U1, expected, expectedOrdered, expectedOrdered3);
                HashTestCore((byte)value, new KeyType(DataKind.U1, 0, byte.MaxValue - 1), eKey, eoKey, e3Key);
            }
            if (value <= ushort.MaxValue)
            {
                HashTestCore((ushort)value, NumberType.U2, expected, expectedOrdered, expectedOrdered3);
                HashTestCore((ushort)value, new KeyType(DataKind.U2, 0, ushort.MaxValue - 1), eKey, eoKey, e3Key);
            }
            if (value <= uint.MaxValue)
            {
                HashTestCore((uint)value, NumberType.U4, expected, expectedOrdered, expectedOrdered3);
                HashTestCore((uint)value, new KeyType(DataKind.U4, 0, int.MaxValue - 1), eKey, eoKey, e3Key);
            }
            HashTestCore(value, NumberType.U8, expected, expectedOrdered, expectedOrdered3);
            HashTestCore((ulong)value, new KeyType(DataKind.U8, 0, 0), eKey, eoKey, e3Key);

            HashTestCore(new UInt128(value, 0), NumberType.UG, expected, expectedOrdered, expectedOrdered3);

            // Next let's check signed numbers.

            if (value <= (ulong)sbyte.MaxValue)
                HashTestCore((sbyte)value, NumberType.I1, expected, expectedOrdered, expectedOrdered3);
            if (value <= (ulong)short.MaxValue)
                HashTestCore((short)value, NumberType.I2, expected, expectedOrdered, expectedOrdered3);
            if (value <= int.MaxValue)
                HashTestCore((int)value, NumberType.I4, expected, expectedOrdered, expectedOrdered3);
            if (value <= long.MaxValue)
                HashTestCore((long)value, NumberType.I8, expected, expectedOrdered, expectedOrdered3);
        }

        [Fact]
        public void TestHashIntegerNumbers()
        {
            HashTestPositiveIntegerCore(0, 848, 567, 518);
            HashTestPositiveIntegerCore(1, 492, 523, 1013);
            HashTestPositiveIntegerCore(2, 676, 512, 863);
        }

        [Fact]
        public void TestHashString()
        {
            HashTestCore("".AsMemory(), TextType.Instance, 0, 0, 0);
            HashTestCore("hello".AsMemory(), TextType.Instance, 326, 636, 307);
        }

        [Fact]
        public void TestHashFloatingPointNumbers()
        {
            HashTestCore(1f, NumberType.R4, 933, 67, 270);
            HashTestCore(-1f, NumberType.R4, 505, 589, 245);
            HashTestCore(0f, NumberType.R4, 848, 567, 518);
            // Note that while we have the hash for numeric types be equal, the same is not necessarily the case for floating point numbers.
            HashTestCore(1d, NumberType.R8, 671, 728, 123);
            HashTestCore(-1d, NumberType.R8, 803, 699, 790);
            HashTestCore(0d, NumberType.R8, 848, 567, 518);
        }

        [Fact]
        public void TestHashBool()
        {
            // These are the same for the hashes of 0 and 1.
            HashTestCore(false, BoolType.Instance, 848, 567, 518);
            HashTestCore(true, BoolType.Instance, 492, 523, 1013);
        }
    }
}
