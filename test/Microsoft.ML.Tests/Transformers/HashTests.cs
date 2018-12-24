// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.Tools;
using Microsoft.ML.Transforms.Conversions;
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
            var pipe = new HashingEstimator(Env, new[]{
                    new HashingTransformer.ColumnInfo("A", "HashA", hashBits:4, invertHash:-1),
                    new HashingTransformer.ColumnInfo("B", "HashB", hashBits:3, ordered:true),
                    new HashingTransformer.ColumnInfo("C", "HashC", seed:42),
                    new HashingTransformer.ColumnInfo("A", "HashD"),
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
            var pipe = new HashingEstimator(Env, new[] {
                new HashingTransformer.ColumnInfo("A", "HashA", invertHash:1, hashBits:10),
                new HashingTransformer.ColumnInfo("A", "HashAUnlim", invertHash:-1, hashBits:10),
                new HashingTransformer.ColumnInfo("A", "HashAUnlimOrdered", invertHash:-1, hashBits:10, ordered:true)
            });
            var result = pipe.Fit(dataView).Transform(dataView);
            ValidateMetadata(result);
            Done();
        }

        private void ValidateMetadata(IDataView result)
        {
            VBuffer<ReadOnlyMemory<char>> keys = default;
            var column = result.Schema["HashA"];
            Assert.Equal(column.Metadata.Schema.Single().Name, MetadataUtils.Kinds.KeyValues);
            column.Metadata.GetValue(MetadataUtils.Kinds.KeyValues, ref keys);
            Assert.Equal(keys.Items().Select(x => x.Value.ToString()), new string[2] { "2.5", "3.5" });

            column = result.Schema["HashAUnlim"];
            Assert.Equal(column.Metadata.Schema.Single().Name, MetadataUtils.Kinds.KeyValues);
            column.Metadata.GetValue(MetadataUtils.Kinds.KeyValues, ref keys);
            Assert.Equal(keys.Items().Select(x => x.Value.ToString()), new string[2] { "2.5", "3.5" });

            column = result.Schema["HashAUnlimOrdered"];
            Assert.Equal(column.Metadata.Schema.Single().Name, MetadataUtils.Kinds.KeyValues);
            column.Metadata.GetValue(MetadataUtils.Kinds.KeyValues, ref keys);
            Assert.Equal(keys.Items().Select(x => x.Value.ToString()), new string[2] { "0:3.5", "1:2.5" });
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
            var pipe = new HashingEstimator(Env, new[]{
                    new HashingTransformer.ColumnInfo("A", "HashA", hashBits:4, invertHash:-1),
                    new HashingTransformer.ColumnInfo("B", "HashB", hashBits:3, ordered:true),
                    new HashingTransformer.ColumnInfo("C", "HashC", seed:42),
                    new HashingTransformer.ColumnInfo("A", "HashD"),
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

        private void HashTestCore<T>(T val, PrimitiveType type, uint expected, uint expectedOrdered, uint expectedOrdered3)
        {
            const int bits = 10;

            var builder = new MetadataBuilder();
            builder.AddPrimitiveValue("Foo", type, val);
            var inRow = MetadataUtils.MetadataAsRow(builder.GetMetadata());

            // First do an unordered hash.
            var info = new HashingTransformer.ColumnInfo("Foo", "Bar", hashBits: bits);
            var xf = new HashingTransformer(Env, new[] { info });
            var mapper = xf.GetRowToRowMapper(inRow.Schema);
            mapper.OutputSchema.TryGetColumnIndex("Bar", out int outCol);
            var outRow = mapper.GetRow(inRow, c => c == outCol);

            var getter = outRow.GetGetter<uint>(outCol);
            uint result = 0;
            getter(ref result);
            Assert.Equal(expected, result);

            // Next do an ordered hash.
            info = new HashingTransformer.ColumnInfo("Foo", "Bar", hashBits: bits, ordered: true);
            xf = new HashingTransformer(Env, new[] { info });
            mapper = xf.GetRowToRowMapper(inRow.Schema);
            mapper.OutputSchema.TryGetColumnIndex("Bar", out outCol);
            outRow = mapper.GetRow(inRow, c => c == outCol);

            getter = outRow.GetGetter<uint>(outCol);
            getter(ref result);
            Assert.Equal(expectedOrdered, result);

            // Next build up a vector to make sure that hashing is consistent between scalar values
            // at least in the first position, and in the unordered case, the last position.
            const int vecLen = 5;
            var denseVec = new VBuffer<T>(vecLen, Utils.CreateArray(vecLen, val));
            builder = new MetadataBuilder();
            builder.Add("Foo", new VectorType(type, vecLen), (ref VBuffer<T> dst) => denseVec.CopyTo(ref dst));
            inRow = MetadataUtils.MetadataAsRow(builder.GetMetadata());

            info = new HashingTransformer.ColumnInfo("Foo", "Bar", hashBits: bits, ordered: false);
            xf = new HashingTransformer(Env, new[] { info });
            mapper = xf.GetRowToRowMapper(inRow.Schema);
            mapper.OutputSchema.TryGetColumnIndex("Bar", out outCol);
            outRow = mapper.GetRow(inRow, c => c == outCol);

            var vecGetter = outRow.GetGetter<VBuffer<uint>>(outCol);
            VBuffer<uint> vecResult = default;
            vecGetter(ref vecResult);

            Assert.Equal(vecLen, vecResult.Length);
            // They all should equal this in this case.
            Assert.All(vecResult.DenseValues(), v => Assert.Equal(expected, v));

            // Now do ordered with the dense vector.
            info = new HashingTransformer.ColumnInfo("Foo", "Bar", hashBits: bits, ordered: true);
            xf = new HashingTransformer(Env, new[] { info });
            mapper = xf.GetRowToRowMapper(inRow.Schema);
            mapper.OutputSchema.TryGetColumnIndex("Bar", out outCol);
            outRow = mapper.GetRow(inRow, c => c == outCol);
            vecGetter = outRow.GetGetter<VBuffer<uint>>(outCol);
            vecGetter(ref vecResult);

            Assert.Equal(vecLen, vecResult.Length);
            Assert.Equal(expectedOrdered, vecResult.GetItemOrDefault(0));
            Assert.Equal(expectedOrdered3, vecResult.GetItemOrDefault(3));
            Assert.All(vecResult.DenseValues(), v => Assert.True((v == 0) == (expectedOrdered == 0)));

            // Let's now do a sparse vector.
            var sparseVec = new VBuffer<T>(10, 3, Utils.CreateArray(3, val), new[] { 0, 3, 7 });
            builder = new MetadataBuilder();
            builder.Add("Foo", new VectorType(type, vecLen), (ref VBuffer<T> dst) => sparseVec.CopyTo(ref dst));
            inRow = MetadataUtils.MetadataAsRow(builder.GetMetadata());

            info = new HashingTransformer.ColumnInfo("Foo", "Bar", hashBits: bits, ordered: false);
            xf = new HashingTransformer(Env, new[] { info });
            mapper = xf.GetRowToRowMapper(inRow.Schema);
            mapper.OutputSchema.TryGetColumnIndex("Bar", out outCol);
            outRow = mapper.GetRow(inRow, c => c == outCol);
            vecGetter = outRow.GetGetter<VBuffer<uint>>(outCol);
            vecGetter(ref vecResult);

            Assert.Equal(10, vecResult.Length);
            Assert.Equal(expected, vecResult.GetItemOrDefault(0));
            Assert.Equal(expected, vecResult.GetItemOrDefault(3));
            Assert.Equal(expected, vecResult.GetItemOrDefault(7));

            info = new HashingTransformer.ColumnInfo("Foo", "Bar", hashBits: bits, ordered: true);
            xf = new HashingTransformer(Env, new[] { info });
            mapper = xf.GetRowToRowMapper(inRow.Schema);
            mapper.OutputSchema.TryGetColumnIndex("Bar", out outCol);
            outRow = mapper.GetRow(inRow, c => c == outCol);
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
                HashTestCore((byte)value, new KeyType(typeof(byte), 0, byte.MaxValue - 1), eKey, eoKey, e3Key);
            }
            if (value <= ushort.MaxValue)
            {
                HashTestCore((ushort)value, NumberType.U2, expected, expectedOrdered, expectedOrdered3);
                HashTestCore((ushort)value, new KeyType(typeof(ushort), 0, ushort.MaxValue - 1), eKey, eoKey, e3Key);
            }
            if (value <= uint.MaxValue)
            {
                HashTestCore((uint)value, NumberType.U4, expected, expectedOrdered, expectedOrdered3);
                HashTestCore((uint)value, new KeyType(typeof(uint), 0, int.MaxValue - 1), eKey, eoKey, e3Key);
            }
            HashTestCore(value, NumberType.U8, expected, expectedOrdered, expectedOrdered3);
            HashTestCore((ulong)value, new KeyType(typeof(ulong), 0, 0), eKey, eoKey, e3Key);

            HashTestCore(new RowId(value, 0), NumberType.UG, expected, expectedOrdered, expectedOrdered3);

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
