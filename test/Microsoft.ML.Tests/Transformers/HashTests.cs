// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFrameworkCommon;
using Microsoft.ML.Tools;
using Microsoft.ML.Transforms;
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

            var dataView = ML.Data.LoadFromEnumerable(data);
            var pipe = ML.Transforms.Conversion.Hash(new[]{
                    new HashingEstimator.ColumnOptions("HashA", "A", numberOfBits:4, maximumNumberOfInverts:-1),
                    new HashingEstimator.ColumnOptions("HashB", "B", numberOfBits:3, useOrderedHashing:true),
                    new HashingEstimator.ColumnOptions("HashC", "C", seed:42),
                    new HashingEstimator.ColumnOptions("HashD", "A"),
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


            var dataView = ML.Data.LoadFromEnumerable(data);
            var pipe = ML.Transforms.Conversion.Hash(new[] {
                new HashingEstimator.ColumnOptions("HashA", "A", maximumNumberOfInverts:1, numberOfBits:10),
                new HashingEstimator.ColumnOptions("HashAUnlim", "A", maximumNumberOfInverts:-1, numberOfBits:10),
                new HashingEstimator.ColumnOptions("HashAUnlimOrdered", "A", maximumNumberOfInverts:-1, numberOfBits:10, useOrderedHashing:true)
            });
            var result = pipe.Fit(dataView).Transform(dataView);
            ValidateMetadata(result);
            Done();
        }

        private void ValidateMetadata(IDataView result)
        {
            VBuffer<ReadOnlyMemory<char>> keys = default;
            var column = result.Schema["HashA"];
            Assert.Equal(column.Annotations.Schema.Single().Name, AnnotationUtils.Kinds.KeyValues);
            column.GetKeyValues(ref keys);
            Assert.Equal(keys.Items().Select(x => x.Value.ToString()), new string[2] { "2.5", "3.5" });

            column = result.Schema["HashAUnlim"];
            Assert.Equal(column.Annotations.Schema.Single().Name, AnnotationUtils.Kinds.KeyValues);
            column.GetKeyValues(ref keys);
            Assert.Equal(keys.Items().Select(x => x.Value.ToString()), new string[2] { "2.5", "3.5" });

            column = result.Schema["HashAUnlimOrdered"];
            Assert.Equal(column.Annotations.Schema.Single().Name, AnnotationUtils.Kinds.KeyValues);
            column.GetKeyValues(ref keys);
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
            var dataView = ML.Data.LoadFromEnumerable(data);
            var pipe = ML.Transforms.Conversion.Hash(new[]{
                    new HashingEstimator.ColumnOptions("HashA", "A", numberOfBits:4, maximumNumberOfInverts:-1),
                    new HashingEstimator.ColumnOptions("HashB", "B", numberOfBits:3, useOrderedHashing:true),
                    new HashingEstimator.ColumnOptions("HashC", "C", seed:42),
                    new HashingEstimator.ColumnOptions("HashD" ,"A"),
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

        private void HashTestCore<T>(T val, PrimitiveDataViewType type, uint expected, uint expectedOrdered, uint expectedOrdered3, uint expectedCombined, uint expectedCombinedSparse)
        {
            const int bits = 10;

            var builder = new DataViewSchema.Annotations.Builder();
            builder.AddPrimitiveValue("Foo", type, val);
            var inRow = AnnotationUtils.AnnotationsAsRow(builder.ToAnnotations());

            //helper
            ValueGetter<TType> hashGetter<TType>(HashingEstimator.ColumnOptions colInfo)
            {
                var xf = new HashingTransformer(Env, new[] { colInfo });
                var mapper = ((ITransformer)xf).GetRowToRowMapper(inRow.Schema);
                var col = mapper.OutputSchema["Bar"];
                var outRow = mapper.GetRow(inRow, col);

                return outRow.GetGetter<TType>(col);
            };

            // First do an unordered hash.
            var info = new HashingEstimator.ColumnOptions("Bar", "Foo", numberOfBits: bits);
            var getter = hashGetter<uint>(info);
            uint result = 0;
            getter(ref result);
            Assert.Equal(expected, result);

            // Next do an ordered hash.
            info = new HashingEstimator.ColumnOptions("Bar", "Foo", numberOfBits: bits, useOrderedHashing: true);
            getter = hashGetter<uint>(info);
            getter(ref result);
            Assert.Equal(expectedOrdered, result);

            // Next build up a vector to make sure that hashing is consistent between scalar values
            // at least in the first position, and in the unordered case, the last position.
            const int vecLen = 5;
            var denseVec = new VBuffer<T>(vecLen, Utils.CreateArray(vecLen, val));
            builder = new DataViewSchema.Annotations.Builder();
            builder.Add("Foo", new VectorDataViewType(type, vecLen), (ref VBuffer<T> dst) => denseVec.CopyTo(ref dst));
            inRow = AnnotationUtils.AnnotationsAsRow(builder.ToAnnotations());

            info = new HashingEstimator.ColumnOptions("Bar", "Foo", numberOfBits: bits, useOrderedHashing: false);
            var vecGetter = hashGetter<VBuffer<uint>>(info);
            VBuffer<uint> vecResult = default;
            vecGetter(ref vecResult);

            Assert.Equal(vecLen, vecResult.Length);
            // They all should equal this in this case.
            Assert.All(vecResult.DenseValues(), v => Assert.Equal(expected, v));

            // Now do ordered with the dense vector.
            info = new HashingEstimator.ColumnOptions("Bar", "Foo", numberOfBits: bits, useOrderedHashing: true);
            vecGetter = hashGetter<VBuffer<uint>>(info);
            vecGetter(ref vecResult);

            Assert.Equal(vecLen, vecResult.Length);
            Assert.Equal(expectedOrdered, vecResult.GetItemOrDefault(0));
            Assert.Equal(expectedOrdered3, vecResult.GetItemOrDefault(3));
            Assert.All(vecResult.DenseValues(), v => Assert.True((v == 0) == (expectedOrdered == 0)));

            // Now combine into one hash.
            info = new HashingEstimator.ColumnOptions("Bar", "Foo", numberOfBits: bits, combine: true);
            getter = hashGetter<uint>(info);
            getter(ref result);
            Assert.Equal(expectedCombined, result);

            // Let's now do a sparse vector.
            var sparseVec = new VBuffer<T>(10, 3, Utils.CreateArray(3, val), new[] { 0, 3, 7 });
            builder = new DataViewSchema.Annotations.Builder();
            builder.Add("Foo", new VectorDataViewType(type, vecLen), (ref VBuffer<T> dst) => sparseVec.CopyTo(ref dst));
            inRow = AnnotationUtils.AnnotationsAsRow(builder.ToAnnotations());

            info = new HashingEstimator.ColumnOptions("Bar", "Foo", numberOfBits: bits, useOrderedHashing: false);
            vecGetter = hashGetter<VBuffer<uint>>(info);
            vecGetter(ref vecResult);

            Assert.Equal(10, vecResult.Length);
            Assert.Equal(expected, vecResult.GetItemOrDefault(0));
            Assert.Equal(expected, vecResult.GetItemOrDefault(3));
            Assert.Equal(expected, vecResult.GetItemOrDefault(7));

            info = new HashingEstimator.ColumnOptions("Bar", "Foo", numberOfBits: bits, useOrderedHashing: true);
            vecGetter = hashGetter<VBuffer<uint>>(info);
            vecGetter(ref vecResult);

            Assert.Equal(10, vecResult.Length);
            Assert.Equal(expectedOrdered, vecResult.GetItemOrDefault(0));
            Assert.Equal(expectedOrdered3, vecResult.GetItemOrDefault(3));

            info = new HashingEstimator.ColumnOptions("Bar", "Foo", numberOfBits: bits, combine: true);
            getter = hashGetter<uint>(info);
            getter(ref result);
            Assert.Equal(expectedCombinedSparse, result);
        }

        private void HashTestPositiveIntegerCore32Bits(ulong value, uint expected, uint expectedOrdered, uint expectedOrdered3, uint expectedCombined, uint expectedCombinedSparse)
        {
            uint eKey = value == 0 ? 0 : expected;
            uint eoKey = value == 0 ? 0 : expectedOrdered;
            uint e3Key = value == 0 ? 0 : expectedOrdered3;
            uint ecKey = value == 0 ? 0 : expectedCombined;

            if (value <= byte.MaxValue)
            {
                HashTestCore((byte)value, NumberDataViewType.Byte, expected, expectedOrdered, expectedOrdered3, expectedCombined, expectedCombinedSparse);
                HashTestCore((byte)value, new KeyDataViewType(typeof(byte), byte.MaxValue - 1), eKey, eoKey, e3Key, ecKey, 0);
            }
            if (value <= ushort.MaxValue)
            {
                HashTestCore((ushort)value, NumberDataViewType.UInt16, expected, expectedOrdered, expectedOrdered3, expectedCombined, expectedCombinedSparse);
                HashTestCore((ushort)value, new KeyDataViewType(typeof(ushort), ushort.MaxValue - 1), eKey, eoKey, e3Key, ecKey, 0);
            }
            if (value <= uint.MaxValue)
            {
                HashTestCore((uint)value, NumberDataViewType.UInt32, expected, expectedOrdered, expectedOrdered3, expectedCombined, expectedCombinedSparse);
                HashTestCore((uint)value, new KeyDataViewType(typeof(uint), int.MaxValue - 1), eKey, eoKey, e3Key, ecKey, 0);
            }

            // Next let's check signed numbers.
            if (value <= (ulong)sbyte.MaxValue)
                HashTestCore((sbyte)value, NumberDataViewType.SByte, expected, expectedOrdered, expectedOrdered3, expectedCombined, expectedCombinedSparse);
            if (value <= (ulong)short.MaxValue)
                HashTestCore((short)value, NumberDataViewType.Int16, expected, expectedOrdered, expectedOrdered3, expectedCombined, expectedCombinedSparse);
            if (value <= int.MaxValue)
                HashTestCore((int)value, NumberDataViewType.Int32, expected, expectedOrdered, expectedOrdered3, expectedCombined, expectedCombinedSparse);
        }

        private void HashTestPositiveIntegerCore64Bits(ulong value, uint expected, uint expectedOrdered, uint expectedOrdered3, uint expectedCombined, uint expectedCombinedSparse)
        {
            uint eKey = value == 0 ? 0 : expected;
            uint eoKey = value == 0 ? 0 : expectedOrdered;
            uint e3Key = value == 0 ? 0 : expectedOrdered3;
            uint ecKey = value == 0 ? 0 : expectedCombined;

            HashTestCore(value, NumberDataViewType.UInt64, expected, expectedOrdered, expectedOrdered3, expectedCombined, expectedCombinedSparse);

            // Next let's check signed numbers.
            if (value <= long.MaxValue)
                HashTestCore((long)value, NumberDataViewType.Int64, expected, expectedOrdered, expectedOrdered3, expectedCombined, expectedCombinedSparse);

            // ulong keys
            HashTestCore(value, new KeyDataViewType(typeof(ulong), int.MaxValue - 1), eKey, eoKey, e3Key, ecKey, 0);
        }

        private void HashTestPositiveIntegerCore128Bits(ulong value, uint expected, uint expectedOrdered, uint expectedOrdered3, uint expectedCombined, uint expectedCombinedSparse)
        {
            HashTestCore(new DataViewRowId(value, 0), RowIdDataViewType.Instance, expected, expectedOrdered, expectedOrdered3, expectedCombined, expectedCombinedSparse);
        }

        [Fact]
        public void TestHashIntegerNumbers()
        {
            HashTestPositiveIntegerCore32Bits(0, 842, 358, 20, 429, 333);
            HashTestPositiveIntegerCore32Bits(1, 502, 537, 746, 847, 711);
            HashTestPositiveIntegerCore32Bits(2, 407, 801, 652, 727, 462);

            HashTestPositiveIntegerCore64Bits(0, 512, 851, 795, 333, 113);
            HashTestPositiveIntegerCore64Bits(1, 329, 190, 574, 880, 471);
            HashTestPositiveIntegerCore64Bits(2, 484, 713, 128, 95, 9);

            HashTestPositiveIntegerCore128Bits(0, 362, 161, 115, 429, 333);
            HashTestPositiveIntegerCore128Bits(1294, 712, 920, 291, 859, 353);
        }

        [Fact]
        public void TestHashString()
        {
            HashTestCore("".AsMemory(), TextDataViewType.Instance, 0, 0, 0, 0, 0);
            HashTestCore("hello".AsMemory(), TextDataViewType.Instance, 940, 951, 857, 770, 0);
        }

        [Fact]
        public void TestHashFloatingPointNumbers()
        {
            HashTestCore(1f, NumberDataViewType.Single, 463, 855, 732, 56, 557);
            HashTestCore(-1f, NumberDataViewType.Single, 252, 612, 780, 116, 515);
            HashTestCore(0f, NumberDataViewType.Single, 842, 358, 20, 429, 333);
            HashTestCore(float.NaN, NumberDataViewType.Single, 0, 0, 0, 0, 0);

            HashTestCore(1d, NumberDataViewType.Double, 188, 57, 690, 655, 896);
            HashTestCore(-1d, NumberDataViewType.Double, 885, 804, 22, 461, 309);
            HashTestCore(0d, NumberDataViewType.Double, 512, 851, 795, 333, 113);
            HashTestCore(double.NaN, NumberDataViewType.Double, 0, 0, 0, 0, 0);
        }

        [Fact]
        public void TestHashBool()
        {
            // These are the same for the hashes of 0 and 1.
            HashTestCore(false, BooleanDataViewType.Instance, 842, 358, 20, 429, 333);
            HashTestCore(true, BooleanDataViewType.Instance, 502, 537, 746, 847, 711);
        }

        private class HashData
        {
            public ReadOnlyMemory<char> Foo { get; set; }
        }

        [Fact]
        public void TestHashBackCompatability()
        {
            var samples = new[]
            {
                new HashData {Foo = "alibaba".AsMemory()},
                new HashData {Foo = "ba ba".AsMemory()},
            };

            IDataView data = ML.Data.LoadFromEnumerable(samples);

            var modelPath = GetDataPath("backcompat", "MurmurHashV1.zip");
            var model = ML.Model.Load(modelPath, out var _);

            var outputPath = DeleteOutputPath("Text", "murmurHash.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true });
                using (var fs = File.Create(outputPath))
                {
                    var transformedData = model.Transform(data);
                    DataSaverUtils.SaveDataView(ch, saver, transformedData, fs, keepHidden: true);
                }
            }
            CheckEquality("Text", "murmurHash.tsv");
        }

        [Fact]
        public void TestBackCompatNoCombineOption()
        {
            string dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var dataView = ML.Data.LoadFromTextFile(dataPath, new[]
            {
                new TextLoader.Column("Features", DataKind.Single, 1, 9)
            });

            string modelPath = GetDataPath("backcompat", "hashing-before-combine.zip");
            var model = ML.Model.Load(modelPath, out _);

            var hashed = model.Transform(dataView);
            var hashedCol = hashed.Schema["Features"];
            Assert.True(hashedCol.Type.GetItemType() is KeyDataViewType);
            Assert.Equal(9, hashedCol.Type.GetValueCount());
            Assert.Equal(Math.Pow(2, 31), hashedCol.Type.GetItemType().GetKeyCount());
        }

        [Fact]
        public void TestCombineLengthOneVector()
        {
            var data = new[]
            {
                new TestClass() { A = 1, B = 2, C = 3 },
                new TestClass() { A = 4, B = 5, C = 6 },
                new TestClass() { A = float.NaN, B = 3, C = 12 }
            };
            var dataView = ML.Data.LoadFromEnumerable(data);

            var pipeline = ML.Transforms.Concatenate("D", "A")
                .Append(ML.Transforms.Conversion.Hash(
                    new HashingEstimator.ColumnOptions("AHashed", "A"),
                    new HashingEstimator.ColumnOptions("DHashed", "D"),
                    new HashingEstimator.ColumnOptions("DHashedCombined", "D", combine: true)));

            var transformed = pipeline.Fit(dataView).Transform(dataView);
            Assert.True(transformed.Schema["D"].Type.IsKnownSizeVector());
            Assert.True(transformed.Schema["DHashed"].Type.IsKnownSizeVector());
            Assert.Equal(1, transformed.Schema["DHashed"].Type.GetValueCount());
            Assert.False(transformed.Schema["DHashedCombined"].Type.IsKnownSizeVector());
            Assert.Equal(1, transformed.Schema["DHashedCombined"].Type.GetValueCount());

            var aHashed = transformed.GetColumn<uint>(transformed.Schema["AHashed"]);
            var dHashed = transformed.GetColumn<VBuffer<uint>>(transformed.Schema["DHashed"]).Select(buffer =>
            {
                Assert.True(buffer.Length == 1);
                return buffer.DenseValues().First();
            });
            var dHashedCombined = transformed.GetColumn<uint>(transformed.Schema["DHashedCombined"]);

            Assert.Equal(aHashed, dHashed);
            Assert.Equal(aHashed, dHashedCombined);
            Assert.Equal((uint)0, aHashed.Last());
        }
    }
}
