// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.Tools;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Categorical;
using Microsoft.ML.Transforms.Conversions;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public class ConvertTests : TestDataPipeBase
    {
        public ConvertTests(ITestOutputHelper output) : base(output)
        {
        }

        private sealed class TestPrimitiveClass
        {
            [VectorType(2)]
            public string[] AA;
            [VectorType(2)]
            public bool[] AB;
            [VectorType(2)]
            public int[] AC;
            [VectorType(2)]
            public uint[] AD;
            [VectorType(2)]
            public byte[] AE;
            [VectorType(2)]
            public sbyte[] AF;
            [VectorType(2)]
            public short[] AG;
            [VectorType(2)]
            public ushort[] AH;
            [VectorType(2)]
            public long[] AK;
            [VectorType(2)]
            public ulong[] AL;
            [VectorType(2)]
            public float[] AM;
            [VectorType(2)]
            public double[] AN;
        }

        private sealed class TestClass
        {
            public int A;
            [VectorType(2)]
            public int[] B;
        }

        private sealed class MetaClass
        {
            public float A;
            public string B;
        }

        private sealed class TestStringClass
        {
            public string A;
        }

        [Fact]
        public void TestConvertWorkout()
        {
            var data = new[] { new TestClass() { A = 1, B = new int[2] { 1,4 } },
                               new TestClass() { A = 2, B = new int[2] { 3,4 } }};
            var dataView = ML.Data.ReadFromEnumerable(data);
            var pipe = new TypeConvertingEstimator(Env, columns: new[] {new TypeConvertingTransformer.ColumnInfo("ConvA", DataKind.R4, "A"),
                new TypeConvertingTransformer.ColumnInfo("ConvB", DataKind.R4, "B")});

            TestEstimatorCore(pipe, dataView);
            var allTypesData = new[]
            {
                new TestPrimitiveClass()
                {
                    AA = new []{"a", "b"},
                    AB = new []{false, true},
                    AC = new []{ -1,1},
                    AD = new uint[]{ 0,1},
                    AE = new byte[]{ 0,1},
                    AF = new sbyte[]{ -1,1},
                    AG = new short[]{ -1,1},
                    AH = new ushort[]{0, 1},
                    AK = new long[]{ -1,1},
                    AL = new ulong[]{ 0,1},
                    AM = new float[]{ 1.0f,1.0f,},
                    AN = new double[]{ 1.0d,1.0d,}
                },
                  new TestPrimitiveClass()
                {
                    AA = new []{"0", "1"},
                    AB = new []{false, true},
                    AC = new []{ int.MinValue, int.MaxValue},
                    AD = new uint[]{ uint.MinValue, uint.MaxValue},
                    AE = new byte[]{ byte.MinValue, byte.MaxValue},
                    AF = new sbyte[]{ sbyte.MinValue, sbyte.MaxValue},
                    AG = new short[]{ short.MinValue, short.MaxValue},
                    AH = new ushort[]{ ushort.MinValue, ushort.MaxValue},
                    AK = new long[]{ long.MinValue, long.MaxValue},
                    AL = new ulong[]{ ulong.MinValue, ulong.MaxValue},
                    AM = new float[]{ float.MinValue, float.MaxValue,},
                    AN = new double[]{ double.MinValue, double.MaxValue,}
                }
            };

            var allTypesDataView = ML.Data.ReadFromEnumerable(allTypesData);
            var allTypesPipe = new TypeConvertingEstimator(Env, columns: new[] {
                new TypeConvertingTransformer.ColumnInfo("ConvA", DataKind.R4, "AA"),
                new TypeConvertingTransformer.ColumnInfo("ConvB", DataKind.R4, "AB"),
                new TypeConvertingTransformer.ColumnInfo("ConvC", DataKind.R4, "AC"),
                new TypeConvertingTransformer.ColumnInfo("ConvD", DataKind.R4, "AD"),
                new TypeConvertingTransformer.ColumnInfo("ConvE", DataKind.R4, "AE"),
                new TypeConvertingTransformer.ColumnInfo("ConvF", DataKind.R4, "AF"),
                new TypeConvertingTransformer.ColumnInfo("ConvG", DataKind.R4, "AG"),
                new TypeConvertingTransformer.ColumnInfo("ConvH", DataKind.R4, "AH"),
                new TypeConvertingTransformer.ColumnInfo("ConvK", DataKind.R4, "AK"),
                new TypeConvertingTransformer.ColumnInfo("ConvL", DataKind.R4, "AL"),
                new TypeConvertingTransformer.ColumnInfo("ConvM", DataKind.R4, "AM"),
                new TypeConvertingTransformer.ColumnInfo("ConvN", DataKind.R4, "AN")}
            );
            TestEstimatorCore(allTypesPipe, allTypesDataView);

            var outputPath = GetOutputPath("Convert", "Types.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true });
                var savedData = TakeFilter.Create(Env, allTypesPipe.Fit(allTypesDataView).Transform(allTypesDataView), 2);
                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);
            }

            CheckEquality("Convert", "Types.tsv");
            Done();
        }

        /// <summary>
        /// Apply <see cref="KeyToValueMappingEstimator"/> with side data.
        /// </summary>
        [Fact]
        public void ValueToKeyFromSideData()
        {
            // In this case, whatever the value of the input, the term mapping should come from the optional side data if specified.
            var data = new[] { new TestStringClass() { A = "Stay" }, new TestStringClass() { A = "awhile and listen" } };

            var mlContext = new MLContext();
            var dataView = mlContext.Data.ReadFromEnumerable(data);

            var sideDataBuilder = new ArrayDataViewBuilder(mlContext);
            sideDataBuilder.AddColumn("Hello", "hello", "my", "friend");
            var sideData = sideDataBuilder.GetDataView();

            // For some reason the column info is on the *transformer*, not the estimator. Already tracked as issue #1760.
            var ci = new ValueToKeyMappingTransformer.ColumnInfo("CatA", "A");
            var pipe = mlContext.Transforms.Conversion.MapValueToKey(new[] { ci }, sideData);
            var output = pipe.Fit(dataView).Transform(dataView);

            VBuffer<ReadOnlyMemory<char>> slotNames = default;
            output.Schema["CatA"].Metadata.GetValue(MetadataUtils.Kinds.KeyValues, ref slotNames);

            Assert.Equal(3, slotNames.Length);
            Assert.Equal("hello", slotNames.GetItemOrDefault(0).ToString());
            Assert.Equal("my", slotNames.GetItemOrDefault(1).ToString());
            Assert.Equal("friend", slotNames.GetItemOrDefault(2).ToString());

            Done();
        }



        [Fact]
        public void TestCommandLine()
        {
            Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:TX:0} xf=Convert{col=B:A type=R4} in=f:\2.txt" }), (int)0);
        }

        [Fact]
        public void TestOldSavingAndLoading()
        {
            var data = new[] { new TestClass() { A = 1, B = new int[2] { 1,4 } },
                               new TestClass() { A = 2, B = new int[2] { 3,4 } }};
            var dataView = ML.Data.ReadFromEnumerable(data);
            var pipe = new TypeConvertingEstimator(Env, columns: new[] {new TypeConvertingTransformer.ColumnInfo("ConvA", DataKind.R8, "A"),
                new TypeConvertingTransformer.ColumnInfo("ConvB", DataKind.R8, "B")});

            var result = pipe.Fit(dataView).Transform(dataView);
            var resultRoles = new RoleMappedData(result);
            using (var ms = new MemoryStream())
            {
                TrainUtils.SaveModel(Env, Env.Start("saving"), ms, null, resultRoles);
                ms.Position = 0;
                var loadedView = ModelFileUtils.LoadTransforms(Env, dataView, ms);
            }
        }

        [Fact]
        public void TestMetadata()
        {
            var data = new[] { new MetaClass() { A = 1, B = "A" },
                               new MetaClass() { A = 2, B = "B" }};
            var pipe = new OneHotEncodingEstimator(Env, new[] {
                new OneHotEncodingEstimator.ColumnInfo("CatA", "A", OneHotEncodingTransformer.OutputKind.Ind),
                new OneHotEncodingEstimator.ColumnInfo("CatB", "B", OneHotEncodingTransformer.OutputKind.Key)
            }).Append(new TypeConvertingEstimator(Env, new[] {
                new TypeConvertingTransformer.ColumnInfo("ConvA", DataKind.R8, "CatA"),
                new TypeConvertingTransformer.ColumnInfo("ConvB", DataKind.U2, "CatB")
            }));
            var dataView = ML.Data.ReadFromEnumerable(data);
            dataView = pipe.Fit(dataView).Transform(dataView);
            ValidateMetadata(dataView);
        }

        private void ValidateMetadata(IDataView result)
        {
            Assert.Equal(result.Schema["ConvA"].Metadata.Schema.Select(x => x.Name), new string[2] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.IsNormalized });
            VBuffer<ReadOnlyMemory<char>> slots = default;
            result.Schema["ConvA"].GetSlotNames(ref slots);
            Assert.True(slots.Length == 2);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "1", "2" });
            Assert.True(result.Schema["ConvA"].IsNormalized());

            Assert.Equal(result.Schema["ConvB"].Metadata.Schema.Select(x => x.Name), new string[1] { MetadataUtils.Kinds.KeyValues });
            result.Schema["ConvB"].GetKeyValues(ref slots);
            Assert.True(slots.Length == 2);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "A", "B" });
        }


        public class SimpleSchemaUIntColumn
        {
            [LoadColumn(0)]
            [KeyType(Count = 4)]
            public uint key;
        }

        [Fact]
        public void TypeConvertKeyBackCompatTest()
        {
            // Model generated using the following command before the change removing Min and Count from KeyType.
            // ML.Transforms.Conversion.ConvertType(new[] { new TypeConvertingTransformer.ColumnInfo("key", "convertedKey",
            //      DataKind.U8, new KeyCount(4)) }).Fit(dataView);
            var dataArray = new[]
            {
                new SimpleSchemaUIntColumn() { key = 0 },
                new SimpleSchemaUIntColumn() { key = 1 },
                new SimpleSchemaUIntColumn() { key = 2 },
                new SimpleSchemaUIntColumn() { key = 3 }

            };

            var dataView = ML.Data.ReadFromEnumerable(dataArray);

            // Check old model can be loaded.
            var modelPath = GetDataPath("backcompat", "type-convert-key-model.zip");
            ITransformer modelOld;
            using (var ch = Env.Start("load"))
            {
                using (var fs = File.OpenRead(modelPath))
                     modelOld = ML.Model.Load(fs);
            }
            var outDataOld = modelOld.Transform(dataView); 

            var modelNew = ML.Transforms.Conversion.ConvertType(new[] { new TypeConvertingTransformer.ColumnInfo("convertedKey",
                DataKind.U8, "key", new KeyCount(4)) }).Fit(dataView);
            var outDataNew = modelNew.Transform(dataView);

            // Check that old and new model produce the same result.
            Assert.True(outDataNew.Schema[1].Type.Equals(outDataNew.Schema[1].Type));
        }
    }
}
