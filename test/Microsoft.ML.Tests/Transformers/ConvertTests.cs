// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
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

        private class TestPrimitiveClass
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

        private class TestClass
        {
            public int A;
            [VectorType(2)]
            public int[] B;
        }

        public class MetaClass
        {
            public float A;
            public string B;

        }


        [Fact]
        public void TestConvertWorkout()
        {
            var data = new[] { new TestClass() { A = 1, B = new int[2] { 1,4 } },
                               new TestClass() { A = 2, B = new int[2] { 3,4 } }};
            var dataView = ML.Data.ReadFromEnumerable(data);
            var pipe = new TypeConvertingEstimator(Env, columns: new[] {new TypeConvertingTransformer.ColumnInfo("A", "ConvA", DataKind.R4),
                new TypeConvertingTransformer.ColumnInfo("B", "ConvB", DataKind.R4)});

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
                new TypeConvertingTransformer.ColumnInfo("AA", "ConvA", DataKind.R4),
                new TypeConvertingTransformer.ColumnInfo("AB", "ConvB", DataKind.R4),
                new TypeConvertingTransformer.ColumnInfo("AC", "ConvC", DataKind.R4),
                new TypeConvertingTransformer.ColumnInfo("AD", "ConvD", DataKind.R4),
                new TypeConvertingTransformer.ColumnInfo("AE", "ConvE", DataKind.R4),
                new TypeConvertingTransformer.ColumnInfo("AF", "ConvF", DataKind.R4),
                new TypeConvertingTransformer.ColumnInfo("AG", "ConvG", DataKind.R4),
                new TypeConvertingTransformer.ColumnInfo("AH", "ConvH", DataKind.R4),
                new TypeConvertingTransformer.ColumnInfo("AK", "ConvK", DataKind.R4),
                new TypeConvertingTransformer.ColumnInfo("AL", "ConvL", DataKind.R4),
                new TypeConvertingTransformer.ColumnInfo("AM", "ConvM", DataKind.R4),
                new TypeConvertingTransformer.ColumnInfo("AN", "ConvN", DataKind.R4)}
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
            var pipe = new TypeConvertingEstimator(Env, columns: new[] {new TypeConvertingTransformer.ColumnInfo("A", "ConvA", DataKind.R8),
                new TypeConvertingTransformer.ColumnInfo("B", "ConvB", DataKind.R8)});

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
                new OneHotEncodingEstimator.ColumnInfo("A", "CatA", OneHotEncodingTransformer.OutputKind.Ind),
                new OneHotEncodingEstimator.ColumnInfo("B", "CatB", OneHotEncodingTransformer.OutputKind.Key)
            }).Append(new TypeConvertingEstimator(Env, new[] {
                new TypeConvertingTransformer.ColumnInfo("CatA", "ConvA", DataKind.R8),
                new TypeConvertingTransformer.ColumnInfo("CatB", "ConvB", DataKind.U2)
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
            result.Schema["ConvB"].Metadata.GetValue(MetadataUtils.Kinds.KeyValues, ref slots);
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

            var modelNew = ML.Transforms.Conversion.ConvertType(new[] { new TypeConvertingTransformer.ColumnInfo("key", "convertedKey",
                DataKind.U8, new KeyCount(4)) }).Fit(dataView);
            var outDataNew = modelNew.Transform(dataView);

            // Check that old and new model produce the same result.
            Assert.True(outDataNew.Schema[1].Type.Equals(outDataNew.Schema[1].Type));
        }
    }
}
