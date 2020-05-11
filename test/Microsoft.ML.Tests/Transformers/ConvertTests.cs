// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.Tools;
using Microsoft.ML.Transforms;
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

        private sealed class TestPrimitiveClassConverted
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
            [VectorType(2)]
            public float[] ConvA;
            [VectorType(2)]
            public float[] ConvB;
            [VectorType(2)]
            public float[] ConvC;
            [VectorType(2)]
            public float[] ConvD;
            [VectorType(2)]
            public float[] ConvE;
            [VectorType(2)]
            public float[] ConvF;
            [VectorType(2)]
            public float[] ConvG;
            [VectorType(2)]
            public float[] ConvH;
            [VectorType(2)]
            public float[] ConvK;
            [VectorType(2)]
            public float[] ConvL;
            [VectorType(2)]
            public float[] ConvM;
            [VectorType(2)]
            public float[] ConvN;
            [VectorType(2)]
            public int[] ConvBI;
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
            var dataView = ML.Data.LoadFromEnumerable(data);
            var pipe = ML.Transforms.Conversion.ConvertType(columns: new[] {new TypeConvertingEstimator.ColumnOptions("ConvA", DataKind.Single, "A"),
                new TypeConvertingEstimator.ColumnOptions("ConvB", DataKind.Single, "B")});

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
                    AB = new []{true, false},
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

            var allTypesDataView = ML.Data.LoadFromEnumerable(allTypesData);
            var allTypesPipe = ML.Transforms.Conversion.ConvertType(columns: new[] {
                new TypeConvertingEstimator.ColumnOptions("ConvA", DataKind.Single, "AA"),
                new TypeConvertingEstimator.ColumnOptions("ConvB", DataKind.Single, "AB"),
                new TypeConvertingEstimator.ColumnOptions("ConvC", DataKind.Single, "AC"),
                new TypeConvertingEstimator.ColumnOptions("ConvD", DataKind.Single, "AD"),
                new TypeConvertingEstimator.ColumnOptions("ConvE", DataKind.Single, "AE"),
                new TypeConvertingEstimator.ColumnOptions("ConvF", DataKind.Single, "AF"),
                new TypeConvertingEstimator.ColumnOptions("ConvG", DataKind.Single, "AG"),
                new TypeConvertingEstimator.ColumnOptions("ConvH", DataKind.Single, "AH"),
                new TypeConvertingEstimator.ColumnOptions("ConvK", DataKind.Single, "AK"),
                new TypeConvertingEstimator.ColumnOptions("ConvL", DataKind.Single, "AL"),
                new TypeConvertingEstimator.ColumnOptions("ConvM", DataKind.Single, "AM"),
                new TypeConvertingEstimator.ColumnOptions("ConvN", DataKind.Single, "AN"),
                new TypeConvertingEstimator.ColumnOptions("ConvBI", DataKind.Int32, "AB") // verify Boolean -> Int32 conversion
            }
            );
            TestEstimatorCore(allTypesPipe, allTypesDataView);

            var actualConvertedValues = allTypesPipe.Fit(allTypesDataView).Transform(allTypesDataView);

            var allTypesDataConverted = new[]
            {
                new TestPrimitiveClassConverted()
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
                    AN = new double[]{ 1.0d,1.0d,},
                    ConvA = new float[] { float.NaN, float.NaN },
                    ConvB = new float[] { 0, 1 },
                    ConvC = new float[] { -1, 1 },
                    ConvD = new float[] { 0, 1 },
                    ConvE = new float[] { 0, 1 },
                    ConvF = new float[] { -1, 1 },
                    ConvG = new float[] { -1, 1 },
                    ConvH = new float[] { 0, 1},
                    ConvK = new float[] { -1, 1 },
                    ConvL = new float[] { 0, 1 },
                    ConvM = new float[] { 1, 1 },
                    ConvN = new float[] { 1, 1 },
                    ConvBI = new int[] { 0, 1 },
                },
                new TestPrimitiveClassConverted()
                {
                    AA = new []{"0", "1"},
                    AB = new []{true, false},
                    AC = new []{ int.MinValue, int.MaxValue},
                    AD = new uint[]{ uint.MinValue, uint.MaxValue},
                    AE = new byte[]{ byte.MinValue, byte.MaxValue},
                    AF = new sbyte[]{ sbyte.MinValue, sbyte.MaxValue},
                    AG = new short[]{ short.MinValue, short.MaxValue},
                    AH = new ushort[]{ ushort.MinValue, ushort.MaxValue},
                    AK = new long[]{ long.MinValue, long.MaxValue},
                    AL = new ulong[]{ ulong.MinValue, ulong.MaxValue},
                    AM = new float[]{ float.MinValue, float.MaxValue,},
                    AN = new double[]{ double.MinValue, double.MaxValue,},
                    ConvA = new float[] { 0, 1 },
                    ConvB = new float[] { 1, 0 },
                    ConvC = new float[] { int.MinValue, int.MaxValue },
                    ConvD = new float[] { uint.MinValue, uint.MaxValue },
                    ConvE = new float[] { byte.MinValue, byte.MaxValue },
                    ConvF = new float[] { sbyte.MinValue, sbyte.MaxValue },
                    ConvG = new float[] { short.MinValue, short.MaxValue },
                    ConvH = new float[] { ushort.MinValue, ushort.MaxValue },
                    ConvK = new float[] { long.MinValue, long.MaxValue },
                    ConvL = new float[] { ulong.MinValue, ulong.MaxValue },
                    ConvM = new float[] { float.MinValue, float.MaxValue },
                    ConvN = new float[] { float.NegativeInfinity, float.PositiveInfinity },
                    ConvBI = new int[] { 1, 0 },
                }
            };
            var expectedConvertedValues = ML.Data.LoadFromEnumerable(allTypesDataConverted);

            CheckSameValues(expectedConvertedValues, actualConvertedValues);

            var allInputTypesData = new[] { new { A = (sbyte)sbyte.MinValue, B = (byte)byte.MinValue, C = double.MaxValue, D = float.MinValue, E = "already a string", F = false } };
            var allInputTypesDataView = ML.Data.LoadFromEnumerable(allInputTypesData);
            var allInputTypesDataPipe = ML.Transforms.Conversion.ConvertType(columns: new[] {new TypeConvertingEstimator.ColumnOptions("A1", DataKind.String, "A"),
                new TypeConvertingEstimator.ColumnOptions("B1", DataKind.String, "B"),
                new TypeConvertingEstimator.ColumnOptions("C1", DataKind.String, "C"),
                new TypeConvertingEstimator.ColumnOptions("D1", DataKind.String, "D"),
                new TypeConvertingEstimator.ColumnOptions("E1", DataKind.String, "E"),
                new TypeConvertingEstimator.ColumnOptions("F1", DataKind.String, "F"),
            });

            var convertedValues = allInputTypesDataPipe.Fit(allInputTypesDataView).Transform(allInputTypesDataView);

            var expectedValuesData = new[] { new { A = (sbyte)sbyte.MinValue, B = (byte)byte.MinValue, C = double.MaxValue, D = float.MinValue, E = "already a string", F = false,
                A1 = "-128", B1 = "0", C1 = "1.7976931348623157E+308", D1 = "-3.402823E+38", E1 = "already a string", F1 = "False" } };
            var expectedValuesDataView = ML.Data.LoadFromEnumerable(expectedValuesData);

            CheckSameValues(expectedValuesDataView, convertedValues);
            TestEstimatorCore(allInputTypesDataPipe, allInputTypesDataView);

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

            var mlContext = new MLContext(1);
            var dataView = mlContext.Data.LoadFromEnumerable(data);

            var sideDataBuilder = new ArrayDataViewBuilder(mlContext);
            sideDataBuilder.AddColumn("Hello", "hello", "my", "friend");
            var sideData = sideDataBuilder.GetDataView();

            // For some reason the column info is on the *transformer*, not the estimator. Already tracked as issue #1760.
            var ci = new ValueToKeyMappingEstimator.ColumnOptions("CatA", "A");
            var pipe = mlContext.Transforms.Conversion.MapValueToKey(new[] { ci }, sideData);
            var output = pipe.Fit(dataView).Transform(dataView);

            VBuffer<ReadOnlyMemory<char>> slotNames = default;
            output.Schema["CatA"].Annotations.GetValue(AnnotationUtils.Kinds.KeyValues, ref slotNames);

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
            var dataView = ML.Data.LoadFromEnumerable(data);
            var pipe = ML.Transforms.Conversion.ConvertType(columns: new[] {new TypeConvertingEstimator.ColumnOptions("ConvA", typeof(double), "A"),
                new TypeConvertingEstimator.ColumnOptions("ConvB", typeof(double), "B")});

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
            var pipe = ML.Transforms.Categorical.OneHotEncoding(new[] {
                new OneHotEncodingEstimator.ColumnOptions("CatA", "A", OneHotEncodingEstimator.OutputKind.Indicator),
                new OneHotEncodingEstimator.ColumnOptions("CatB", "B", OneHotEncodingEstimator.OutputKind.Key)
            }).Append(ML.Transforms.Conversion.ConvertType(new[] {
                new TypeConvertingEstimator.ColumnOptions("ConvA", DataKind.Double, "CatA"),
                new TypeConvertingEstimator.ColumnOptions("ConvB", DataKind.UInt16, "CatB")
            }));
            var dataView = ML.Data.LoadFromEnumerable(data);
            dataView = pipe.Fit(dataView).Transform(dataView);
            ValidateMetadata(dataView);
        }

        private void ValidateMetadata(IDataView result)
        {
            Assert.Equal(result.Schema["ConvA"].Annotations.Schema.Select(x => x.Name), new string[2] { AnnotationUtils.Kinds.SlotNames, AnnotationUtils.Kinds.IsNormalized });
            VBuffer<ReadOnlyMemory<char>> slots = default;
            result.Schema["ConvA"].GetSlotNames(ref slots);
            Assert.True(slots.Length == 2);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "1", "2" });
            Assert.True(result.Schema["ConvA"].IsNormalized());

            Assert.Equal(result.Schema["ConvB"].Annotations.Schema.Select(x => x.Name), new string[1] { AnnotationUtils.Kinds.KeyValues });
            result.Schema["ConvB"].GetKeyValues(ref slots);
            Assert.True(slots.Length == 2);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "A", "B" });
        }


        public class SimpleSchemaUIntColumn
        {
            [LoadColumn(0)]
            [KeyType(4)]
            public uint key;
        }

        [Fact]
        public void TypeConvertKeyBackCompatTest()
        {
            // Model generated using the following command before the change removing Min and Count from KeyType.
            // ML.Transforms.Conversion.ConvertType(new[] { new TypeConvertingEstimator.ColumnOptions("key", "convertedKey",
            //      DataKind.UInt64, new KeyCount(4)) }).Fit(dataView);
            var dataArray = new[]
            {
                new SimpleSchemaUIntColumn() { key = 0 },
                new SimpleSchemaUIntColumn() { key = 1 },
                new SimpleSchemaUIntColumn() { key = 2 },
                new SimpleSchemaUIntColumn() { key = 3 }

            };

            var dataView = ML.Data.LoadFromEnumerable(dataArray);

            // Check old model can be loaded.
            var modelPath = GetDataPath("backcompat", "type-convert-key-model.zip");
            ITransformer modelOld;
            using (var ch = Env.Start("load"))
            {
                using (var fs = File.OpenRead(modelPath))
                    modelOld = ML.Model.Load(fs, out var schema);
            }
            var outDataOld = modelOld.Transform(dataView);

            var modelNew = ML.Transforms.Conversion.ConvertType(new[] { new TypeConvertingEstimator.ColumnOptions("convertedKey",
                DataKind.UInt64, "key", new KeyCount(4)) }).Fit(dataView);
            var outDataNew = modelNew.Transform(dataView);

            // Check that old and new model produce the same result.
            Assert.True(outDataNew.Schema[1].Type.Equals(outDataNew.Schema[1].Type));
        }
    }
}
