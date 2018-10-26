// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.Tools;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Text;
using System;
using System.IO;
using System.Linq;
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
        public void TextNormalizerWorkout()
        {
            var data = new[] { new TestClass() { A = 1, B = new int[2] { 1,4 } },
                               new TestClass() { A = 2, B = new int[2] { 3,4 } }};
            var dataView = ComponentCreation.CreateDataView(Env, data);
            var pipe = new ConvertEstimator(Env, columns: new[] {new ConvertTransform.ColumnInfo("A", "ConvA", DataKind.R4),
                new ConvertTransform.ColumnInfo("B", "ConvB", DataKind.R4)});


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
                }
            };

            var allTypesDataView = new ConvertEstimator(Env, columns: new[] {
                new ConvertTransform.ColumnInfo("A", "ABL", DataKind.BL),
                new ConvertTransform.ColumnInfo("B", "ConvB", DataKind.R4)}
            );

            var dataPath = GetDataPath("wikipedia-detox-250-line-data.tsv");
            var reader = TextLoader.CreateReader(Env, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadText(1)), hasHeader: true);
            var dataSource = new MultiFileSource(dataPath);
            dataView = reader.Read(dataSource).AsDynamic;

            var pipeVariations = new TextNormalizerEstimator(Env, columns: new[] { ("text", "NormText") }).Append(
                                new TextNormalizerEstimator(Env, textCase: TextNormalizerEstimator.CaseNormalizationMode.Upper, columns: new[] { ("text", "UpperText") })).Append(
                                new TextNormalizerEstimator(Env, keepDiacritics: true, columns: new[] { ("text", "WithDiacriticsText") })).Append(
                                new TextNormalizerEstimator(Env, keepNumbers: false, columns: new[] { ("text", "NoNumberText") })).Append(
                                new TextNormalizerEstimator(Env, keepPunctuations: false, columns: new[] { ("text", "NoPuncText") }));

            var outputPath = GetOutputPath("Text", "Normalized.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true });
                var savedData = TakeFilter.Create(Env, pipeVariations.Fit(dataView).Transform(dataView), 5);
                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);
            }

            CheckEquality("Text", "Normalized.tsv");
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
            var dataView = ComponentCreation.CreateDataView(Env, data);
            var pipe = new ConvertEstimator(Env, columns: new[] {new ConvertTransform.ColumnInfo("A", "ConvA", DataKind.R8),
                new ConvertTransform.ColumnInfo("B", "ConvB", DataKind.R8)});

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
            var pipe = new CategoricalEstimator(Env, new[] {
                new CategoricalEstimator.ColumnInfo("A", "CatA", CategoricalTransform.OutputKind.Ind),
                new CategoricalEstimator.ColumnInfo("B", "CatB", CategoricalTransform.OutputKind.Key)
            }).Append(new ConvertEstimator(Env, new[] {
                new ConvertTransform.ColumnInfo("CatA", "ConvA", DataKind.R8),
                new ConvertTransform.ColumnInfo("CatB", "ConvB", DataKind.U2)
            }));
            var dataView = ComponentCreation.CreateDataView(Env, data);
            dataView = pipe.Fit(dataView).Transform(dataView);
            ValidateMetadata(dataView);
        }

        private void ValidateMetadata(IDataView result)
        {
            Assert.True(result.Schema.TryGetColumnIndex("ConvA", out int colA));
            Assert.True(result.Schema.TryGetColumnIndex("ConvB", out int colB));
            var types = result.Schema.GetMetadataTypes(colA);
            Assert.Equal(types.Select(x => x.Key), new string[2] { MetadataUtils.Kinds.SlotNames,MetadataUtils.Kinds.IsNormalized });
            VBuffer<ReadOnlyMemory<char>> slots = default;
            bool normalized = default;
            result.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, colA, ref slots);
            Assert.True(slots.Length == 2);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "1", "2"});
            result.Schema.GetMetadata(MetadataUtils.Kinds.IsNormalized, colA, ref normalized);
            Assert.True(normalized);

            types = result.Schema.GetMetadataTypes(colB);
            Assert.Equal(types.Select(x => x.Key), new string[1] { MetadataUtils.Kinds.KeyValues});
            result.Schema.GetMetadata(MetadataUtils.Kinds.KeyValues, colB, ref slots);
            Assert.True(slots.Length == 2);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "A", "B" });
        }
    }
}
