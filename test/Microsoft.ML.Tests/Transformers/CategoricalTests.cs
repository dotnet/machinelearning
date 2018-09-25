﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.Tools;
using System;
using System.IO;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public class CategoricalTests : TestDataPipeBase
    {
        public CategoricalTests(ITestOutputHelper output) : base(output)
        {
        }

        private class TestClass
        {
            public int A;
            public int B;
            public int C;
        }

        private class TestMeta
        {
            [VectorType(2)]
            public string[] A;
            public string B;
            [VectorType(2)]
            public int[] C;
            public int D;
            [VectorType(2)]
            public float[] E;
            public float F;
            [VectorType(2)]
            public string[] G;
            public string H;
        }

        [Fact]
        public void CategoricalWorkout()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };

            var dataView = ComponentCreation.CreateDataView(Env, data);
            var pipe = new CategoricalEstimator(Env, new[]{
                    new CategoricalEstimator.ColumnInfo("A", "CatA", CategoricalTransform.OutputKind.Bag),
                    new CategoricalEstimator.ColumnInfo("A", "CatB", CategoricalTransform.OutputKind.Bin),
                    new CategoricalEstimator.ColumnInfo("A", "CatC", CategoricalTransform.OutputKind.Ind),
                    new CategoricalEstimator.ColumnInfo("A", "CatD", CategoricalTransform.OutputKind.Key),
                });

            TestEstimatorCore(pipe, dataView);
            Done();
        }

        [Fact]
        public void CategoricalStatic()
        {
            string dataPath = GetDataPath("breast-cancer.txt");
            var reader = TextLoader.CreateReader(Env, ctx => (
                ScalarString: ctx.LoadText(1),
                VectorString: ctx.LoadText(1, 4)));
            var data = reader.Read(new MultiFileSource(dataPath));
            var wrongCollection = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };

            var invalidData = ComponentCreation.CreateDataView(Env, wrongCollection);
            var est = data.MakeNewEstimator().
                  Append(row => (
                  A: row.ScalarString.OneHotEncoding(outputKind: CategoricalStaticExtensions.OneHotScalarOutputKind.Ind),
                  B: row.VectorString.OneHotEncoding(outputKind: CategoricalStaticExtensions.OneHotVectorOutputKind.Ind),
                  C: row.VectorString.OneHotEncoding(outputKind: CategoricalStaticExtensions.OneHotVectorOutputKind.Bag),
                  D: row.ScalarString.OneHotEncoding(outputKind: CategoricalStaticExtensions.OneHotScalarOutputKind.Bin),
                  E: row.VectorString.OneHotEncoding(outputKind: CategoricalStaticExtensions.OneHotVectorOutputKind.Bin)
                  ));

            TestEstimatorCore(est.AsDynamic, data.AsDynamic, invalidInput: invalidData);

            var outputPath = GetOutputPath("Categorical", "featurized.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true });
                IDataView savedData = TakeFilter.Create(Env, est.Fit(data).Transform(data).AsDynamic, 4);
                savedData = new ChooseColumnsTransform(Env, savedData, "A", "B", "C", "D", "E");
                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);
            }

            CheckEquality("Categorical", "featurized.tsv");
            Done();
        }

        [Fact]
        public void TestMetadataPropagation()
        {
            var data = new[] {
                new TestMeta() { A=new string[2] { "A", "B"}, B="C", C=new int[2] { 3,5}, D= 6, E= new float[2] { 1.0f,2.0f}, F = 1.0f , G= new string[2]{ "A","D"}, H="D"},
                new TestMeta() { A=new string[2] { "A", "B"}, B="C", C=new int[2] { 5,3}, D= 1, E=new float[2] { 3.0f,4.0f}, F = -1.0f ,G= new string[2]{"E", "A"}, H="E"},
                new TestMeta() { A=new string[2] { "A", "B"}, B="C", C=new int[2] { 3,5}, D= 6, E=new float[2] { 5.0f,6.0f}, F = 1.0f ,G= new string[2]{ "D", "E"}, H="D"} };


            var dataView = ComponentCreation.CreateDataView(Env, data);
            var bagPipe = new CategoricalEstimator(Env,
                new CategoricalEstimator.ColumnInfo("A", "CatA", CategoricalTransform.OutputKind.Bag),
                new CategoricalEstimator.ColumnInfo("B", "CatB", CategoricalTransform.OutputKind.Bag),
                new CategoricalEstimator.ColumnInfo("C", "CatC", CategoricalTransform.OutputKind.Bag),
                new CategoricalEstimator.ColumnInfo("D", "CatD", CategoricalTransform.OutputKind.Bag),
                new CategoricalEstimator.ColumnInfo("E", "CatE", CategoricalTransform.OutputKind.Ind),
                new CategoricalEstimator.ColumnInfo("F", "CatF", CategoricalTransform.OutputKind.Ind),
                new CategoricalEstimator.ColumnInfo("G", "CatG", CategoricalTransform.OutputKind.Key),
                new CategoricalEstimator.ColumnInfo("H", "CatH", CategoricalTransform.OutputKind.Key));

            var binPipe = new CategoricalEstimator(Env,
                new CategoricalEstimator.ColumnInfo("A", "CatA", CategoricalTransform.OutputKind.Bin),
                new CategoricalEstimator.ColumnInfo("B", "CatB", CategoricalTransform.OutputKind.Bin),
                new CategoricalEstimator.ColumnInfo("C", "CatC", CategoricalTransform.OutputKind.Bin),
                new CategoricalEstimator.ColumnInfo("D", "CatD", CategoricalTransform.OutputKind.Bin),
                new CategoricalEstimator.ColumnInfo("E", "CatE", CategoricalTransform.OutputKind.Ind),
                new CategoricalEstimator.ColumnInfo("F", "CatF", CategoricalTransform.OutputKind.Ind),
                new CategoricalEstimator.ColumnInfo("G", "CatG", CategoricalTransform.OutputKind.Key),
                new CategoricalEstimator.ColumnInfo("H", "CatH", CategoricalTransform.OutputKind.Key));

            var bagResult = bagPipe.Fit(dataView).Transform(dataView);
            var binResult = binPipe.Fit(dataView).Transform(dataView);

            ValidateBagMetadata(bagResult);
            ValidateBinMetadata(binResult);
            Done();
        }

        private void ValidateBinMetadata(IDataView result)
        {
            Assert.True(result.Schema.TryGetColumnIndex("CatA", out int colA));
            Assert.True(result.Schema.TryGetColumnIndex("CatB", out int colB));
            Assert.True(result.Schema.TryGetColumnIndex("CatC", out int colC));
            Assert.True(result.Schema.TryGetColumnIndex("CatD", out int colD));
            Assert.True(result.Schema.TryGetColumnIndex("CatE", out int colE));
            Assert.True(result.Schema.TryGetColumnIndex("CatF", out int colF));
            Assert.True(result.Schema.TryGetColumnIndex("CatE", out int colG));
            Assert.True(result.Schema.TryGetColumnIndex("CatF", out int colH));
            var types = result.Schema.GetMetadataTypes(colA);
            Assert.Equal(types.Select(x => x.Key), new string[1] { MetadataUtils.Kinds.SlotNames });
            VBuffer<ReadOnlyMemory<char>> slots = default;
            bool normalized = default;
            result.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, colA, ref slots);
            Assert.True(slots.Length == 6);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[6] { "[0].Bit2", "[0].Bit1", "[0].Bit0", "[1].Bit2", "[1].Bit1", "[1].Bit0" });

            types = result.Schema.GetMetadataTypes(colB);
            Assert.Equal(types.Select(x => x.Key), new string[2] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.IsNormalized });
            result.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, colB, ref slots);
            Assert.True(slots.Length == 2);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "Bit1", "Bit0" });
            result.Schema.GetMetadata(MetadataUtils.Kinds.IsNormalized, colB, ref normalized);
            Assert.True(normalized);

            types = result.Schema.GetMetadataTypes(colC);
            Assert.Equal(types.Select(x => x.Key), new string[1] { MetadataUtils.Kinds.SlotNames });
            result.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, colC, ref slots);
            Assert.True(slots.Length == 6);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[6] { "[0].Bit2", "[0].Bit1", "[0].Bit0", "[1].Bit2", "[1].Bit1", "[1].Bit0" });


            types = result.Schema.GetMetadataTypes(colD);
            Assert.Equal(types.Select(x => x.Key), new string[2] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.IsNormalized });
            result.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, colD, ref slots);
            Assert.True(slots.Length == 3);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[3] { "Bit2", "Bit1", "Bit0" });
            result.Schema.GetMetadata(MetadataUtils.Kinds.IsNormalized, colD, ref normalized);
            Assert.True(normalized);


            types = result.Schema.GetMetadataTypes(colE);
            Assert.Equal(types.Select(x => x.Key), new string[1] { MetadataUtils.Kinds.SlotNames });
            result.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, colE, ref slots);
            Assert.True(slots.Length == 8);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[8] { "[0].Bit3", "[0].Bit2", "[0].Bit1", "[0].Bit0", "[1].Bit3", "[1].Bit2", "[1].Bit1", "[1].Bit0" });

            types = result.Schema.GetMetadataTypes(colF);
            Assert.Equal(types.Select(x => x.Key), new string[2] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.IsNormalized });
            result.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, colE, ref slots);
            Assert.True(slots.Length == 8);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[8] { "[0].Bit3", "[0].Bit2", "[0].Bit1", "[0].Bit0", "[1].Bit3", "[1].Bit2", "[1].Bit1", "[1].Bit0" });
            result.Schema.GetMetadata(MetadataUtils.Kinds.IsNormalized, colF, ref normalized);
            Assert.True(normalized);

            types = result.Schema.GetMetadataTypes(colG);
            Assert.Equal(types.Select(x => x.Key), new string[1] { MetadataUtils.Kinds.SlotNames });
            result.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, colG, ref slots);
            Assert.True(slots.Length == 8);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[8] { "[0].Bit3", "[0].Bit2", "[0].Bit1", "[0].Bit0", "[1].Bit3", "[1].Bit2", "[1].Bit1", "[1].Bit0" });


            types = result.Schema.GetMetadataTypes(colH);
            Assert.Equal(types.Select(x => x.Key), new string[2] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.IsNormalized });
            result.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, colH, ref slots);
            Assert.True(slots.Length == 3);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[3] { "Bit2", "Bit1", "Bit0" });
            result.Schema.GetMetadata(MetadataUtils.Kinds.IsNormalized, colH, ref normalized);
            Assert.True(normalized);
        }

        private void ValidateBagMetadata(IDataView result)
        {
            Assert.True(result.Schema.TryGetColumnIndex("CatA", out int colA));
            Assert.True(result.Schema.TryGetColumnIndex("CatB", out int colB));
            Assert.True(result.Schema.TryGetColumnIndex("CatC", out int colC));
            Assert.True(result.Schema.TryGetColumnIndex("CatD", out int colD));
            Assert.True(result.Schema.TryGetColumnIndex("CatE", out int colE));
            Assert.True(result.Schema.TryGetColumnIndex("CatF", out int colF));
            Assert.True(result.Schema.TryGetColumnIndex("CatE", out int colG));
            Assert.True(result.Schema.TryGetColumnIndex("CatF", out int colH));
            var types = result.Schema.GetMetadataTypes(colA);
            Assert.Equal(types.Select(x => x.Key), new string[1] { MetadataUtils.Kinds.SlotNames });
            VBuffer<ReadOnlyMemory<char>> slots = default;
            VBuffer<int> slotRanges = default;
            bool normalized = default;
            result.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, colA, ref slots);
            Assert.True(slots.Length == 2);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "A", "B" });

            types = result.Schema.GetMetadataTypes(colB);
            Assert.Equal(types.Select(x => x.Key), new string[2] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.IsNormalized });
            result.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, colB, ref slots);
            Assert.True(slots.Length == 1);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[1] { "C" });
            result.Schema.GetMetadata(MetadataUtils.Kinds.IsNormalized, colB, ref normalized);
            Assert.True(normalized);

            types = result.Schema.GetMetadataTypes(colC);
            Assert.Equal(types.Select(x => x.Key), new string[1] { MetadataUtils.Kinds.SlotNames });
            result.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, colC, ref slots);
            Assert.True(slots.Length == 2);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "3", "5" });


            types = result.Schema.GetMetadataTypes(colD);
            Assert.Equal(types.Select(x => x.Key), new string[2] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.IsNormalized });
            result.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, colD, ref slots);
            Assert.True(slots.Length == 2);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "6", "1" });
            result.Schema.GetMetadata(MetadataUtils.Kinds.IsNormalized, colD, ref normalized);
            Assert.True(normalized);


            types = result.Schema.GetMetadataTypes(colE);
            Assert.Equal(types.Select(x => x.Key), new string[3] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.CategoricalSlotRanges, MetadataUtils.Kinds.IsNormalized });
            result.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, colE, ref slots);
            Assert.True(slots.Length == 12);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[12] { "[0].1", "[0].2", "[0].3", "[0].4", "[0].5", "[0].6", "[1].1", "[1].2", "[1].3", "[1].4", "[1].5", "[1].6" });
            result.Schema.GetMetadata(MetadataUtils.Kinds.CategoricalSlotRanges, colE, ref slotRanges);
            Assert.True(slotRanges.Length == 4);
            Assert.Equal(slotRanges.Items().Select(x => x.Value.ToString()), new string[4] { "0", "5", "6", "11" });
            result.Schema.GetMetadata(MetadataUtils.Kinds.IsNormalized, colE, ref normalized);
            Assert.True(normalized);

            types = result.Schema.GetMetadataTypes(colF);
            Assert.Equal(types.Select(x => x.Key), new string[3] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.CategoricalSlotRanges, MetadataUtils.Kinds.IsNormalized });
            result.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, colF, ref slots);
            Assert.True(slots.Length == 2);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "1", "-1" });
            result.Schema.GetMetadata(MetadataUtils.Kinds.CategoricalSlotRanges, colF, ref slotRanges);
            Assert.True(slotRanges.Length == 2);
            Assert.Equal(slotRanges.Items().Select(x => x.Value.ToString()), new string[2] { "0", "1" });
            result.Schema.GetMetadata(MetadataUtils.Kinds.IsNormalized, colF, ref normalized);
            Assert.True(normalized);

            types = result.Schema.GetMetadataTypes(colG);
            Assert.Equal(types.Select(x => x.Key), new string[3] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.CategoricalSlotRanges, MetadataUtils.Kinds.IsNormalized });
            result.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, colG, ref slots);
            Assert.True(slots.Length == 12);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[12] { "[0].1", "[0].2", "[0].3", "[0].4", "[0].5", "[0].6", "[1].1", "[1].2", "[1].3", "[1].4", "[1].5", "[1].6" });
            result.Schema.GetMetadata(MetadataUtils.Kinds.CategoricalSlotRanges, colG, ref slotRanges);
            Assert.True(slotRanges.Length == 4);
            Assert.Equal(slotRanges.Items().Select(x => x.Value.ToString()), new string[4] { "0", "5", "6", "11" });
            result.Schema.GetMetadata(MetadataUtils.Kinds.IsNormalized, colG, ref normalized);


            types = result.Schema.GetMetadataTypes(colH);
            Assert.Equal(types.Select(x => x.Key), new string[3] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.CategoricalSlotRanges, MetadataUtils.Kinds.IsNormalized });
            result.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, colH, ref slots);
            Assert.True(slots.Length == 2);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "1", "-1" });
            result.Schema.GetMetadata(MetadataUtils.Kinds.CategoricalSlotRanges, colH, ref slotRanges);
            Assert.True(slotRanges.Length == 2);
            Assert.Equal(slotRanges.Items().Select(x => x.Value.ToString()), new string[2] { "0", "1" });
            result.Schema.GetMetadata(MetadataUtils.Kinds.IsNormalized, colH, ref normalized);
        }

        [Fact]
        public void TestCommandLine()
        {
            Assert.Equal(0, Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0} xf=Cat{col=B:A} in=f:\2.txt" }));
        }

        [Fact]
        public void TestOldSavingAndLoading()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var dataView = ComponentCreation.CreateDataView(Env, data);
            var pipe = new CategoricalEstimator(Env, new[]{
                    new CategoricalEstimator.ColumnInfo("A", "TermA"),
                    new CategoricalEstimator.ColumnInfo("B", "TermB"),
                    new CategoricalEstimator.ColumnInfo("C", "TermC")
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

    }
}
