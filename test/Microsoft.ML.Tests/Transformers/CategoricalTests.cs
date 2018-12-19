// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.Tools;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Categorical;
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
            var pipe = new OneHotEncodingEstimator(Env, new[]{
                    new OneHotEncodingEstimator.ColumnInfo("A", "CatA", OneHotEncodingTransformer.OutputKind.Bag),
                    new OneHotEncodingEstimator.ColumnInfo("A", "CatB", OneHotEncodingTransformer.OutputKind.Bin),
                    new OneHotEncodingEstimator.ColumnInfo("A", "CatC", OneHotEncodingTransformer.OutputKind.Ind),
                    new OneHotEncodingEstimator.ColumnInfo("A", "CatD", OneHotEncodingTransformer.OutputKind.Key),
                });

            TestEstimatorCore(pipe, dataView);
            Done();
        }

        [Fact]
        public void CategoricalOneHotHashEncoding()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };

            var mlContext = new MLContext();
            var dataView = ComponentCreation.CreateDataView(mlContext, data);

            var pipe = mlContext.Transforms.Categorical.OneHotHashEncoding("A", "CatA", 16, 0, OneHotEncodingTransformer.OutputKind.Bag);

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
            var data = reader.Read(dataPath);
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
                var savedData = TakeFilter.Create(Env, est.Fit(data).Transform(data).AsDynamic, 4);
                var view = new ColumnSelectingTransformer(Env, new string[]{"A", "B", "C", "D", "E" }, null, false).Transform(savedData);
                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, view, fs, keepHidden: true);
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
            var pipe = new OneHotEncodingEstimator(Env, new[] {
                new OneHotEncodingEstimator.ColumnInfo("A", "CatA", OneHotEncodingTransformer.OutputKind.Bag),
                new OneHotEncodingEstimator.ColumnInfo("B", "CatB", OneHotEncodingTransformer.OutputKind.Bag),
                new OneHotEncodingEstimator.ColumnInfo("C", "CatC", OneHotEncodingTransformer.OutputKind.Bag),
                new OneHotEncodingEstimator.ColumnInfo("D", "CatD", OneHotEncodingTransformer.OutputKind.Bag),
                new OneHotEncodingEstimator.ColumnInfo("E", "CatE", OneHotEncodingTransformer.OutputKind.Ind),
                new OneHotEncodingEstimator.ColumnInfo("F", "CatF", OneHotEncodingTransformer.OutputKind.Ind),
                new OneHotEncodingEstimator.ColumnInfo("G", "CatG", OneHotEncodingTransformer.OutputKind.Key),
                new OneHotEncodingEstimator.ColumnInfo("H", "CatH", OneHotEncodingTransformer.OutputKind.Key),
                new OneHotEncodingEstimator.ColumnInfo("A", "CatI", OneHotEncodingTransformer.OutputKind.Bin),
                new OneHotEncodingEstimator.ColumnInfo("B", "CatJ", OneHotEncodingTransformer.OutputKind.Bin),
                new OneHotEncodingEstimator.ColumnInfo("C", "CatK", OneHotEncodingTransformer.OutputKind.Bin),
                new OneHotEncodingEstimator.ColumnInfo("D", "CatL", OneHotEncodingTransformer.OutputKind.Bin) });


            var result = pipe.Fit(dataView).Transform(dataView);

            ValidateMetadata(result);
            Done();
        }


        private void ValidateMetadata(IDataView result)
        {
            VBuffer<ReadOnlyMemory<char>> slots = default;
            VBuffer<int> slotRanges = default;

            var column = result.Schema["CatA"];
            Assert.Equal(column.Metadata.Schema.Select(x => x.Name), new string[1] { MetadataUtils.Kinds.SlotNames });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 2);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "A", "B" });

            column = result.Schema["CatB"];
            Assert.Equal(column.Metadata.Schema.Select(x => x.Name), new string[2] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.IsNormalized });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 1);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[1] { "C" });
            Assert.True(column.IsNormalized());

            column = result.Schema["CatC"];
            Assert.Equal(column.Metadata.Schema.Select(x => x.Name), new string[1] { MetadataUtils.Kinds.SlotNames });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 2);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "3", "5" });

            column = result.Schema["CatD"];
            Assert.Equal(column.Metadata.Schema.Select(x => x.Name), new string[2] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.IsNormalized });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 2);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "6", "1" });
            Assert.True(column.IsNormalized());

            column = result.Schema["CatE"];
            Assert.Equal(column.Metadata.Schema.Select(x => x.Name), new string[3] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.CategoricalSlotRanges, MetadataUtils.Kinds.IsNormalized });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 12);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[12] { "[0].1", "[0].2", "[0].3", "[0].4", "[0].5", "[0].6", "[1].1", "[1].2", "[1].3", "[1].4", "[1].5", "[1].6" });
            column.Metadata.GetValue(MetadataUtils.Kinds.CategoricalSlotRanges, ref slotRanges);
            Assert.True(slotRanges.Length == 4);
            Assert.Equal(slotRanges.Items().Select(x => x.Value.ToString()), new string[4] { "0", "5", "6", "11" });
            Assert.True(column.IsNormalized());

            column = result.Schema["CatF"];
            Assert.Equal(column.Metadata.Schema.Select(x => x.Name), new string[3] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.CategoricalSlotRanges, MetadataUtils.Kinds.IsNormalized });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 2);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "1", "-1" });
            column.Metadata.GetValue(MetadataUtils.Kinds.CategoricalSlotRanges, ref slotRanges);
            Assert.True(slotRanges.Length == 2);
            Assert.Equal(slotRanges.Items().Select(x => x.Value.ToString()), new string[2] { "0", "1" });
            Assert.True(column.IsNormalized());

            column = result.Schema["CatG"];
            Assert.Equal(column.Metadata.Schema.Select(x => x.Name), new string[1] { MetadataUtils.Kinds.KeyValues});
            column.Metadata.GetValue(MetadataUtils.Kinds.KeyValues, ref slots);
            Assert.True(slots.Length == 3);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[3] {"A","D","E"});

            column = result.Schema["CatH"];
            Assert.Equal(column.Metadata.Schema.Select(x => x.Name), new string[1] { MetadataUtils.Kinds.KeyValues});
            column.Metadata.GetValue(MetadataUtils.Kinds.KeyValues, ref slots);
            Assert.True(slots.Length == 2);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "D", "E" });

            column = result.Schema["CatI"];
            Assert.Equal(column.Metadata.Schema.Select(x => x.Name), new string[1] { MetadataUtils.Kinds.SlotNames });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 6);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[6] { "[0].Bit2", "[0].Bit1", "[0].Bit0", "[1].Bit2", "[1].Bit1", "[1].Bit0" });

            column = result.Schema["CatJ"];
            Assert.Equal(column.Metadata.Schema.Select(x => x.Name), new string[2] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.IsNormalized });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 2);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "Bit1", "Bit0" });
            Assert.True(column.IsNormalized());

            column = result.Schema["CatK"];
            Assert.Equal(column.Metadata.Schema.Select(x => x.Name), new string[1] { MetadataUtils.Kinds.SlotNames });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 6);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[6] { "[0].Bit2", "[0].Bit1", "[0].Bit0", "[1].Bit2", "[1].Bit1", "[1].Bit0" });

            column = result.Schema["CatL"];
            Assert.Equal(column.Metadata.Schema.Select(x => x.Name), new string[2] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.IsNormalized });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 3);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[3] { "Bit2", "Bit1", "Bit0" });
            Assert.True(column.IsNormalized());
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
            var pipe = new OneHotEncodingEstimator(Env, new[]{
                    new OneHotEncodingEstimator.ColumnInfo("A", "TermA"),
                    new OneHotEncodingEstimator.ColumnInfo("B", "TermB"),
                    new OneHotEncodingEstimator.ColumnInfo("C", "TermC")
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
