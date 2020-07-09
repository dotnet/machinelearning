// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFrameworkCommon;
using Microsoft.ML.Tools;
using Microsoft.ML.Transforms;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public class CategoricalTests : TestDataPipeBase
    {
        public CategoricalTests(ITestOutputHelper output) : base(output)
        {
        }

        private sealed class TestClass
        {
            public int A;
            [VectorType(2)]
            public int[] B;
            public int[] C;

        }

        private sealed class TestClassWithLabel
        {
            public int A;
            public int B;
            public bool Label;
        }

        private sealed class TestMeta
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

        private sealed class TestStringClass
        {
            public string A;
        }

        [Fact]
        public void CategoricalWorkout()
        {
            var data = new[] {
                new TestClass() { A = 1, B = new int[2] { 2, 3 }, C = new int[2] { 3, 4 } },
                new TestClass() { A = 4, B = new int[2] { 2, 4 }, C = new int[3] { 2, 4, 3 } }
            };

            var dataView = ML.Data.LoadFromEnumerable(data);
            var pipe = ML.Transforms.Categorical.OneHotEncoding(new[]{
                    new OneHotEncodingEstimator.ColumnOptions("CatA", "A", OneHotEncodingEstimator.OutputKind.Bag),
                    new OneHotEncodingEstimator.ColumnOptions("CatB", "A", OneHotEncodingEstimator.OutputKind.Binary),
                    new OneHotEncodingEstimator.ColumnOptions("CatC", "A", OneHotEncodingEstimator.OutputKind.Indicator),
                    new OneHotEncodingEstimator.ColumnOptions("CatD", "A", OneHotEncodingEstimator.OutputKind.Key),
                    new OneHotEncodingEstimator.ColumnOptions("CatVA", "B", OneHotEncodingEstimator.OutputKind.Bag),
                    new OneHotEncodingEstimator.ColumnOptions("CatVB", "B", OneHotEncodingEstimator.OutputKind.Binary),
                    new OneHotEncodingEstimator.ColumnOptions("CatVC", "B", OneHotEncodingEstimator.OutputKind.Indicator),
                    new OneHotEncodingEstimator.ColumnOptions("CatVD", "B", OneHotEncodingEstimator.OutputKind.Key),
                    new OneHotEncodingEstimator.ColumnOptions("CatVVA", "C", OneHotEncodingEstimator.OutputKind.Bag),
                    new OneHotEncodingEstimator.ColumnOptions("CatVVB", "C", OneHotEncodingEstimator.OutputKind.Binary),
                    new OneHotEncodingEstimator.ColumnOptions("CatVVC", "C", OneHotEncodingEstimator.OutputKind.Indicator),
                    new OneHotEncodingEstimator.ColumnOptions("CatVVD", "C", OneHotEncodingEstimator.OutputKind.Key),
                });

            TestEstimatorCore(pipe, dataView);
            var outputPath = GetOutputPath("Categorical", "oneHot.tsv");
            var savedData = pipe.Fit(dataView).Transform(dataView);

            using (var fs = File.Create(outputPath))
                ML.Data.SaveAsText(savedData, fs, headerRow: true, keepHidden: true);
            CheckEquality("Categorical", "oneHot.tsv");
            Done();
        }

        [Fact]
        public void CategoricalOneHotEncodingVector()
        {
            var data = new[] {
                new TestClassWithLabel() { A = 301, B = 2000, Label = true },
                new TestClassWithLabel() { A = 450, B = 3000, Label = true },
                new TestClassWithLabel() { A = -300, B = 4000, Label = true },
                new TestClassWithLabel() { A = 300, B = 2000, Label = false },
                new TestClassWithLabel() { A = 115, B = 2000, Label = false },
                new TestClassWithLabel() { A = 115, B = 2000, Label = false }};

            var mlContext = new MLContext(1);
            var dataView = mlContext.Data.LoadFromEnumerable(data);
            var pipe = mlContext.Transforms.Conversion.ConvertType("A", outputKind: DataKind.Single)
                .Append(mlContext.Transforms.Conversion.ConvertType("B", outputKind: DataKind.Single))
                .Append(mlContext.Transforms.Concatenate("Features", new string[] { "A", "B" }))
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Label"))
                .Append(mlContext.Transforms.NormalizeSupervisedBinning("Features", fixZero: false, maximumBinCount: 5, labelColumnName: "Label"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Features", outputKind: OneHotEncodingEstimator.OutputKind.Indicator));

            TestEstimatorCore(pipe, dataView);
            Done();
        }

        /// <summary>
        /// In which we take a categorical value and map it to a vector, but we get the mapping from a side data view
        /// rather than the data we are fitting.
        /// </summary>
        [Fact]
        public void CategoricalOneHotEncodingFromSideData()
        {
            // In this case, whatever the value of the input, the term mapping should come from the optional side data if specified.
            var data = new[] { new TestStringClass() { A = "Stay" }, new TestStringClass() { A = "awhile and listen" } };

            var mlContext = new MLContext(1);
            var dataView = mlContext.Data.LoadFromEnumerable(data);

            var sideDataBuilder = new ArrayDataViewBuilder(mlContext);
            sideDataBuilder.AddColumn("Hello", "hello", "my", "friend");
            var sideData = sideDataBuilder.GetDataView();

            var ci = new OneHotEncodingEstimator.ColumnOptions("CatA", "A", OneHotEncodingEstimator.OutputKind.Bag);
            var pipe = mlContext.Transforms.Categorical.OneHotEncoding(new[] { ci }, sideData);

            var output = pipe.Fit(dataView).Transform(dataView);

            VBuffer<ReadOnlyMemory<char>> slotNames = default;
            output.Schema["CatA"].GetSlotNames(ref slotNames);

            Assert.Equal(3, slotNames.Length);
            Assert.Equal("hello", slotNames.GetItemOrDefault(0).ToString());
            Assert.Equal("my", slotNames.GetItemOrDefault(1).ToString());
            Assert.Equal("friend", slotNames.GetItemOrDefault(2).ToString());

            Done();
        }

        [Fact]
        public void Categorical()
        {
            string dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var data = ML.Data.LoadFromTextFile(dataPath, new[] {
                new TextLoader.Column("ScalarString", DataKind.String, 1),
                new TextLoader.Column("VectorString", DataKind.String, 1, 4)
            });
            var wrongCollection = new[] { new TestClass() { A = 1, B = new int[2] { 2, 3 } }, new TestClass() { A = 4, B = new int[2] { 2, 4 } } };

            var invalidData = ML.Data.LoadFromEnumerable(wrongCollection);
            var est = ML.Transforms.Text.TokenizeIntoWords("VarVectorString", "ScalarString")
                .Append(ML.Transforms.Categorical.OneHotEncoding("A", "ScalarString", outputKind: OneHotEncodingEstimator.OutputKind.Indicator))
                .Append(ML.Transforms.Categorical.OneHotEncoding("B", "VectorString", outputKind: OneHotEncodingEstimator.OutputKind.Indicator))
                .Append(ML.Transforms.Categorical.OneHotEncoding("C", "VectorString", outputKind: OneHotEncodingEstimator.OutputKind.Bag))
                .Append(ML.Transforms.Categorical.OneHotEncoding("D", "ScalarString", outputKind: OneHotEncodingEstimator.OutputKind.Binary))
                .Append(ML.Transforms.Categorical.OneHotEncoding("E", "VectorString", outputKind: OneHotEncodingEstimator.OutputKind.Binary));

            TestEstimatorCore(est, data, invalidInput: invalidData);

            var outputPath = GetOutputPath("Categorical", "featurized.tsv");
            var savedData = ML.Data.TakeRows(est.Fit(data).Transform(data), 4);
            var view = ML.Transforms.SelectColumns("A", "B", "C", "D", "E").Fit(savedData).Transform(savedData);
            using (var fs = File.Create(outputPath))
                ML.Data.SaveAsText(view, fs, headerRow: true, keepHidden: true);

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


            var dataView = ML.Data.LoadFromEnumerable(data);
            var pipe = ML.Transforms.Categorical.OneHotEncoding(new[] {
                new OneHotEncodingEstimator.ColumnOptions("CatA", "A", OneHotEncodingEstimator.OutputKind.Bag),
                new OneHotEncodingEstimator.ColumnOptions("CatB", "B", OneHotEncodingEstimator.OutputKind.Bag),
                new OneHotEncodingEstimator.ColumnOptions("CatC", "C", OneHotEncodingEstimator.OutputKind.Bag),
                new OneHotEncodingEstimator.ColumnOptions("CatD", "D", OneHotEncodingEstimator.OutputKind.Bag),
                new OneHotEncodingEstimator.ColumnOptions("CatE", "E",OneHotEncodingEstimator.OutputKind.Indicator),
                new OneHotEncodingEstimator.ColumnOptions("CatF", "F", OneHotEncodingEstimator.OutputKind.Indicator),
                new OneHotEncodingEstimator.ColumnOptions("CatG", "G", OneHotEncodingEstimator.OutputKind.Key),
                new OneHotEncodingEstimator.ColumnOptions("CatH", "H", OneHotEncodingEstimator.OutputKind.Key),
                new OneHotEncodingEstimator.ColumnOptions("CatI", "A", OneHotEncodingEstimator.OutputKind.Binary),
                new OneHotEncodingEstimator.ColumnOptions("CatJ", "B", OneHotEncodingEstimator.OutputKind.Binary),
                new OneHotEncodingEstimator.ColumnOptions("CatK", "C", OneHotEncodingEstimator.OutputKind.Binary),
                new OneHotEncodingEstimator.ColumnOptions("CatL", "D", OneHotEncodingEstimator.OutputKind.Binary) });


            var result = pipe.Fit(dataView).Transform(dataView);

            ValidateMetadata(result);
            Done();
        }


        private void ValidateMetadata(IDataView result)
        {
            VBuffer<ReadOnlyMemory<char>> slots = default;
            VBuffer<int> slotRanges = default;

            var column = result.Schema["CatA"];
            Assert.Equal(column.Annotations.Schema.Select(x => x.Name), new string[1] { AnnotationUtils.Kinds.SlotNames });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 2);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "A", "B" });

            column = result.Schema["CatB"];
            Assert.Equal(column.Annotations.Schema.Select(x => x.Name), new string[2] { AnnotationUtils.Kinds.SlotNames, AnnotationUtils.Kinds.IsNormalized });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 1);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[1] { "C" });
            Assert.True(column.IsNormalized());

            column = result.Schema["CatC"];
            Assert.Equal(column.Annotations.Schema.Select(x => x.Name), new string[1] { AnnotationUtils.Kinds.SlotNames });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 2);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "3", "5" });

            column = result.Schema["CatD"];
            Assert.Equal(column.Annotations.Schema.Select(x => x.Name), new string[2] { AnnotationUtils.Kinds.SlotNames, AnnotationUtils.Kinds.IsNormalized });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 2);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "6", "1" });
            Assert.True(column.IsNormalized());

            column = result.Schema["CatE"];
            Assert.Equal(column.Annotations.Schema.Select(x => x.Name), new string[3] { AnnotationUtils.Kinds.SlotNames, AnnotationUtils.Kinds.CategoricalSlotRanges, AnnotationUtils.Kinds.IsNormalized });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 12);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[12] { "[0].1", "[0].2", "[0].3", "[0].4", "[0].5", "[0].6", "[1].1", "[1].2", "[1].3", "[1].4", "[1].5", "[1].6" });
            column.Annotations.GetValue(AnnotationUtils.Kinds.CategoricalSlotRanges, ref slotRanges);
            Assert.True(slotRanges.Length == 4);
            Assert.Equal(slotRanges.Items().Select(x => x.Value.ToString()), new string[4] { "0", "5", "6", "11" });
            Assert.True(column.IsNormalized());

            column = result.Schema["CatF"];
            Assert.Equal(column.Annotations.Schema.Select(x => x.Name), new string[3] { AnnotationUtils.Kinds.SlotNames, AnnotationUtils.Kinds.CategoricalSlotRanges, AnnotationUtils.Kinds.IsNormalized });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 2);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "1", "-1" });
            column.Annotations.GetValue(AnnotationUtils.Kinds.CategoricalSlotRanges, ref slotRanges);
            Assert.True(slotRanges.Length == 2);
            Assert.Equal(slotRanges.Items().Select(x => x.Value.ToString()), new string[2] { "0", "1" });
            Assert.True(column.IsNormalized());

            column = result.Schema["CatG"];
            Assert.Equal(column.Annotations.Schema.Select(x => x.Name), new string[1] { AnnotationUtils.Kinds.KeyValues });
            column.GetKeyValues(ref slots);
            Assert.True(slots.Length == 3);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[3] { "A", "D", "E" });

            column = result.Schema["CatH"];
            Assert.Equal(column.Annotations.Schema.Select(x => x.Name), new string[1] { AnnotationUtils.Kinds.KeyValues });
            column.GetKeyValues(ref slots);
            Assert.True(slots.Length == 2);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "D", "E" });

            column = result.Schema["CatI"];
            Assert.Equal(column.Annotations.Schema.Select(x => x.Name), new string[1] { AnnotationUtils.Kinds.SlotNames });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 6);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[6] { "[0].Bit2", "[0].Bit1", "[0].Bit0", "[1].Bit2", "[1].Bit1", "[1].Bit0" });

            column = result.Schema["CatJ"];
            Assert.Equal(column.Annotations.Schema.Select(x => x.Name), new string[2] { AnnotationUtils.Kinds.SlotNames, AnnotationUtils.Kinds.IsNormalized });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 2);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "Bit1", "Bit0" });
            Assert.True(column.IsNormalized());

            column = result.Schema["CatK"];
            Assert.Equal(column.Annotations.Schema.Select(x => x.Name), new string[1] { AnnotationUtils.Kinds.SlotNames });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 6);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[6] { "[0].Bit2", "[0].Bit1", "[0].Bit0", "[1].Bit2", "[1].Bit1", "[1].Bit0" });

            column = result.Schema["CatL"];
            Assert.Equal(column.Annotations.Schema.Select(x => x.Name), new string[2] { AnnotationUtils.Kinds.SlotNames, AnnotationUtils.Kinds.IsNormalized });
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
            var data = new[] {
                new TestClass() { A = 1, B = new int[2] { 2, 3 }, C = new int[2] { 3, 4 } },
                new TestClass() { A = 4, B = new int[2] { 2, 4 }, C = new int[3] { 2, 4, 3 } }
            };
            var dataView = ML.Data.LoadFromEnumerable(data);
            var pipe = ML.Transforms.Categorical.OneHotEncoding(new[]{
                    new OneHotEncodingEstimator.ColumnOptions("CatA", "A"),
                    new OneHotEncodingEstimator.ColumnOptions("CatB", "B"),
                    new OneHotEncodingEstimator.ColumnOptions("CatC", "C")
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
