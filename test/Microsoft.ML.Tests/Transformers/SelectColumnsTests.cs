// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.Tools;
using Microsoft.ML.Transforms;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public class SelectColumnsTransformsTests : TestDataPipeBase
    {
        class TestClass
        {
            public int A;
            public int B;
            public int C;
        }

        class TestClass2
        {
            public int D;
            public int E;
        }
        class TestClass3
        {
            public string Label;
            public string Features;
            public int A;
            public int B;
            public int C;
        };

        public SelectColumnsTransformsTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        void TestSelectKeep()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var dataView = ML.Data.LoadFromEnumerable(data);
            var est = ColumnSelectingEstimator.KeepColumns(Env, "A", "C");
            var transformer = est.Fit(dataView);
            var result = transformer.Transform(dataView);
            var foundColumnA = result.Schema.TryGetColumnIndex("A", out int aIdx);
            var foundColumnB = result.Schema.TryGetColumnIndex("B", out int bIdx);
            var foundColumnC = result.Schema.TryGetColumnIndex("C", out int cIdx);

            Assert.True(foundColumnA);
            Assert.Equal(0, aIdx);
            Assert.False(foundColumnB);
            Assert.True(foundColumnC);
            Assert.Equal(1, cIdx);
        }

        [Fact]
        void TestSelectKeepWithOrder()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var dataView = ML.Data.LoadFromEnumerable(data);

            // Expected output will be CA
            var est = ColumnSelectingEstimator.KeepColumns(Env, "C", "A");
            var transformer = est.Fit(dataView);
            var result = transformer.Transform(dataView);
            var foundColumnA = result.Schema.TryGetColumnIndex("A", out int aIdx);
            var foundColumnB = result.Schema.TryGetColumnIndex("B", out int bIdx);
            var foundColumnC = result.Schema.TryGetColumnIndex("C", out int cIdx);

            Assert.True(foundColumnA);
            Assert.Equal(1, aIdx);
            Assert.False(foundColumnB);
            Assert.True(foundColumnC);
            Assert.Equal(0, cIdx);
        }

        [Fact]
        void TestSelectDrop()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var dataView = ML.Data.LoadFromEnumerable(data);
            var est = ColumnSelectingEstimator.DropColumns(Env, "A", "C");
            var transformer = est.Fit(dataView);
            var result = transformer.Transform(dataView);
            var foundColumnA = result.Schema.TryGetColumnIndex("A", out int aIdx);
            var foundColumnB = result.Schema.TryGetColumnIndex("B", out int bIdx);
            var foundColumnC = result.Schema.TryGetColumnIndex("C", out int cIdx);

            Assert.False(foundColumnA);
            Assert.True(foundColumnB);
            Assert.Equal(0, bIdx);
            Assert.False(foundColumnC);
        }
        
        [Fact]
        void TestSelectWorkout()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var invalidData = new [] { new TestClass2 { D = 3, E = 5} };
            var dataView = ML.Data.LoadFromEnumerable(data);
            var invalidDataView = ML.Data.LoadFromEnumerable(invalidData);

            // Workout on keep columns
            var est = ML.Transforms.SelectColumns(new[] {"A", "B"});
            TestEstimatorCore(est, validFitInput: dataView, invalidInput: invalidDataView);

            // Workout on select columns with hidden: true
            est = ML.Transforms.SelectColumns(new[] {"A", "B"}, true);
            TestEstimatorCore(est, validFitInput: dataView, invalidInput: invalidDataView);
        }

        [Fact]
        void TestSelectColumnsWithMissing()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var dataView = ML.Data.LoadFromEnumerable(data);
            var est = ColumnSelectingEstimator.KeepColumns(Env, "D", "G");
            Assert.Throws<ArgumentOutOfRangeException>(() => est.Fit(dataView));
        }

        [Fact]
        void TestSelectColumnsWithSameName()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var dataView = ML.Data.LoadFromEnumerable(data);
            var est = new ColumnCopyingEstimator(Env, new[] {("A", "A"), ("B", "B")});
            var chain = est.Append(ColumnSelectingEstimator.KeepColumns(Env, "C", "A"));
            var transformer = chain.Fit(dataView);
            var result = transformer.Transform(dataView);

            // Copied columns should equal AABBC, however we chose to keep A and C
            // so the result is AC
            Assert.Equal(2, result.Schema.Count);
            var foundColumnA = result.Schema.TryGetColumnIndex("A", out int aIdx);
            var foundColumnB = result.Schema.TryGetColumnIndex("B", out int bIdx);
            var foundColumnC = result.Schema.TryGetColumnIndex("C", out int cIdx);
            Assert.True(foundColumnA);
            Assert.Equal(1, aIdx);
            Assert.False(foundColumnB);
            Assert.True(foundColumnC);
            Assert.Equal(0, cIdx);
        }

        [Fact]
        void TestSelectColumnsWithKeepHidden()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var dataView = ML.Data.LoadFromEnumerable(data);
            var est = new ColumnCopyingEstimator(Env, new[] {("A", "A"), ("B", "B")});
            var chain = est.Append(ML.Transforms.SelectColumns(new[] {"B", "A" }, true));
            var transformer = chain.Fit(dataView);
            var result = transformer.Transform(dataView);

            // Input for SelectColumns should be AABBC, we chose to keep A and B
            // and keep hidden columns is true, therefore the output should be AABB
            Assert.Equal(4, result.Schema.Count);
            var foundColumnA = result.Schema.TryGetColumnIndex("A", out int aIdx);
            var foundColumnB = result.Schema.TryGetColumnIndex("B", out int bIdx);
            var foundColumnC = result.Schema.TryGetColumnIndex("C", out int cIdx);
            Assert.True(foundColumnA);
            Assert.Equal(3, aIdx);
            Assert.True(foundColumnB);
            Assert.Equal(1, bIdx);
            Assert.False(foundColumnC);
        }

        [Fact]
        void TestSelectSavingAndLoading()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var dataView = ML.Data.LoadFromEnumerable(data);
            var est = ColumnSelectingEstimator.KeepColumns(Env, "A", "B");
            var transformer = est.Fit(dataView);
            using (var ms = new MemoryStream())
            {
                ML.Model.Save(transformer, null, ms);
                ms.Position = 0;
                var loadedTransformer = ML.Model.Load(ms, out var schema);
                var result = loadedTransformer.Transform(dataView);
                Assert.Equal(2, result.Schema.Count);
                Assert.Equal("A", result.Schema[0].Name);
                Assert.Equal("B", result.Schema[1].Name);
            }
        }

        [Fact]
        void TestSelectSavingAndLoadingWithNoKeepHidden()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var dataView = ML.Data.LoadFromEnumerable(data);
            var est = new ColumnCopyingEstimator(Env, new[] {("A", "A"), ("B", "B")}).Append(
                      ML.Transforms.SelectColumns(new[] { "A", "B" }, false));
            var transformer = est.Fit(dataView);
            using (var ms = new MemoryStream())
            {
                ML.Model.Save(transformer, null, ms);
                ms.Position = 0;
                var loadedTransformer = ML.Model.Load(ms, out var schema);
                var result = loadedTransformer.Transform(dataView);
                Assert.Equal(2, result.Schema.Count);
                Assert.Equal("A", result.Schema[0].Name);
                Assert.Equal("B", result.Schema[1].Name);
            }
        }

        [Fact]
        void TestSelectBackCompatDropColumns()
        {
            // Model generated with: xf=drop{col=A} 
            // Expected output: Features Label B C
            var data = new[] { new TestClass3() { Label="foo", Features="bar", A = 1, B = 2, C = 3, } };
            var dataView = ML.Data.LoadFromEnumerable(data);
            string dropModelPath = GetDataPath("backcompat/drop-model.zip");
            using (FileStream fs = File.OpenRead(dropModelPath))
            {
                var result = ModelFileUtils.LoadTransforms(Env, dataView, fs);
                var foundColumnFeature = result.Schema.TryGetColumnIndex("Features", out int featureIdx);
                var foundColumnLabel = result.Schema.TryGetColumnIndex("Label", out int labelIdx);
                var foundColumnA = result.Schema.TryGetColumnIndex("A", out int aIdx);
                var foundColumnB = result.Schema.TryGetColumnIndex("B", out int bIdx);
                var foundColumnC = result.Schema.TryGetColumnIndex("C", out int cIdx);
                Assert.True(foundColumnLabel);
                Assert.Equal(0, labelIdx);
                Assert.True(foundColumnFeature);
                Assert.Equal(1, featureIdx);
                Assert.False(foundColumnA);
                Assert.True(foundColumnB);
                Assert.Equal(2, bIdx);
                Assert.True(foundColumnC);
                Assert.Equal(3, cIdx);
            }
        }

        [Fact]
        void TestSelectBackCompatKeepColumns()
        {
            // Model generated with: xf=keep{col=Label col=Features col=A col=B}
            // Expected output: Label Features A B
            var data = new[] { new TestClass3() { Label="foo", Features="bar", A = 1, B = 2, C = 3, } };
            var dataView = ML.Data.LoadFromEnumerable(data);
            string dropModelPath = GetDataPath("backcompat/keep-model.zip");
            using (FileStream fs = File.OpenRead(dropModelPath))
            {
                var result = ModelFileUtils.LoadTransforms(Env, dataView, fs);
                var foundColumnFeature = result.Schema.TryGetColumnIndex("Features", out int featureIdx);
                var foundColumnLabel = result.Schema.TryGetColumnIndex("Label", out int labelIdx);
                var foundColumnA = result.Schema.TryGetColumnIndex("A", out int aIdx);
                var foundColumnB = result.Schema.TryGetColumnIndex("B", out int bIdx);
                var foundColumnC = result.Schema.TryGetColumnIndex("C", out int cIdx);
                Assert.True(foundColumnLabel);
                Assert.Equal(0, labelIdx);
                Assert.True(foundColumnFeature);
                Assert.Equal(1, featureIdx);
                Assert.True(foundColumnA);
                Assert.Equal(2, aIdx);
                Assert.True(foundColumnB);
                Assert.Equal(3, bIdx);
                Assert.False(foundColumnC);
            }
        }
        
        [Fact]
        void TestSelectBackCompatChooseColumns()
        {
            // Model generated with: xf=choose{col=Label col=Features col=A col=B}
            // Output expected is Label Features A B
            var data = new[] { new TestClass3() { Label="foo", Features="bar", A = 1, B = 2, C = 3, } };
            var dataView = ML.Data.LoadFromEnumerable(data);
            string dropModelPath = GetDataPath("backcompat/choose-model.zip");
            using (FileStream fs = File.OpenRead(dropModelPath))
            {
                var result = ModelFileUtils.LoadTransforms(Env, dataView, fs);
                var foundColumnFeature = result.Schema.TryGetColumnIndex("Features", out int featureIdx);
                var foundColumnLabel = result.Schema.TryGetColumnIndex("Label", out int labelIdx);
                var foundColumnA = result.Schema.TryGetColumnIndex("A", out int aIdx);
                var foundColumnB = result.Schema.TryGetColumnIndex("B", out int bIdx);
                var foundColumnC = result.Schema.TryGetColumnIndex("C", out int cIdx);
                Assert.True(foundColumnLabel);
                Assert.Equal(0, labelIdx);
                Assert.True(foundColumnFeature);
                Assert.Equal(1, featureIdx);
                Assert.True(foundColumnA);
                Assert.Equal(2, aIdx);
                Assert.True(foundColumnB);
                Assert.Equal(3, bIdx);
                Assert.False(foundColumnC);
            }
        }

        [Fact]
        void TestSelectBackCompatChooseColumnsWithKeep()
        {
            // Model generated with: xf=copy{col=A:A col=B:B} xf=choose{col=Label col=Features col=A col=B hidden=keep}
            // Output expected is Label Features A A B B
            var data = new[] { new TestClass3() { Label="foo", Features="bar", A = 1, B = 2, C = 3, } };
            var dataView = ML.Data.LoadFromEnumerable(data);
            string chooseModelPath = GetDataPath("backcompat/choose-keep-model.zip");
            using (FileStream fs = File.OpenRead(chooseModelPath))
            {
                var result = ModelFileUtils.LoadTransforms(Env, dataView, fs);
                Assert.Equal(6, result.Schema.Count);
                var foundColumnFeature = result.Schema.TryGetColumnIndex("Features", out int featureIdx);
                var foundColumnLabel = result.Schema.TryGetColumnIndex("Label", out int labelIdx);
                var foundColumnA = result.Schema.TryGetColumnIndex("A", out int aIdx);
                var foundColumnB = result.Schema.TryGetColumnIndex("B", out int bIdx);
                var foundColumnC = result.Schema.TryGetColumnIndex("C", out int cIdx);
                Assert.True(foundColumnLabel);
                Assert.Equal(0, labelIdx);
                Assert.True(foundColumnFeature);
                Assert.Equal(1, featureIdx);
                Assert.True(foundColumnA);
                Assert.Equal(3, aIdx);
                Assert.True(foundColumnB);
                Assert.Equal(5, bIdx);
                Assert.False(foundColumnC);
            }
        }

        [Fact]
        void TestCommandLineWithKeep()
        {
            Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0 col=B:R4:1 col=C:R4:2} xf=select{keepcol=A keepcol=B} in=f:\1.txt" }), (int)0);
        }

        [Fact]
        void TestCommandLineWithDrop()
        {
            Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0 col=B:R4:1 col=C:R4:2} xf=select{dropcol=A dropcol=B} in=f:\1.txt" }), (int)0);
        }

        [Fact]
        void TestCommandLineKeepWithoutHidden()
        {
            Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0 col=B:R4:1 col=C:R4:2} xf=select{keepcol=A keepcol=B hidden=-} in=f:\1.txt" }), (int)0);
        }

        [Fact]
        void TestCommandLineKeepWithIgnoreMismatch()
        {
            Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0 col=B:R4:1 col=C:R4:2} xf=select{keepcol=A keepcol=B ignore=-} in=f:\1.txt" }), (int)0);
        }
    }
}
