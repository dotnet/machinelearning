// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.Tools;
using Microsoft.ML.Transforms;
using System;
using System.IO;
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
            var dataView = ComponentCreation.CreateDataView(Env, data);
            var est = ColumnSelectingEstimator.KeepColumns(Env, "A", "C");
            var transformer = est.Fit(dataView);
            var result = transformer.Transform(dataView);
            var foundColumnA = result.Schema.TryGetColumnIndex("A", out int aIdx);
            var foundColumnB = result.Schema.TryGetColumnIndex("B", out int bIdx);
            var foundColumnC = result.Schema.TryGetColumnIndex("C", out int cIdx);

            Assert.True(foundColumnA);
            Assert.Equal(0, aIdx);
            Assert.False(foundColumnB);
            Assert.Equal(0, bIdx);
            Assert.True(foundColumnC);
            Assert.Equal(1, cIdx);
        }

        [Fact]
        void TestSelectKeepWithOrder()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var dataView = ComponentCreation.CreateDataView(Env, data);

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
            Assert.Equal(0, bIdx);
            Assert.True(foundColumnC);
            Assert.Equal(0, cIdx);
        }

        [Fact]
        void TestSelectDrop()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var dataView = ComponentCreation.CreateDataView(Env, data);
            var est = ColumnSelectingEstimator.DropColumns(Env, "A", "C");
            var transformer = est.Fit(dataView);
            var result = transformer.Transform(dataView);
            var foundColumnA = result.Schema.TryGetColumnIndex("A", out int aIdx);
            var foundColumnB = result.Schema.TryGetColumnIndex("B", out int bIdx);
            var foundColumnC = result.Schema.TryGetColumnIndex("C", out int cIdx);

            Assert.False(foundColumnA);
            Assert.Equal(0, aIdx);
            Assert.True(foundColumnB);
            Assert.Equal(0, bIdx);
            Assert.False(foundColumnC);
            Assert.Equal(0, cIdx);
        }
        
        [Fact]
        void TestSelectWorkout()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var invalidData = new [] { new TestClass2 { D = 3, E = 5} };
            var dataView = ComponentCreation.CreateDataView(Env, data);
            var invalidDataView = ComponentCreation.CreateDataView(Env, invalidData);

            // Workout on keep columns
            var est = ML.Transforms.SelectColumns(new[] {"A", "B"}, null, true, false);
            TestEstimatorCore(est, validFitInput: dataView, invalidInput: invalidDataView);

            // Workout on drop columns
            est = ML.Transforms.SelectColumns(null, new[] {"A", "B"}, true, false);
            TestEstimatorCore(est, validFitInput: dataView, invalidInput: invalidDataView);

            // Workout on keep columns with ignore mismatch -- using invalid data set
            est = ML.Transforms.SelectColumns(new[] {"A", "B"}, null, true, true);
            TestEstimatorCore(est, validFitInput: invalidDataView);
        }

        [Fact]
        void TestSelectColumnsWithMissing()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var dataView = ComponentCreation.CreateDataView(Env, data);
            var est = ColumnSelectingEstimator.KeepColumns(Env, "D", "G");
            Assert.Throws<ArgumentOutOfRangeException>(() => est.Fit(dataView));
        }

        [Fact]
        void TestSelectColumnsWithSameName()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var dataView = ComponentCreation.CreateDataView(Env, data);
            var est = new CopyColumnsEstimator(Env, new[] {("A", "A"), ("B", "B")});
            var chain = est.Append(ColumnSelectingEstimator.KeepColumns(Env, "C", "A"));
            var transformer = chain.Fit(dataView);
            var result = transformer.Transform(dataView);

            // Copied columns should equal AABBC, however we chose to keep A and C
            // so the result is AC
            Assert.Equal(2, result.Schema.ColumnCount);
            var foundColumnA = result.Schema.TryGetColumnIndex("A", out int aIdx);
            var foundColumnB = result.Schema.TryGetColumnIndex("B", out int bIdx);
            var foundColumnC = result.Schema.TryGetColumnIndex("C", out int cIdx);
            Assert.True(foundColumnA);
            Assert.Equal(1, aIdx);
            Assert.False(foundColumnB);
            Assert.Equal(0, bIdx);
            Assert.True(foundColumnC);
            Assert.Equal(0, cIdx);
        }

        [Fact]
        void TestSelectColumnsWithKeepHidden()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var dataView = ComponentCreation.CreateDataView(Env, data);
            var est = new CopyColumnsEstimator(Env, new[] {("A", "A"), ("B", "B")});
            var chain = est.Append(ML.Transforms.SelectColumns(new[] {"B", "A" }, null, true));
            var transformer = chain.Fit(dataView);
            var result = transformer.Transform(dataView);

            // Input for SelectColumns should be AABBC, we chose to keep A and B
            // and keep hidden columns is true, therefore the output should be AABB
            Assert.Equal(4, result.Schema.ColumnCount);
            var foundColumnA = result.Schema.TryGetColumnIndex("A", out int aIdx);
            var foundColumnB = result.Schema.TryGetColumnIndex("B", out int bIdx);
            var foundColumnC = result.Schema.TryGetColumnIndex("C", out int cIdx);
            Assert.True(foundColumnA);
            Assert.Equal(3, aIdx);
            Assert.True(foundColumnB);
            Assert.Equal(1, bIdx);
            Assert.False(foundColumnC);
            Assert.Equal(0, cIdx);
        }

        [Fact]
        void TestSelectColumnsDropWithKeepHidden()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var dataView = ComponentCreation.CreateDataView(Env, data);
            var est = new CopyColumnsEstimator(Env, new[] {("A", "A"), ("B", "B")});
            var chain = est.Append(ML.Transforms.SelectColumns(null, new[] { "A" }, true));
            var transformer = chain.Fit(dataView);
            var result = transformer.Transform(dataView);

            // Input for SelectColumns should be AABBC, we chose to drop A
            // and keep hidden columns is true, therefore the output should be BBC
            Assert.Equal(3, result.Schema.ColumnCount);
            var foundColumnA = result.Schema.TryGetColumnIndex("A", out int aIdx);
            var foundColumnB = result.Schema.TryGetColumnIndex("B", out int bIdx);
            var foundColumnC = result.Schema.TryGetColumnIndex("C", out int cIdx);
            Assert.False(foundColumnA);
            Assert.Equal(0, aIdx);
            Assert.True(foundColumnB);
            Assert.Equal(1, bIdx);
            Assert.True(foundColumnC);
            Assert.Equal(2, cIdx);
        }

        [Fact]
        void TestSelectWithKeepAndDropSet()
        {
            // Setting both keep and drop is not allowed.
            var test = new string[]{ "D", "G"};
            Assert.Throws<InvalidOperationException>(() => ML.Transforms.SelectColumns(test, test));
        }

        [Fact]
        void TestSelectNoKeepAndDropSet()
        {
            // Passing null to both keep and drop is not allowed.
            Assert.Throws<InvalidOperationException>(() => ML.Transforms.SelectColumns(null, null));
        }

        [Fact]
        void TestSelectSavingAndLoading()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var dataView = ComponentCreation.CreateDataView(Env, data);
            var est = ColumnSelectingEstimator.KeepColumns(Env, "A", "B");
            var transformer = est.Fit(dataView);
            using (var ms = new MemoryStream())
            {
                transformer.SaveTo(Env, ms);
                ms.Position = 0;
                var loadedTransformer = TransformerChain.LoadFrom(Env, ms);
                var result = loadedTransformer.Transform(dataView);
                Assert.Equal(2, result.Schema.ColumnCount);
                Assert.Equal("A", result.Schema.GetColumnName(0));
                Assert.Equal("B", result.Schema.GetColumnName(1));
            }
        }

        [Fact]
        void TestSelectSavingAndLoadingWithNoKeepHidden()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var dataView = ComponentCreation.CreateDataView(Env, data);
            var est = new CopyColumnsEstimator(Env, new[] {("A", "A"), ("B", "B")}).Append(
                      ML.Transforms.SelectColumns(new[] { "A", "B" }, null, false));
            var transformer = est.Fit(dataView);
            using (var ms = new MemoryStream())
            {
                transformer.SaveTo(Env, ms);
                ms.Position = 0;
                var loadedTransformer = TransformerChain.LoadFrom(Env, ms);
                var result = loadedTransformer.Transform(dataView);
                Assert.Equal(2, result.Schema.ColumnCount);
                Assert.Equal("A", result.Schema.GetColumnName(0));
                Assert.Equal("B", result.Schema.GetColumnName(1));
            }
        }

        [Fact]
        void TestSelectBackCompatDropColumns()
        {
            // Model generated with: xf=drop{col=A} 
            // Expected output: Features Label B C
            var data = new[] { new TestClass3() { Label="foo", Features="bar", A = 1, B = 2, C = 3, } };
            var dataView = ComponentCreation.CreateDataView(Env, data);
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
                Assert.Equal(0, aIdx);
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
            var dataView = ComponentCreation.CreateDataView(Env, data);
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
                Assert.Equal(0, cIdx);
            }
        }
        
        [Fact]
        void TestSelectBackCompatChooseColumns()
        {
            // Model generated with: xf=choose{col=Label col=Features col=A col=B}
            // Output expected is Label Features A B
            var data = new[] { new TestClass3() { Label="foo", Features="bar", A = 1, B = 2, C = 3, } };
            var dataView = ComponentCreation.CreateDataView(Env, data);
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
                Assert.Equal(0, cIdx);
            }
        }

        [Fact]
        void TestSelectBackCompatChooseColumnsWithKeep()
        {
            // Model generated with: xf=copy{col=A:A col=B:B} xf=choose{col=Label col=Features col=A col=B hidden=keep}
            // Output expected is Label Features A A B B
            var data = new[] { new TestClass3() { Label="foo", Features="bar", A = 1, B = 2, C = 3, } };
            var dataView = ComponentCreation.CreateDataView(Env, data);
            string chooseModelPath = GetDataPath("backcompat/choose-keep-model.zip");
            using (FileStream fs = File.OpenRead(chooseModelPath))
            {
                var result = ModelFileUtils.LoadTransforms(Env, dataView, fs);
                Assert.Equal(6, result.Schema.ColumnCount);
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
                Assert.Equal(0, cIdx);
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
