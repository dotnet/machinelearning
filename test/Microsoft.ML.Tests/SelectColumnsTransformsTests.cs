// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.Tools;
using Microsoft.ML.TestFramework;
using Microsoft.ML.Transforms;
using System;
using System.IO;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
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

        public SelectColumnsTransformsTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        void TestSelect()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            using (var env = new ConsoleEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                var est = new SelectColumnsEstimator(env, "A", "C");
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
        }
        
        [Fact]
        void TestSelectWorkout()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var invalidData = new [] { new TestClass2 { D = 3, E = 5} };
            using (var env = new ConsoleEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                var invalidDataView = ComponentCreation.CreateDataView(env, invalidData);

                // Workout on keep columns
                var est = new SelectColumnsEstimator(env,  new[] {"A", "B"}, null, true, false);
                TestEstimatorCore(est, validFitInput: dataView, invalidInput: invalidDataView);

                // Workout on drop columns
                est = new SelectColumnsEstimator(env,  null, new[] {"A", "B"}, true, false);
                TestEstimatorCore(est, validFitInput: dataView, invalidInput: invalidDataView);

                // Workout on keep columns with ignore mismatch -- using invalid data set
                est = new SelectColumnsEstimator(env,  null, new[] {"A", "B"}, true);
                TestEstimatorCore(est, validFitInput: invalidDataView);
            }
        }

        [Fact]
        void TestSelectColumnsNoMatch()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            using (var env = new ConsoleEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                var est = new SelectColumnsEstimator(env, new[] {"D", "G"});
                var transformer = est.Fit(dataView);
                var result = transformer.Transform(dataView);
                var foundColumnA = result.Schema.TryGetColumnIndex("A", out int aIdx);
                var foundColumnB = result.Schema.TryGetColumnIndex("B", out int bIdx);
                var foundColumnC = result.Schema.TryGetColumnIndex("C", out int cIdx);

                Assert.False(foundColumnA);
                Assert.Equal(0, aIdx);
                Assert.False(foundColumnB);
                Assert.Equal(0, bIdx);
                Assert.False(foundColumnC);
                Assert.Equal(0, cIdx);
            }
        }

        [Fact]
        void TestSelectColumnsWithSameName()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            using (var env = new ConsoleEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                var est = new CopyColumnsEstimator(env, new[] {("A", "A"), ("B", "B")});
                var chain = est.Append(new SelectColumnsEstimator(env, new[]{"A", "C" }));
                var transformer = chain.Fit(dataView);
                var result = transformer.Transform(dataView);

                // Copied columns should equal AABBC, however we chose to keep A and C
                // so the result is AAC
                Assert.Equal(3, result.Schema.ColumnCount);
                var foundColumnA = result.Schema.TryGetColumnIndex("A", out int aIdx);
                var foundColumnB = result.Schema.TryGetColumnIndex("B", out int bIdx);
                var foundColumnC = result.Schema.TryGetColumnIndex("C", out int cIdx);
                Assert.True(foundColumnA);
                Assert.Equal(1, aIdx);
                Assert.False(foundColumnB);
                Assert.Equal(0, bIdx);
                Assert.True(foundColumnC);
                Assert.Equal(2, cIdx);
            }
        }

        [Fact]
        void TestSelectColumnsWithNoKeepHidden()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            using (var env = new ConsoleEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                var est = new CopyColumnsEstimator(env, new[] {("A", "A"), ("B", "B")});
                var chain = est.Append(new SelectColumnsEstimator(env, new[] {"A", "B" }, null, false));
                var transformer = chain.Fit(dataView);
                var result = transformer.Transform(dataView);

                // Copied columns should equal AABBC, however we chose to keep A and B
                // and not keeping the columns hidden, so the result should be AB
                Assert.Equal(2, result.Schema.ColumnCount);
                var foundColumnA = result.Schema.TryGetColumnIndex("A", out int aIdx);
                var foundColumnB = result.Schema.TryGetColumnIndex("B", out int bIdx);
                var foundColumnC = result.Schema.TryGetColumnIndex("C", out int cIdx);
                Assert.True(foundColumnA);
                Assert.Equal(0, aIdx);
                Assert.True(foundColumnB);
                Assert.Equal(1, bIdx);
                Assert.False(foundColumnC);
                Assert.Equal(0, cIdx);
            }
        }

        [Fact]
        void TestSelectWithKeepAndDropSet()
        {
            using (var env = new ConsoleEnvironment())
            {
                var test = new string[]{ "D", "G"};
                Assert.Throws<InvalidOperationException>(() => new SelectColumnsEstimator(env, test, test));
            }
        }

        [Fact]
        void TestSelectNoKeepAndDropSet()
        {
            using (var env = new ConsoleEnvironment())
            {
                Assert.Throws<InvalidOperationException>(() => new SelectColumnsEstimator(env, null, null));
            }
        }

        [Fact]
        void TestSelectSavingAndLoading()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            using (var env = new ConsoleEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                var est = new SelectColumnsEstimator(env, new[] { "A", "B" });
                var transformer = est.Fit(dataView);
                using (var ms = new MemoryStream())
                {
                    transformer.SaveTo(env, ms);
                    ms.Position = 0;
                    var loadedTransformer = TransformerChain.LoadFrom(env, ms);
                    var result = loadedTransformer.Transform(dataView);
                    Assert.Equal(2, result.Schema.ColumnCount);
                    Assert.Equal("A", result.Schema.GetColumnName(0));
                    Assert.Equal("B", result.Schema.GetColumnName(1));
                }
            }
        }

        [Fact]
        void TestSelectSavingAndLoadingWithNoKeepHidden()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            using (var env = new ConsoleEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                var est = new CopyColumnsEstimator(env, new[] {("A", "A"), ("B", "B")}).Append(
                          new SelectColumnsEstimator(env, new[] { "A", "B" }, null, false));
                var transformer = est.Fit(dataView);
                using (var ms = new MemoryStream())
                {
                    transformer.SaveTo(env, ms);
                    ms.Position = 0;
                    var loadedTransformer = TransformerChain.LoadFrom(env, ms);
                    var result = loadedTransformer.Transform(dataView);
                    Assert.Equal(2, result.Schema.ColumnCount);
                    Assert.Equal("A", result.Schema.GetColumnName(0));
                    Assert.Equal("B", result.Schema.GetColumnName(1));
                }
            }
        }

        [Fact]
        void TestSelectBackCompatDropColumns()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            using (var env = new ConsoleEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                string dropModelPath = GetDataPath("backcompat/drop-model.zip");
                using (FileStream fs = File.OpenRead(dropModelPath))
                {
                    var result = ModelFileUtils.LoadTransforms(env, dataView, fs);
                    var foundColumnA = result.Schema.TryGetColumnIndex("A", out int aIdx);
                    var foundColumnB = result.Schema.TryGetColumnIndex("B", out int bIdx);
                    var foundColumnC = result.Schema.TryGetColumnIndex("C", out int cIdx);
                    Assert.False(foundColumnA);
                    Assert.Equal(0, aIdx);
                    Assert.True(foundColumnB);
                    Assert.Equal(0, bIdx);
                    Assert.True(foundColumnC);
                    Assert.Equal(1, cIdx);
                }
            }
        }

        [Fact]
        void TestSelectBackCompatKeepColumns()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            using (var env = new ConsoleEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                string dropModelPath = GetDataPath("backcompat/keep-model.zip");
                using (FileStream fs = File.OpenRead(dropModelPath))
                {
                    var result = ModelFileUtils.LoadTransforms(env, dataView, fs);
                    var foundColumnA = result.Schema.TryGetColumnIndex("A", out int aIdx);
                    var foundColumnB = result.Schema.TryGetColumnIndex("B", out int bIdx);
                    var foundColumnC = result.Schema.TryGetColumnIndex("C", out int cIdx);
                    Assert.True(foundColumnA);
                    Assert.Equal(0, aIdx);
                    Assert.True(foundColumnB);
                    Assert.Equal(1, bIdx);
                    Assert.False(foundColumnC);
                    Assert.Equal(0, cIdx);
                }
            }
        }
        
        [Fact]
        void TestSelectBackCompatChooseColumns()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            using (var env = new ConsoleEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                string dropModelPath = GetDataPath("backcompat/choose-model.zip");
                using (FileStream fs = File.OpenRead(dropModelPath))
                {
                    var result = ModelFileUtils.LoadTransforms(env, dataView, fs);
                    var foundColumnA = result.Schema.TryGetColumnIndex("A", out int aIdx);
                    var foundColumnB = result.Schema.TryGetColumnIndex("B", out int bIdx);
                    var foundColumnC = result.Schema.TryGetColumnIndex("C", out int cIdx);
                    Assert.True(foundColumnA);
                    Assert.Equal(0, aIdx);
                    Assert.True(foundColumnB);
                    Assert.Equal(1, bIdx);
                    Assert.False(foundColumnC);
                    Assert.Equal(0, cIdx);
                }
            }
        }

        [Fact]
        void TestSelectBackCompatChooseColumnsWithKeep()
        {
            // Model generated with: xf=copy{col=A:A col=B:B} xf=choose{col=Label col=Features col=A col=B hidden=keep}
            // Output expected is AABB
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            using (var env = new ConsoleEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                string chooseModelPath = GetDataPath("backcompat/choose-keep-model.zip");
                using (FileStream fs = File.OpenRead(chooseModelPath))
                {
                    var result = ModelFileUtils.LoadTransforms(env, dataView, fs);
                    Assert.Equal(4, result.Schema.ColumnCount);
                    var foundColumnA = result.Schema.TryGetColumnIndex("A", out int aIdx);
                    var foundColumnB = result.Schema.TryGetColumnIndex("B", out int bIdx);
                    var foundColumnC = result.Schema.TryGetColumnIndex("C", out int cIdx);
                    Assert.True(foundColumnA);
                    Assert.Equal(1, aIdx);
                    Assert.True(foundColumnB);
                    Assert.Equal(3, bIdx);
                    Assert.False(foundColumnC);
                    Assert.Equal(0, cIdx);
                }
            }
        }

        [Fact]
        void TestCommandLineWithKeep()
        {
            using (var env = new ConsoleEnvironment())
            {
                Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0 col=B:R4:1 col=C:R4:2} xf=select{keepcol=A keepcol=B} in=f:\1.txt" }), (int)0);
            }
        }

        [Fact]
        void TestCommandLineWithDrop()
        {
            using (var env = new ConsoleEnvironment())
            {
                Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0 col=B:R4:1 col=C:R4:2} xf=select{dropcol=A dropcol=B} in=f:\1.txt" }), (int)0);
            }
        }

        [Fact]
        void TestCommandLineKeepWithoutHidden()
        {
            using (var env = new ConsoleEnvironment())
            {
                Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0 col=B:R4:1 col=C:R4:2} xf=select{keepcol=A keepcol=B hidden=-} in=f:\1.txt" }), (int)0);
            }
        }

        [Fact]
        void TestCommandLineKeepWithIgnoreMismatch()
        {
            using (var env = new ConsoleEnvironment())
            {
                Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0 col=B:R4:1 col=C:R4:2} xf=select{keepcol=A keepcol=B ignore=-} in=f:\1.txt" }), (int)0);
            }
        }
    }
}
