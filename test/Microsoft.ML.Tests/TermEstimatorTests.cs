// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.Tools;
using Microsoft.ML.Transforms;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public class TermEstimatorTests : TestDataPipeBase
    {
        public TermEstimatorTests(ITestOutputHelper output) : base(output)
        {
        }

        class TestClass
        {
            public int A;
            public int B;
            public int C;
        }

        class TestClassXY
        {
            public int X;
            public int Y;
        }

        class TestClassDifferentTypes
        {
            public string A;
            public string B;
            public string C;
        }


        class TestMetaClass
        {
            public int NotUsed;
            public string Term;
        }

        [Fact]
        void TestDifferentTypes()
        {
            string dataPath = GetDataPath("adult.tiny.with-schema.txt");

            var loader = new TextLoader(ML, new TextLoader.Options
            {
                Columns = new[]{
                    new TextLoader.Column("float1", DataKind.Single, 9),
                    new TextLoader.Column("float4", DataKind.Single, new[]{new TextLoader.Range(9), new TextLoader.Range(10), new TextLoader.Range(11), new TextLoader.Range(12) }),
                    new TextLoader.Column("double1", DataKind.Double, 9),
                    new TextLoader.Column("double4", DataKind.Double, new[]{new TextLoader.Range(9), new TextLoader.Range(10), new TextLoader.Range(11), new TextLoader.Range(12) }),
                    new TextLoader.Column("int1", DataKind.Int32, 9),
                    new TextLoader.Column("text1", DataKind.String, 1),
                    new TextLoader.Column("text2", DataKind.String, new[]{new TextLoader.Range(1), new TextLoader.Range(2)}),
                },
                Separator = "\t",
                HasHeader = true
            }, new MultiFileSource(dataPath));

            var pipe = new ValueToKeyMappingEstimator(ML, new[]{
                    new ValueToKeyMappingEstimator.ColumnOptions("TermFloat1", "float1"),
                    new ValueToKeyMappingEstimator.ColumnOptions("TermFloat4", "float4"),
                    new ValueToKeyMappingEstimator.ColumnOptions("TermDouble1", "double1"),
                    new ValueToKeyMappingEstimator.ColumnOptions("TermDouble4", "double4"),
                    new ValueToKeyMappingEstimator.ColumnOptions("TermInt1", "int1"),
                    new ValueToKeyMappingEstimator.ColumnOptions("TermText1", "text1"),
                    new ValueToKeyMappingEstimator.ColumnOptions("TermText2", "text2")
                });
            var data = loader.Load(dataPath);
            data = ML.Data.TakeRows(data, 10);
            var outputPath = GetOutputPath("Term", "Term.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(ML, new TextSaver.Arguments { Silent = true });
                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, pipe.Fit(data).Transform(data), fs, keepHidden: true);
            }

            CheckEquality("Term", "Term.tsv");
            Done();
        }

        [Fact]
        void TestSimpleCase()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };

            var xydata = new[] { new TestClassXY() { X = 10, Y = 100 }, new TestClassXY() { X = -1, Y = -100 } };
            var stringData = new[] { new TestClassDifferentTypes { A = "1", B = "c", C = "b" } };
            var dataView = ML.Data.LoadFromEnumerable(data);
            var pipe = new ValueToKeyMappingEstimator(Env, new[]{
                   new ValueToKeyMappingEstimator.ColumnOptions("TermA", "A"),
                   new ValueToKeyMappingEstimator.ColumnOptions("TermB", "B"),
                   new ValueToKeyMappingEstimator.ColumnOptions("TermC", "C")
                });
            var invalidData = ML.Data.LoadFromEnumerable(xydata);
            var validFitNotValidTransformData = ML.Data.LoadFromEnumerable(stringData);
            TestEstimatorCore(pipe, dataView, null, invalidData, validFitNotValidTransformData);
        }

        [Fact]
        void TestOldSavingAndLoading()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var dataView = ML.Data.LoadFromEnumerable(data);
            var est = new ValueToKeyMappingEstimator(Env, new[]{
                    new ValueToKeyMappingEstimator.ColumnOptions("TermA", "A"),
                    new ValueToKeyMappingEstimator.ColumnOptions("TermB", "B"),
                    new ValueToKeyMappingEstimator.ColumnOptions("TermC", "C")
                });
            var transformer = est.Fit(dataView);
            var result = transformer.Transform(dataView);
            var resultRoles = new RoleMappedData(result);
            using (var ms = new MemoryStream())
            {
                TrainUtils.SaveModel(Env, Env.Start("saving"), ms, null, resultRoles);
                ms.Position = 0;
                var loadedView = ModelFileUtils.LoadTransforms(Env, dataView, ms);
                ValidateTermTransformer(loadedView);
            }
        }

        [Fact]
        void TestMetadataCopy()
        {
            var data = new[] { new TestMetaClass() { Term = "A", NotUsed = 1 }, new TestMetaClass() { Term = "B" }, new TestMetaClass() { Term = "C" } };
            var dataView = ML.Data.LoadFromEnumerable(data);
            var termEst = new ValueToKeyMappingEstimator(Env, new[] {
                    new ValueToKeyMappingEstimator.ColumnOptions("T", "Term") });

            var termTransformer = termEst.Fit(dataView);
            var result = termTransformer.Transform(dataView);
            result.Schema.TryGetColumnIndex("T", out int termIndex);
            var names1 = default(VBuffer<ReadOnlyMemory<char>>);
            var type1 = result.Schema[termIndex].Type;
            var itemType1 = (type1 as VectorType)?.ItemType ?? type1;
            result.Schema[termIndex].GetKeyValues(ref names1);
            Assert.True(names1.GetValues().Length > 0);
        }

        [Fact]
        void TestCommandLine()
        {
            Assert.Equal(0, Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0} xf=Term{col=B:A} in=f:\2.txt" }));
        }

        private void ValidateTermTransformer(IDataView result)
        {
            using (var cursor = result.GetRowCursorForAllColumns())
            {
                uint avalue = 0;
                uint bvalue = 0;
                uint cvalue = 0;

                var aGetter = cursor.GetGetter<uint>(result.Schema["TermA"]);
                var bGetter = cursor.GetGetter<uint>(result.Schema["TermB"]);
                var cGetter = cursor.GetGetter<uint>(result.Schema["TermC"]);
                uint i = 1;
                while (cursor.MoveNext())
                {
                    aGetter(ref avalue);
                    bGetter(ref bvalue);
                    cGetter(ref cvalue);
                    Assert.Equal(i, avalue);
                    Assert.Equal(i, bvalue);
                    Assert.Equal(i, cvalue);
                    i++;
                }
            }
        }
    }
}

