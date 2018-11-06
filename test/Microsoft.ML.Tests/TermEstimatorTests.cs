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
using Microsoft.ML.Transforms.Categorical;
using System;
using System.IO;
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
            string dataPath = GetDataPath("adult.test");

            var loader = new TextLoader(Env, new TextLoader.Arguments
            {
                Column = new[]{
                    new TextLoader.Column("float1", DataKind.R4, 0),
                    new TextLoader.Column("float4", DataKind.R4, new[]{new TextLoader.Range(0), new TextLoader.Range(2), new TextLoader.Range(4), new TextLoader.Range(10) }),
                    new TextLoader.Column("double1", DataKind.R8, 0),
                    new TextLoader.Column("double4", DataKind.R8, new[]{new TextLoader.Range(0), new TextLoader.Range(2), new TextLoader.Range(4), new TextLoader.Range(10) }),
                    new TextLoader.Column("int1", DataKind.I4, 0),
                    new TextLoader.Column("text1", DataKind.TX, 1),
                    new TextLoader.Column("text2", DataKind.TX, new[]{new TextLoader.Range(1), new TextLoader.Range(3)}),
                },
                Separator = ",",
                HasHeader = true
            }, new MultiFileSource(dataPath));

            var pipe = new ValueToKeyMappingEstimator(Env, new[]{
                    new TermTransform.ColumnInfo("float1", "TermFloat1"),
                    new TermTransform.ColumnInfo("float4", "TermFloat4"),
                    new TermTransform.ColumnInfo("double1", "TermDouble1"),
                    new TermTransform.ColumnInfo("double4", "TermDouble4"),
                    new TermTransform.ColumnInfo("int1", "TermInt1"),
                    new TermTransform.ColumnInfo("text1", "TermText1"),
                    new TermTransform.ColumnInfo("text2", "TermText2")
                });
            var data = loader.Read(dataPath);
            data = TakeFilter.Create(Env, data, 10);
            var outputPath = GetOutputPath("Term", "Term.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true });
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
            var dataView = ComponentCreation.CreateDataView(Env, data);
            var pipe = new ValueToKeyMappingEstimator(Env, new[]{
                    new TermTransform.ColumnInfo("A", "TermA"),
                    new TermTransform.ColumnInfo("B", "TermB"),
                    new TermTransform.ColumnInfo("C", "TermC")
                });
            var invalidData = ComponentCreation.CreateDataView(Env, xydata);
            var validFitNotValidTransformData = ComponentCreation.CreateDataView(Env, stringData);
            TestEstimatorCore(pipe, dataView, null, invalidData, validFitNotValidTransformData);
        }

        [Fact]
        void TestOldSavingAndLoading()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var dataView = ComponentCreation.CreateDataView(Env, data);
            var est = new ValueToKeyMappingEstimator(Env, new[]{
                    new TermTransform.ColumnInfo("A", "TermA"),
                    new TermTransform.ColumnInfo("B", "TermB"),
                    new TermTransform.ColumnInfo("C", "TermC")
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
            var dataView = ComponentCreation.CreateDataView(Env, data);
            var termEst = new ValueToKeyMappingEstimator(Env, new[] {
                    new TermTransform.ColumnInfo("Term" ,"T") });
                    
            var termTransformer = termEst.Fit(dataView);
            var result = termTransformer.Transform(dataView);
            result.Schema.TryGetColumnIndex("T", out int termIndex);
            var names1 = default(VBuffer<ReadOnlyMemory<char>>);
            var type1 = result.Schema.GetColumnType(termIndex);
            var itemType1 = (type1 as VectorType)?.ItemType ?? type1;
            int size = itemType1 is KeyType keyType ? keyType.Count : -1;
            result.Schema.GetMetadata(MetadataUtils.Kinds.KeyValues, termIndex, ref names1);
            Assert.True(names1.Count > 0);
        }

        [Fact]
        void TestCommandLine()
        {
            using (var env = new ConsoleEnvironment())
            {
                Assert.Equal(0, Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0} xf=Term{col=B:A} in=f:\2.txt" }));
            }
        }

        private void ValidateTermTransformer(IDataView result)
        {
            result.Schema.TryGetColumnIndex("TermA", out int ColA);
            result.Schema.TryGetColumnIndex("TermB", out int ColB);
            result.Schema.TryGetColumnIndex("TermC", out int ColC);
            using (var cursor = result.GetRowCursor(x => true))
            {
                uint avalue = 0;
                uint bvalue = 0;
                uint cvalue = 0;

                var aGetter = cursor.GetGetter<uint>(ColA);
                var bGetter = cursor.GetGetter<uint>(ColB);
                var cGetter = cursor.GetGetter<uint>(ColC);
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

