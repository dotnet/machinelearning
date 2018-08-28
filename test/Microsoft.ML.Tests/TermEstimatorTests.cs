// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Tools;
using System.IO;
using Xunit;

namespace Microsoft.ML.Tests
{
    public class TermEstimatorTests
    {
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
        void TestWorking()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            using (var env = new TlcEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                var est = new TermEstimator(env, columns: new[] {
                    new TermTransform.Column { Name = "TermA", Source = "A" },
                    new TermTransform.Column { Name = "TermB", Source = "B" },
                    new TermTransform.Column { Name = "TermC", Source = "C" }
                });
                var transformer = est.Fit(dataView);
                var result = transformer.Transform(dataView);
                ValidateTermTransformer(result);
            }
        }

        [Fact]
        void TestBadOriginalSchema()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            using (var env = new TlcEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                var est = new TermEstimator(env, columns: new[] {
                    new TermTransform.Column { Name = "TermA", Source = "A" },
                    new TermTransform.Column { Name = "TermB", Source = "D" },
                    new TermTransform.Column { Name = "TermC", Source = "B" }
                });
                try
                {
                    var transformer = est.Fit(dataView);
                    Assert.False(true);
                }
                catch
                {
                }
            }
        }

        [Fact]
        void TestBadTransformSchmea()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var xydata = new[] { new TestClassXY() { X = 10, Y = 100 }, new TestClassXY() { X = -1, Y = -100 } };
            var stringData = new[] { new TestClassDifferentTypes { A = "1", B = "c", C = "b" } };
            using (var env = new TlcEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                var xyDataView = ComponentCreation.CreateDataView(env, xydata);
                var est = new TermEstimator(env, columns: new[] {
                    new TermTransform.Column { Name = "TermA", Source = "A" },
                    new TermTransform.Column { Name = "TermB", Source = "B" },
                    new TermTransform.Column { Name = "TermC", Source = "C" }
                });
                var transformer = est.Fit(dataView);
                try
                {
                    var result = transformer.Transform(xyDataView);
                    Assert.False(true);
                }
                catch
                {
                }
                var stringView = ComponentCreation.CreateDataView(env, stringData);
                try
                {
                    var result = transformer.Transform(stringView);
                    Assert.False(true);
                }
                catch
                {
                }
            }
        }

        [Fact]
        void TestSavingAndLoading()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            using (var env = new TlcEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                var est = new TermEstimator(env, columns: new[] {
                    new TermTransform.Column { Name = "TermA", Source = "A" },
                    new TermTransform.Column { Name = "TermB", Source = "B" },
                    new TermTransform.Column { Name = "TermC", Source = "C" }
                });
                var transformer = est.Fit(dataView);
                using (var ms = new MemoryStream())
                {
                    transformer.SaveTo(env, ms);
                    ms.Position = 0;
                    var loadedTransformer = TransformerChain.LoadFrom(env, ms);
                    var result = loadedTransformer.Transform(dataView);
                    ValidateTermTransformer(result);
                }
            }
        }

        [Fact]
        void TestOldSavingAndLoading()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            using (var env = new TlcEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                var est = new TermEstimator(env, columns: new[] {
                    new TermTransform.Column { Name = "TermA", Source = "A" },
                    new TermTransform.Column { Name = "TermB", Source = "B" },
                    new TermTransform.Column { Name = "TermC", Source = "C" }
                });
                var transformer = est.Fit(dataView);
                var result = transformer.Transform(dataView);
                var resultRoles = new RoleMappedData(result);
                using (var ms = new MemoryStream())
                {
                    TrainUtils.SaveModel(env, env.Start("saving"), ms, null, resultRoles);
                    ms.Position = 0;
                    var loadedView = ModelFileUtils.LoadTransforms(env, dataView, ms);
                    ValidateTermTransformer(loadedView);
                }
            }
        }

        [Fact]
        void TestMetadataCopy()
        {
            var data = new[] { new TestMetaClass() { Term = "A", NotUsed = 1 }, new TestMetaClass() { Term = "B" }, new TestMetaClass() { Term = "C" } };
            using (var env = new TlcEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                var termEst = new TermEstimator(env, columns: new[] {
                    new TermTransform.Column { Name = "T", Source = "Term" } });
                var termTransformer = termEst.Fit(dataView);
                var result = termTransformer.Transform(dataView);

                result.Schema.TryGetColumnIndex("T", out int termIndex);
                var names1 = default(VBuffer<DvText>);
                var type1 = result.Schema.GetColumnType(termIndex);
                int size = type1.ItemType.IsKey ? type1.ItemType.KeyCount : -1;
                result.Schema.GetMetadata(MetadataUtils.Kinds.KeyValues, termIndex, ref names1);
                Assert.True(names1.Count > 0);
            }
        }

        [Fact]
        void TestCommandLine()
        {
            using (var env = new TlcEnvironment())
            {
                Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0} xf=Term{col=B:A} in=f:\2.txt" }), (int)0);
            }
        }

        [Fact]
        void OldSaveLoad()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            using (var env = new TlcEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);

                using (var ms = File.OpenRead("term.zip"))
                {
                    var loadedView = ModelFileUtils.LoadTransforms(env, dataView, ms);
                    ValidateTermTransformer(loadedView);
                }
            }
        }

        private void ValidateTermTransformer(IDataView result)
        {
            using (var cursor = result.GetRowCursor(x => true))
            {
                uint avalue = 0;
                uint bvalue = 0;
                uint cvalue = 0;
                var aGetter = cursor.GetGetter<uint>(3);
                var bGetter = cursor.GetGetter<uint>(4);
                var cGetter = cursor.GetGetter<uint>(5);
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

