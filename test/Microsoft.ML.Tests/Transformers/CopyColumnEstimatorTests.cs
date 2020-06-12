// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Microsoft.ML.Runtime;
using Microsoft.ML.TestFramework;
using Microsoft.ML.TestFrameworkCommon;
using Microsoft.ML.Tools;
using Microsoft.ML.Transforms;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public class CopyColumnEstimatorTests : BaseTestClass
    {
        public CopyColumnEstimatorTests(ITestOutputHelper output) : base(output)
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

        class TestMetaClass
        {
            public int NotUsed;
            public string Term;
        }

        [Fact]
        public void TestWorking()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var env = new MLContext(1);
            var dataView = env.Data.LoadFromEnumerable(data);
            var est = new ColumnCopyingEstimator(env, new[] { ("D", "A"), ("E", "B"), ("F", "A") });
            var transformer = est.Fit(dataView);
            var result = transformer.Transform(dataView);
            ValidateCopyColumnTransformer(result);
        }

        [Fact]
        public void TestBadOriginalSchema()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var env = new MLContext(1);
            var dataView = env.Data.LoadFromEnumerable(data);
            var est = new ColumnCopyingEstimator(env, new[] { ("A", "D"), ("E", "B") });
            try
            {
                var transformer = est.Fit(dataView);
                Assert.False(true);
            }
            catch
            {
            }
        }

        [Fact]
        public void TestBadTransformSchema()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var xydata = new[] { new TestClassXY() { X = 10, Y = 100 }, new TestClassXY() { X = -1, Y = -100 } };
            var env = new MLContext(1);
            var dataView = env.Data.LoadFromEnumerable(data);
            var xyDataView = env.Data.LoadFromEnumerable(xydata);
            var est = new ColumnCopyingEstimator(env, new[] { ("D", "A"), ("E", "B"), ("F", "A") });
            var transformer = est.Fit(dataView);
            try
            {
                var result = transformer.Transform(xyDataView);
                Assert.False(true);
            }
            catch
            {
            }
        }

        [Fact]
        public void TestSavingAndLoading()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var env = new MLContext(1);
            var dataView = env.Data.LoadFromEnumerable(data);
            var est = new ColumnCopyingEstimator(env, new[] { ("D", "A"), ("E", "B"), ("F", "A") });
            var transformer = est.Fit(dataView);
            using (var ms = new MemoryStream())
            {
                env.Model.Save(transformer, null, ms);
                ms.Position = 0;
                var loadedTransformer = env.Model.Load(ms, out var schema);
                var result = loadedTransformer.Transform(dataView);
                ValidateCopyColumnTransformer(result);
            }
        }

        [Fact]
        public void TestOldSavingAndLoading()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var env = new MLContext(1);
            var dataView = env.Data.LoadFromEnumerable(data);
            var est = new ColumnCopyingEstimator(env, new[] { ("D", "A"), ("E", "B"), ("F", "A") });
            var transformer = est.Fit(dataView);
            var result = transformer.Transform(dataView);
            var resultRoles = new RoleMappedData(result);
            using (var ms = new MemoryStream())
            {
                TrainUtils.SaveModel(env, ((IHostEnvironment)env).Start("saving"), ms, null, resultRoles);
                ms.Position = 0;
                var loadedView = ModelFileUtils.LoadTransforms(env, dataView, ms);
                ValidateCopyColumnTransformer(loadedView);
            }
        }

        [Fact]
        public void TestMetadataCopy()
        {
            var data = new[] { new TestMetaClass() { Term = "A", NotUsed = 1 }, new TestMetaClass() { Term = "B" }, new TestMetaClass() { Term = "C" } };
            var env = new MLContext(1);
            var dataView = env.Data.LoadFromEnumerable(data);
            var term = ValueToKeyMappingTransformer.Create(env, new ValueToKeyMappingTransformer.Options()
            {
                Columns = new[] { new ValueToKeyMappingTransformer.Column() { Source = "Term", Name = "T" } }
            }, dataView);
            var est = new ColumnCopyingEstimator(env, "T1", "T");
            var transformer = est.Fit(term);
            var result = transformer.Transform(term);
            result.Schema.TryGetColumnIndex("T", out int termIndex);
            result.Schema.TryGetColumnIndex("T1", out int copyIndex);
            var names1 = default(VBuffer<ReadOnlyMemory<char>>);
            var names2 = default(VBuffer<ReadOnlyMemory<char>>);
            var type1 = result.Schema[termIndex].Type;
            var itemType1 = (type1 as VectorDataViewType)?.ItemType ?? type1;
            var key = itemType1 as KeyDataViewType;
            Assert.NotNull(key);
            Assert.InRange<ulong>(key.Count, 0, int.MaxValue);
            int size = (int)key.Count;
            var type2 = result.Schema[copyIndex].Type;
            result.Schema[termIndex].GetKeyValues(ref names1);
            result.Schema[copyIndex].GetKeyValues(ref names2);
            Assert.True(TestCommon.CompareVec(in names1, in names2, size, (a, b) => a.Span.SequenceEqual(b.Span)));
        }

        [Fact]
        public void TestCommandLine()
        {
            Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0} xf=copy{col=B:A} in=f:\1.txt" }), (int)0);
        }

        private void ValidateCopyColumnTransformer(IDataView result)
        {
            using (var cursor = result.GetRowCursorForAllColumns())
            {
                int avalue = 0;
                int bvalue = 0;
                int dvalue = 0;
                int evalue = 0;
                int fvalue = 0;
                var aGetter = cursor.GetGetter<int>(cursor.Schema[0]);
                var bGetter = cursor.GetGetter<int>(cursor.Schema[1]);
                var dGetter = cursor.GetGetter<int>(cursor.Schema[3]);
                var eGetter = cursor.GetGetter<int>(cursor.Schema[4]);
                var fGetter = cursor.GetGetter<int>(cursor.Schema[5]);
                while (cursor.MoveNext())
                {
                    aGetter(ref avalue);
                    bGetter(ref bvalue);
                    dGetter(ref dvalue);
                    eGetter(ref evalue);
                    fGetter(ref fvalue);
                    Assert.Equal(avalue, dvalue);
                    Assert.Equal(bvalue, evalue);
                    Assert.Equal(avalue, fvalue);
                }
            }
        }
    }
}

