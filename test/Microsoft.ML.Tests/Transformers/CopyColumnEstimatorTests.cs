// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Microsoft.ML.Runtime;
using Microsoft.ML.Tools;
using Microsoft.ML.Transforms;
using Xunit;

namespace Microsoft.ML.Tests
{
    public class CopyColumnEstimatorTests
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

        class TestMetaClass
        {
            public int NotUsed;
            public string Term;
        }

        [Fact]
        void TestWorking()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var env = new MLContext();
            var dataView = env.Data.LoadFromEnumerable(data);
            var est = new ColumnCopyingEstimator(env, new[] { ("D", "A"), ("E", "B"), ("F", "A") });
            var transformer = est.Fit(dataView);
            var result = transformer.Transform(dataView);
            ValidateCopyColumnTransformer(result);
        }

        [Fact]
        void TestBadOriginalSchema()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var env = new MLContext();
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
        void TestBadTransformSchema()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var xydata = new[] { new TestClassXY() { X = 10, Y = 100 }, new TestClassXY() { X = -1, Y = -100 } };
            var env = new MLContext();
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
        void TestSavingAndLoading()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var env = new MLContext();
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
        void TestOldSavingAndLoading()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var env = new MLContext();
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
        void TestMetadataCopy()
        {
            var data = new[] { new TestMetaClass() { Term = "A", NotUsed = 1 }, new TestMetaClass() { Term = "B" }, new TestMetaClass() { Term = "C" } };
            var env = new MLContext();
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
            var itemType1 = (type1 as VectorType)?.ItemType ?? type1;
            var key = itemType1 as KeyType;
            Assert.NotNull(key);
            Assert.InRange<ulong>(key.Count, 0, int.MaxValue);
            int size = (int)key.Count;
            var type2 = result.Schema[copyIndex].Type;
            result.Schema[termIndex].GetKeyValues(ref names1);
            result.Schema[copyIndex].GetKeyValues(ref names2);
            Assert.True(CompareVec(in names1, in names2, size, (a, b) => a.Span.SequenceEqual(b.Span)));
        }

        [Fact]
        void TestCommandLine()
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
        private bool CompareVec<T>(in VBuffer<T> v1, in VBuffer<T> v2, int size, Func<T, T, bool> fn)
        {
            return CompareVec(in v1, in v2, size, (i, x, y) => fn(x, y));
        }

        private bool CompareVec<T>(in VBuffer<T> v1, in VBuffer<T> v2, int size, Func<int, T, T, bool> fn)
        {
            Contracts.Assert(size == 0 || v1.Length == size);
            Contracts.Assert(size == 0 || v2.Length == size);
            Contracts.Assert(v1.Length == v2.Length);

            var v1Values = v1.GetValues();
            var v2Values = v2.GetValues();

            if (v1.IsDense && v2.IsDense)
            {
                for (int i = 0; i < v1.Length; i++)
                {
                    var x1 = v1Values[i];
                    var x2 = v2Values[i];
                    if (!fn(i, x1, x2))
                        return false;
                }
                return true;
            }

            var v1Indices = v1.GetIndices();
            var v2Indices = v2.GetIndices();

            Contracts.Assert(!v1.IsDense || !v2.IsDense);
            int iiv1 = 0;
            int iiv2 = 0;
            for (; ; )
            {
                int iv1 = v1.IsDense ? iiv1 : iiv1 < v1Indices.Length ? v1Indices[iiv1] : v1.Length;
                int iv2 = v2.IsDense ? iiv2 : iiv2 < v2Indices.Length ? v2Indices[iiv2] : v2.Length;
                T x1, x2;
                int iv;
                if (iv1 == iv2)
                {
                    if (iv1 == v1.Length)
                        return true;
                    x1 = v1Values[iiv1];
                    x2 = v2Values[iiv2];
                    iv = iv1;
                    iiv1++;
                    iiv2++;
                }
                else if (iv1 < iv2)
                {
                    x1 = v1Values[iiv1];
                    x2 = default(T);
                    iv = iv1;
                    iiv1++;
                }
                else
                {
                    x1 = default(T);
                    x2 = v2Values[iiv2];
                    iv = iv2;
                    iiv2++;
                }
                if (!fn(iv, x1, x2))
                    return false;
            }
        }
    }
}

