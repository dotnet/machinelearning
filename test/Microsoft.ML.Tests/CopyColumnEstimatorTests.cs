using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.RunTests;
using System;
using System.Collections.Generic;
using System.IO;
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
            public string Term;
        }

        [Fact]
        void TestWorking()
        {
            var data = new List<TestClass>() { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            using (var env = new TlcEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                var est = new CopyColumnsEstimator(env, new[] { ("A", "D"), ("B", "E") });
                var transformer = est.Fit(dataView);
                var result = transformer.Transform(dataView);
                ValidateCopyColumnTransformer(result);
            }
        }

        private void ValidateCopyColumnTransformer(IDataView result)
        {
            using (var cursor = result.GetRowCursor(x => true))
            {
                DvInt4 avalue = 0;
                DvInt4 bvalue = 0;
                DvInt4 dvalue = 0;
                DvInt4 evalue = 0;
                var aGetter = cursor.GetGetter<DvInt4>(0);
                var bGetter = cursor.GetGetter<DvInt4>(1);
                var dGetter = cursor.GetGetter<DvInt4>(3);
                var eGetter = cursor.GetGetter<DvInt4>(4);
                while (cursor.MoveNext())
                {
                    aGetter(ref avalue);
                    bGetter(ref bvalue);
                    dGetter(ref dvalue);
                    eGetter(ref evalue);
                    Assert.Equal(avalue, dvalue);
                    Assert.Equal(bvalue, evalue);
                }
            }
        }

        [Fact]
        void TestBadOriginalSchema()
        {
            var data = new List<TestClass>() { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            using (var env = new TlcEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                var est = new CopyColumnsEstimator(env, new[] { ("D", "A"), ("B", "E") });
                bool failed = false;
                try
                {
                    var transformer = est.Fit(dataView);
                }
                catch
                {
                    failed = true;
                }
                Assert.True(failed);
            }
        }

        [Fact]
        void TestBadTransformSchmea()
        {
            var data = new List<TestClass>() { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var xydata = new List<TestClassXY>() { new TestClassXY() { X = 10, Y = 100 }, new TestClassXY() { X = -1, Y = -100 } };
            using (var env = new TlcEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                var xyDataView = ComponentCreation.CreateDataView(env, xydata);
                var est = new CopyColumnsEstimator(env, new[] { ("A", "D"), ("B", "E") });
                var transformer = est.Fit(dataView);
                bool failed = false;
                try
                {
                    var result = transformer.Transform(xyDataView);
                }
                catch
                {
                    failed = true;
                }
                Assert.True(failed);
            }
        }

        [Fact]
        void TestSavingAndLoading()
        {
            var data = new List<TestClass>() { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            using (var env = new TlcEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                var est = new CopyColumnsEstimator(env, new[] { ("A", "D"), ("B", "E") });
                var transformer = est.Fit(dataView);
                using (var ms = new MemoryStream())
                {
                    transformer.SaveTo(env, ms);
                    ms.Position = 0;
                    var loadedTransformer = TransformerChain.LoadFrom(env, ms);
                    var result = loadedTransformer.Transform(dataView);
                    ValidateCopyColumnTransformer(result);
                }

            }
        }

        [Fact]
        void TestOldSavingAndLoading()
        {
            var data = new List<TestClass>() { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            using (var env = new TlcEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                var est = new CopyColumnsEstimator(env, new[] { ("A", "D"), ("B", "E") });
                var transformer = est.Fit(dataView);
                var result = transformer.Transform(dataView);
                var resultRoles = new RoleMappedData(result);
                using (var ms = new MemoryStream())
                {
                    TrainUtils.SaveModel(env, env.Start("saving"), ms, null, resultRoles);
                    ms.Position = 0;
                    var loadedView = ModelFileUtils.LoadTransforms(env, dataView, ms);
                    ValidateCopyColumnTransformer(loadedView);
                }
            }
        }

        [Fact]
        void TestMetadataCopy()
        {
            var data = new List<TestMetaClass>() { new TestMetaClass() { Term = "A" }, new TestMetaClass() { Term = "B" }, new TestMetaClass() { Term = "C" } };
            using (var env = new TlcEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                var term = new TermTransform(env, new TermTransform.Arguments()
                {
                    Column = new[] { new TermTransform.Column() { Source = "Term", Name = "T" } }
                }, dataView);
                var est = new CopyColumnsEstimator(env, "T", "T1");
                var transformer = est.Fit(term);
                var result = transformer.Transform(term);
                result.Schema.TryGetColumnIndex("T", out int termIndex);
                var termTypes = result.Schema.GetMetadataTypes(termIndex);

                result.Schema.TryGetColumnIndex("T1", out int copyIndex);
                var copyTypes = result.Schema.GetMetadataTypes(copyIndex);


                var names1 = default(VBuffer<DvText>);
                var names2 = default(VBuffer<DvText>);

                var t1 = result.Schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.KeyValues, termIndex);
                var t2 = result.Schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.KeyValues, copyIndex);
                var type1 = result.Schema.GetColumnType(termIndex);
                int size = type1.ItemType.IsKey ? type1.ItemType.KeyCount : -1;
                var type2 = result.Schema.GetColumnType(copyIndex);
                result.Schema.GetMetadata(MetadataUtils.Kinds.KeyValues, termIndex, ref names1);
                result.Schema.GetMetadata(MetadataUtils.Kinds.KeyValues, copyIndex, ref names2);
                Assert.True(CompareVec(ref names1, ref names2, size, DvText.Identical));
            }
        }
        protected bool CompareVec<T>(ref VBuffer<T> v1, ref VBuffer<T> v2, int size, Func<T, T, bool> fn)
        {
            return CompareVec(ref v1, ref v2, size, (i, x, y) => fn(x, y));
        }
        protected bool CompareVec<T>(ref VBuffer<T> v1, ref VBuffer<T> v2, int size, Func<int, T, T, bool> fn)
        {
            Contracts.Assert(size == 0 || v1.Length == size);
            Contracts.Assert(size == 0 || v2.Length == size);
            Contracts.Assert(v1.Length == v2.Length);

            if (v1.IsDense && v2.IsDense)
            {
                for (int i = 0; i < v1.Length; i++)
                {
                    var x1 = v1.Values[i];
                    var x2 = v2.Values[i];
                    if (!fn(i, x1, x2))
                        return false;
                }
                return true;
            }

            Contracts.Assert(!v1.IsDense || !v2.IsDense);
            int iiv1 = 0;
            int iiv2 = 0;
            for (; ; )
            {
                int iv1 = v1.IsDense ? iiv1 : iiv1 < v1.Count ? v1.Indices[iiv1] : v1.Length;
                int iv2 = v2.IsDense ? iiv2 : iiv2 < v2.Count ? v2.Indices[iiv2] : v2.Length;
                T x1, x2;
                int iv;
                if (iv1 == iv2)
                {
                    if (iv1 == v1.Length)
                        return true;
                    x1 = v1.Values[iiv1];
                    x2 = v2.Values[iiv2];
                    iv = iv1;
                    iiv1++;
                    iiv2++;
                }
                else if (iv1 < iv2)
                {
                    x1 = v1.Values[iiv1];
                    x2 = default(T);
                    iv = iv1;
                    iiv1++;
                }
                else
                {
                    x1 = default(T);
                    x2 = v2.Values[iiv2];
                    iv = iv2;
                    iiv2++;
                }
                if (!fn(iv, x1, x2))
                    return false;
            }
        }
    }
}
