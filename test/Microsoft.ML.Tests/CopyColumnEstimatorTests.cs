using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
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
    }
}
