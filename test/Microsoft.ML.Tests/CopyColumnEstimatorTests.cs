using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using System.Collections.Generic;
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

        [Fact]
        void TestOne()
        {
            var data = new List<TestClass>() { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            using (var env = new TlcEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                var est = new CopyColumnsEstimator(env, new[] { ("A", "D"), ("B", "E") });
                var transformer = est.Fit(dataView);
                var result = transformer.Transform(dataView);
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
        }
    }
}
