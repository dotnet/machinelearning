// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.RunTests;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public class HashTests : TestDataPipeBase
    {
        public HashTests(ITestOutputHelper output) : base(output)
        {
        }

        private class TestClass
        {
            public float A;
            public float B;
            public float C;
        }

       /* private class TestMeta
        {
            [VectorType(2)]
            public string[] A;
            public string B;
            [VectorType(2)]
            public int[] C;
            public int D;
            [VectorType(2)]
            public float[] E;
            public float F;
            [VectorType(2)]
            public string[] G;
            public string H;
        }*/

        [Fact]
        public void HashWorkout()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };

            var dataView = ComponentCreation.CreateDataView(Env, data);
            var pipe = new HashEstimator(Env, new[]{
                    new HashTransform.ColumnInfo("A", "CatA", hashBits:4, invertHash:-1),
                    new HashTransform.ColumnInfo("B", "CatB", hashBits:3, ordered:true),
                    new HashTransform.ColumnInfo("C", "CatC", seed:42),
                    new HashTransform.ColumnInfo("A", "CatD"),
                });

            TestEstimatorCore(pipe, dataView);
            Done();
        }

    }
}
