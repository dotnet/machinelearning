// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.RunTests;
using System.Linq;
using System.Threading;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public class CachingTests : TestDataPipeBase
    {
        public CachingTests(ITestOutputHelper helper) : base(helper)
        {
        }

        private class MyData
        {
            [NoColumn]
            public int AccessCount;
            private float[] _features;

            [VectorType(3)]
            public float[] Features
            {
                get { Interlocked.Increment(ref AccessCount); return _features; }
                set { _features = value; }
            }

            public MyData()
            {
                Features = new float[] { 1, 2, 3 };
            }
        }

        [Fact]
        public void CacheCheckpointTest()
        {
            var trainData = Enumerable.Range(0, 100).Select(c => new MyData()).ToArray();

            var pipe = ML.Transforms.CopyColumns("Features", "F1")
                .Append(ML.Transforms.Normalize("F1", "Norm1"))
                .Append(ML.Transforms.Normalize("F1", "Norm2", Transforms.Normalizers.NormalizingEstimator.NormalizerMode.MeanVariance));

            pipe.Fit(ML.CreateDataView(trainData));

            Assert.True(trainData.All(x => x.AccessCount == 2));

            trainData = Enumerable.Range(0, 100).Select(c => new MyData()).ToArray();
            pipe = ML.Transforms.CopyColumns("Features", "F1")
                .AppendCacheCheckpoint(ML)
                .Append(ML.Transforms.Normalize("F1", "Norm1"))
                .Append(ML.Transforms.Normalize("F1", "Norm2", Transforms.Normalizers.NormalizingEstimator.NormalizerMode.MeanVariance));

            pipe.Fit(ML.CreateDataView(trainData));

            Assert.True(trainData.All(x => x.AccessCount == 1));
        }

        [Fact]
        public void CacheTest()
        {
            var src = Enumerable.Range(0, 100).Select(c => new MyData()).ToArray();
            var data = ML.CreateDataView(src);
            data.GetColumn<float[]>(ML, "Features").ToArray();
            data.GetColumn<float[]>(ML, "Features").ToArray();
            Assert.True(src.All(x => x.AccessCount == 2));

            src = Enumerable.Range(0, 100).Select(c => new MyData()).ToArray();
            data = ML.CreateDataView(src);
            data = ML.Data.Cache(data);
            data.GetColumn<float[]>(ML, "Features").ToArray();
            data.GetColumn<float[]>(ML, "Features").ToArray();
            Assert.True(src.All(x => x.AccessCount == 1));
        }
    }
}
