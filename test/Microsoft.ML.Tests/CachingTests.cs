// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
using System.Threading;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.StaticPipe;
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

            var pipe = ML.Transforms.CopyColumns("F1", "Features")
                .Append(ML.Transforms.Normalize("Norm1", "F1"))
                .Append(ML.Transforms.Normalize("Norm2", "F1", Transforms.NormalizingEstimator.NormalizationMode.MeanVariance));

            pipe.Fit(ML.Data.LoadFromEnumerable(trainData));

            Assert.True(trainData.All(x => x.AccessCount == 2));

            trainData = Enumerable.Range(0, 100).Select(c => new MyData()).ToArray();
            pipe = ML.Transforms.CopyColumns("F1", "Features")
                .AppendCacheCheckpoint(ML)
                .Append(ML.Transforms.Normalize("Norm1", "F1"))
                .Append(ML.Transforms.Normalize("Norm2", "F1", Transforms.NormalizingEstimator.NormalizationMode.MeanVariance));

            pipe.Fit(ML.Data.LoadFromEnumerable(trainData));

            Assert.True(trainData.All(x => x.AccessCount == 1));
        }

        [Fact]
        public void CacheTest()
        {
            var src = Enumerable.Range(0, 100).Select(c => new MyData()).ToArray();
            var data = ML.Data.LoadFromEnumerable(src);
            data.GetColumn<float[]>(data.Schema["Features"]).ToArray();
            data.GetColumn<float[]>(data.Schema["Features"]).ToArray();
            Assert.True(src.All(x => x.AccessCount == 2));

            src = Enumerable.Range(0, 100).Select(c => new MyData()).ToArray();
            data = ML.Data.LoadFromEnumerable(src);
            data = ML.Data.Cache(data);
            data.GetColumn<float[]>(data.Schema["Features"]).ToArray();
            data.GetColumn<float[]>(data.Schema["Features"]).ToArray();
            Assert.True(src.All(x => x.AccessCount == 1));
        }

        [Fact]
        public void StaticDataCacheTest()
        {
            var env = new MLContext(seed: 0);
            var dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var dataSource = new MultiFileSource(dataPath);

            var reader = TextLoaderStatic.CreateLoader(env,
                c => (label: c.LoadBool(0), features: c.LoadFloat(1, 9)));

            var data = reader.Load(dataSource);

            var cachedData = data.Cache();

            // Before caching, we are not able to shuffle the data.
            Assert.True(data.AsDynamic.CanShuffle == false);
            // After caching, we are able to shuffle the data!
            Assert.True(cachedData.AsDynamic.CanShuffle == true);
        }
    }
}
