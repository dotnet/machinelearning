// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.RunTests;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Core.Tests.UnitTests
{
    public sealed class RandomSourceTests : BaseTestBaseline
    {
        public RandomSourceTests(ITestOutputHelper output) : base(output) { }
        [Fact]
        [TestCategory("Utilities")]
        public void MersenneTwisterRandomSource_ReproducibleAndRanges()
        {
            var s1 = new MersenneTwisterRandomSource(12345);
            var s2 = new MersenneTwisterRandomSource(12345);

            // Next() in [0, int.MaxValue)
            for (int i = 0; i < 10; i++)
            {
                var a = s1.Next();
                var b = s2.Next();
                Assert.Equal(a, b);
                Assert.InRange(a, 0, int.MaxValue - 1);
            }

            // Next(max)
            Assert.Throws<ArgumentOutOfRangeException>(() => s1.Next(0));
            for (int i = 1; i <= 5; i++)
            {
                var a = s1.Next(i);
                var b = s2.Next(i);
                Assert.Equal(a, b);
                Assert.InRange(a, 0, i - 1);
            }

            // Next(min,max)
            Assert.Throws<ArgumentOutOfRangeException>(() => s1.Next(5, 5));
            for (int i = 0; i < 5; i++)
            {
                var a = s1.Next(-10, 10);
                var b = s2.Next(-10, 10);
                Assert.Equal(a, b);
                Assert.InRange(a, -10, 9);
            }

            // NextDouble and NextSingle in [0,1)
            for (int i = 0; i < 8; i++)
            {
                var da = s1.NextDouble();
                var db = s2.NextDouble();
                Assert.Equal(da, db);
                Assert.InRange(da, 0.0, 1.0 - double.Epsilon);
            }

            for (int i = 0; i < 8; i++)
            {
                var fa = s1.NextSingle();
                var fb = s2.NextSingle();
                Assert.Equal(fa, fb);
                Assert.InRange(fa, 0.0f, 1.0f);
            }

            // Int64 variants
            Assert.Throws<ArgumentOutOfRangeException>(() => s1.NextInt64(0));
            Assert.Throws<ArgumentOutOfRangeException>(() => s1.NextInt64(5, 5));

            for (int i = 0; i < 5; i++)
            {
                var a = s1.NextInt64();
                var b = s2.NextInt64();
                Assert.Equal(a, b);
            }

            for (int i = 0; i < 5; i++)
            {
                var a = s1.NextInt64(1000);
                var b = s2.NextInt64(1000);
                Assert.Equal(a, b);
                Assert.InRange(a, 0, 999);
            }

            for (int i = 0; i < 5; i++)
            {
                var a = s1.NextInt64(-123, 456);
                var b = s2.NextInt64(-123, 456);
                Assert.Equal(a, b);
                Assert.InRange(a, -123, 455);
            }

            // NextBytes and bulk APIs
            var buf1 = new byte[13];
            var buf2 = new byte[13];
            s1.NextBytes(buf1);
            s2.NextBytes(buf2);
            Assert.Equal(buf1, buf2);

            var doubles1 = new double[7];
            var doubles2 = new double[7];
            s1.NextDoubles(doubles1);
            s2.NextDoubles(doubles2);
            Assert.Equal(doubles1, doubles2);
            foreach (var d in doubles1)
                Assert.InRange(d, 0.0, 1.0);

            var u321 = new uint[9];
            var u322 = new uint[9];
            s1.NextUInt32(u321);
            s2.NextUInt32(u322);
            Assert.Equal(u321, u322);
        }

        [Fact]
        [TestCategory("Utilities")]
        public void RandomSourceAdapter_Matches_SystemRandom()
        {
            const int seed = 777;
            var a1 = new RandomSourceAdapter(new Random(seed));
            var a2 = new RandomSourceAdapter(new Random(seed));

            for (int i = 0; i < 5; i++) Assert.Equal(a1.Next(), a2.Next());
            for (int i = 1; i <= 5; i++) Assert.Equal(a1.Next(i), a2.Next(i));
            for (int i = 0; i < 5; i++) Assert.Equal(a1.Next(-50, 50), a2.Next(-50, 50));

            Assert.Equal(a1.NextDouble(), a2.NextDouble());

            var bytesA = new byte[17];
            var bytesB = new byte[17];
            a1.NextBytes(bytesA);
            a2.NextBytes(bytesB);
            Assert.Equal(bytesA, bytesB);

#if NET6_0_OR_GREATER
            a1 = new RandomSourceAdapter(new Random(seed));
            a2 = new RandomSourceAdapter(new Random(seed));
            Assert.Equal(a1.NextSingle(), a2.NextSingle());
            Assert.Equal(a1.NextInt64(), a2.NextInt64());
            Assert.Equal(a1.NextInt64(1000), a2.NextInt64(1000));
            Assert.Equal(a1.NextInt64(-5, 7), a2.NextInt64(-5, 7));

            a1 = new RandomSourceAdapter(new Random(seed));
            a2 = new RandomSourceAdapter(new Random(seed));
            var spanA = new byte[8];
            var spanB = new byte[8];
            a1.NextBytes(spanA);
            a2.NextBytes(spanB);
            Assert.Equal(spanA, spanB);
#endif
        }

        [Fact]
        [TestCategory("Utilities")]
        public void RandomFromRandomSource_Matches_Source()
        {
            // Use MT source to ensure deterministic test across TFMs
            var srcForRandom = new MersenneTwisterRandomSource(9876);
            var srcForCompare = new MersenneTwisterRandomSource(9876);
            var random = new RandomFromRandomSource(srcForRandom);

            for (int i = 0; i < 5; i++)
                Assert.Equal(srcForCompare.Next(), random.Next());

            for (int i = 1; i <= 5; i++)
                Assert.Equal(srcForCompare.Next(i), random.Next(i));

            for (int i = 0; i < 5; i++)
                Assert.Equal(srcForCompare.Next(-10, 10), random.Next(-10, 10));

            var bytesA = new byte[21];
            var bytesB = new byte[21];
            srcForCompare.NextBytes(bytesA);
            random.NextBytes(bytesB);
            Assert.Equal(bytesA, bytesB);

            Assert.Equal(srcForCompare.NextDouble(), random.NextDouble());

#if NET6_0_OR_GREATER
            // Span-based NextBytes and newer APIs
            var sA = new byte[5];
            var sB = new byte[5];
            srcForCompare.NextBytes(sA);
            random.NextBytes(sB);
            Assert.Equal(sA, sB);

            Assert.Equal(srcForCompare.NextSingle(), random.NextSingle());
            Assert.Equal(srcForCompare.NextInt64(), random.NextInt64());
            Assert.Equal(srcForCompare.NextInt64(1000), random.NextInt64(1000));
            Assert.Equal(srcForCompare.NextInt64(-20, 33), random.NextInt64(-20, 33));
#endif
        }

        [Fact]
        [TestCategory("Utilities")]
        public void ResourceManagerUtils_BuildsUrl_And_ErrorMessages()
        {
            // Preserve env and restore on exit
            var oldBase = Environment.GetEnvironmentVariable(ResourceManagerUtils.CustomResourcesUrlEnvVariable);
            try
            {
                Environment.SetEnvironmentVariable(ResourceManagerUtils.CustomResourcesUrlEnvVariable, null);
                var url = ResourceManagerUtils.GetUrl("foo/bar.txt");
                Assert.StartsWith("https://aka.ms/mlnet-resources/", url);
                Assert.EndsWith("foo/bar.txt", url);

                Environment.SetEnvironmentVariable(ResourceManagerUtils.CustomResourcesUrlEnvVariable, "https://example.com/custom/");
                url = ResourceManagerUtils.GetUrl("a/b");
                Assert.Equal("https://example.com/custom/a/b", url);

                // GetErrorMessage formatting
                var r1 = new ResourceManagerUtils.ResourceDownloadResults("file", "some error");
                var r2 = new ResourceManagerUtils.ResourceDownloadResults("file", "other error", "https://host/resource");

                var first = ResourceManagerUtils.GetErrorMessage(out var msg1, r1, r2);
                Assert.Same(r1, first);
                Assert.Contains("Error downloading resource:", msg1);

                first = ResourceManagerUtils.GetErrorMessage(out var msg2, r2, r1);
                Assert.Same(r2, first);
                Assert.Contains("Error downloading resource from", msg2);

                // IsRedirectToDefaultPage returns false for file URIs and invalid absolute
                Assert.False(ResourceManagerUtils.Instance.IsRedirectToDefaultPage("file:///C:/temp/x"));
                Assert.False(ResourceManagerUtils.Instance.IsRedirectToDefaultPage("not a uri"));
            }
            finally
            {
                Environment.SetEnvironmentVariable(ResourceManagerUtils.CustomResourcesUrlEnvVariable, oldBase);
            }
        }

        [Fact]
        [TestCategory("Utilities")]
        public async System.Threading.Tasks.Task ResourceManagerUtils_Throws_For_NonAkaHost()
        {
            // Force downloads into a temp directory to avoid touching AppData
            var oldPath = Environment.GetEnvironmentVariable(Utils.CustomSearchDirEnvVariable);
            var oldBase = Environment.GetEnvironmentVariable(ResourceManagerUtils.CustomResourcesUrlEnvVariable);
            var tmp = Path.Combine(Path.GetTempPath(), "mlnet-test-resources", Guid.NewGuid().ToString("N"));

            try
            {
                Directory.CreateDirectory(tmp);
                Environment.SetEnvironmentVariable(Utils.CustomSearchDirEnvVariable, tmp);
                Environment.SetEnvironmentVariable(ResourceManagerUtils.CustomResourcesUrlEnvVariable, "https://example.com/base/");

                using var swOut = new StringWriter();
                using var swErr = new StringWriter();
                var env = new ConsoleEnvironment(1, outWriter: swOut, errWriter: swErr);
                using var ch = env.Start("test");

                await Assert.ThrowsAsync<NotSupportedException>(() =>
                    ResourceManagerUtils.Instance.EnsureResourceAsync(env, ch, "rel/path", "file.bin", "subdir", timeout: 1000));
            }
            finally
            {
                Environment.SetEnvironmentVariable(Utils.CustomSearchDirEnvVariable, oldPath);
                Environment.SetEnvironmentVariable(ResourceManagerUtils.CustomResourcesUrlEnvVariable, oldBase);
                try { if (Directory.Exists(tmp)) Directory.Delete(tmp, recursive: true); } catch { }
            }
        }
    }
}
