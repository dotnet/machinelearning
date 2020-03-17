// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.RunTests;
using Microsoft.ML.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Core.Tests.UnitTests
{
    public class TestUtilities : BaseTestBaseline
    {
        public TestUtilities(ITestOutputHelper helper)
            : base(helper)
        {
        }

        [Fact]
        [TestCategory("Utilities")]
        public void CheckIsMonotonicallyIncreasingInt()
        {
            // A sorted (increasing) array
            int[] x = Enumerable.Range(0, 10).ToArray();
            Assert.True(Utils.IsMonotonicallyIncreasing(x));

            // A monotonically increasing array
            var x1Temp = x[1];
            var x7Temp = x[7];
            x[1] = x[0];
            x[7] = x[6];
            Assert.True(Utils.IsMonotonicallyIncreasing(x));
            x[1] = x1Temp;
            x[7] = x7Temp;

            // Not sorted
            x[1] = x[6];
            Assert.False(Utils.IsMonotonicallyIncreasing(x));
            x[1] = x1Temp;

            // Null lists are considered to be sorted
            int[] nullX = null;
            Assert.True(Utils.IsMonotonicallyIncreasing(nullX));
        }
        
        [Fact]
        [TestCategory("Utilities")]
        public void CheckIsMonotonicallyIncreasingFloat()
        {
            // A sorted (increasing) array
            List<float> x = Enumerable.Range(0, 1000000).Select(i => (float)i).ToList();
            Assert.True(Utils.IsMonotonicallyIncreasing(x));

            // A monotonically increasing array
            var x1Temp = x[1];
            var x7Temp = x[7];
            x[1] = x[0];
            x[7] = x[6];
            Assert.True(Utils.IsMonotonicallyIncreasing(x));
            x[1] = x1Temp;
            x[7] = x7Temp;

            // Not sorted
            x[1] = x[6];
            Assert.False(Utils.IsMonotonicallyIncreasing(x));
            x[1] = x1Temp;
            
            // NaN: `Array.Sort()` will put NaNs into the first position,
            // but we want to guarantee that NaNs aren't allowed in these arrays.
            var x0Temp = x[0];
            x[0] = float.NaN;
            Assert.False(Utils.IsMonotonicallyIncreasing(x));
            x[0] = x0Temp;

            // Null lists are considered to be sorted
            List<float> nullX = null;
            Assert.True(Utils.IsMonotonicallyIncreasing(nullX));
        }

        [Fact]
        [TestCategory("Utilities")]
        public void CheckIsMonotonicallyIncreasingDouble()
        {
            // A sorted (increasing) array
            double[] x = Enumerable.Range(0, 1000000).Select(i => (double)i).ToArray();
            Assert.True(Utils.IsMonotonicallyIncreasing(x));

            // A monotonically increasing array
            var x1Temp = x[1];
            var x7Temp = x[7];
            x[1] = x[0];
            x[7] = x[6];
            Assert.True(Utils.IsMonotonicallyIncreasing(x));
            x[1] = x1Temp;
            x[7] = x7Temp;

            // Not sorted
            x[1] = x[6];
            Assert.False(Utils.IsMonotonicallyIncreasing(x));
            x[1] = x1Temp;

            // NaN: `Array.Sort()` will put NaNs into the first position,
            // but we want to guarantee that NaNs aren't allowed in these arrays.
            var x0Temp = x[0];
            x[0] = float.NaN;
            Assert.False(Utils.IsMonotonicallyIncreasing(x));
            x[0] = x0Temp;

            // Null lists are considered to be sorted
            List<float> nullX = null;
            Assert.True(Utils.IsMonotonicallyIncreasing(nullX));
        }

        [Fact]
        [TestCategory("Utilities")]
        public void CheckIsIncreasing()
        {
            // An increasing array
            int[] x = Enumerable.Range(0, 10).ToArray();
            Assert.True(Utils.IsIncreasing(0, x, 10));
            // Check the lower bound
            Assert.False(Utils.IsIncreasing(1, x, 10));
            // The upper bound should be exclusive
            Assert.False(Utils.IsIncreasing(0, x, 9));
            // Any length shorter than the array should work
            Assert.True(Utils.IsIncreasing(0, x, 0, 10));
            Assert.True(Utils.IsIncreasing(0, x, 1, 10));
            Assert.True(Utils.IsIncreasing(0, x, 5, 10));
            Assert.True(Utils.IsIncreasing(0, x, 10, 10));
            // Lengths longer than the array shall throw
            Assert.Throws<InvalidOperationException>(() => Utils.IsIncreasing(0, x, 11, 10));

            // A monotonically increasing array should fail
            var x7Temp = x[7];
            x[7] = x[6];
            Assert.False(Utils.IsIncreasing(0, x, 10));
            // But until then, it should be fine
            Assert.True(Utils.IsIncreasing(0, x, 7, 10));
            x[7] = x7Temp;

            // Not sorted
            x[7] = x[9];
            Assert.False(Utils.IsIncreasing(0, x, 10));
            // Before the mismatched entry, it should be fine
            Assert.True(Utils.IsIncreasing(0, x, 7, 10));
            x[1] = x7Temp;

            // Null arrays return true
            int[] nullX = null;
            Assert.True(Utils.IsIncreasing(0, nullX, 10));

            // Null arrays with a length accession shall throw an exception
            Assert.Throws<InvalidOperationException>(() => Utils.IsIncreasing(0, nullX, 7, 10));
        }

        [Fact]
        [TestCategory("Utilities")]
        public void CheckAreEqualInt()
        {
            // A sorted (increasing) array
            int[] x = Enumerable.Range(0, 10).ToArray();
            int[] y = Enumerable.Range(0, 10).ToArray();
            Assert.True(Utils.AreEqual(x, y));

            // Not Equal
            var x1Temp = x[1];
            x[1] = x[0];
            Assert.False(Utils.AreEqual(x, y));
            x[1] = x1Temp;

            // Beginning is different
            var x0Temp = x[0];
            x[0] = x[x.Length - 1];
            Assert.False(Utils.AreEqual(x, y));
            x[0] = x0Temp;

            // End is different
            var xLengthTemp = x[x.Length - 1];
            x[x.Length - 1] = x[0];
            Assert.False(Utils.AreEqual(x, y));
            x[x.Length - 1] = xLengthTemp;

            // Different Array Lengths
            int[] xOfDifferentLength = Enumerable.Range(0, 9).ToArray();
            Assert.False(Utils.AreEqual(xOfDifferentLength, y));

            // Nulls
            Assert.False(Utils.AreEqual(null, y));
            Assert.False(Utils.AreEqual(x, null));
        }

        [Fact]
        [TestCategory("Utilities")]
        public void CheckAreEqualBool()
        {
            // A sorted (increasing) array
            bool[] x = new bool[] { true, true, false, false };
            bool[] y = new bool[] { true, true, false, false };
            Assert.True(Utils.AreEqual(x, y));

            // Not Equal
            var x1Temp = x[1];
            x[1] = x[2];
            Assert.False(Utils.AreEqual(x, y));
            x[1] = x1Temp;

            // Beginning is different
            var x0Temp = x[0];
            x[0] = x[x.Length - 1];
            Assert.False(Utils.AreEqual(x, y));
            x[0] = x0Temp;

            // End is different
            var xLengthTemp = x[x.Length - 1];
            x[x.Length - 1] = x[0];
            Assert.False(Utils.AreEqual(x, y));
            x[x.Length - 1] = xLengthTemp;

            // Different Array Lengths
            bool[] xOfDifferentLength = new bool[] { true, true, false };
            Assert.False(Utils.AreEqual(xOfDifferentLength, y));

            // Nulls
            Assert.False(Utils.AreEqual(null, y));
            Assert.False(Utils.AreEqual(x, null));
        }

        [Fact]
        [TestCategory("Utilities")]
        public void CheckAreEqualFloat()
        {
            // A sorted (increasing) array
            float[] x = Enumerable.Range(0, 10).Select(i => (float) i).ToArray();
            float[] y = Enumerable.Range(0, 10).Select(i => (float)i).ToArray();
            Assert.True(Utils.AreEqual(x, y));

            // Not Equal
            var x1Temp = x[1];
            x[1] = x[0];
            Assert.False(Utils.AreEqual(x, y));
            x[1] = x1Temp;

            // Beginning is different
            var x0Temp = x[0];
            x[0] = x[x.Length - 1];
            Assert.False(Utils.AreEqual(x, y));
            x[0] = x0Temp;

            // End is different
            var xLengthTemp = x[x.Length - 1];
            x[x.Length - 1] = x[0];
            Assert.False(Utils.AreEqual(x, y));
            x[x.Length - 1] = xLengthTemp;

            // Different Array Lengths
            float[] xOfDifferentLength = Enumerable.Range(0, 9).Select(i => (float)i).ToArray();
            Assert.False(Utils.AreEqual(xOfDifferentLength, y));

            // Nulls
            Assert.False(Utils.AreEqual(null, y));
            Assert.False(Utils.AreEqual(x, null));
        }

        [Fact]
        [TestCategory("Utilities")]
        public void CheckAreEqualDouble()
        {
            // A sorted (increasing) array
            double[] x = Enumerable.Range(0, 10).Select(i => (double)i).ToArray();
            double[] y = Enumerable.Range(0, 10).Select(i => (double)i).ToArray();
            Assert.True(Utils.AreEqual(x, y));

            // Not Equal
            var x1Temp = x[1];
            x[1] = x[0];
            Assert.False(Utils.AreEqual(x, y));
            x[1] = x1Temp;

            // Beginning is different
            var x0Temp = x[0];
            x[0] = x[x.Length - 1];
            Assert.False(Utils.AreEqual(x, y));
            x[0] = x0Temp;

            // End is different
            var xLengthTemp = x[x.Length - 1];
            x[x.Length - 1] = x[0];
            Assert.False(Utils.AreEqual(x, y));
            x[x.Length - 1] = xLengthTemp;

            // Different Array Lengths
            double[] xOfDifferentLength = Enumerable.Range(0, 9).Select(i => (double)i).ToArray();
            Assert.False(Utils.AreEqual(xOfDifferentLength, y));

            // Nulls
            Assert.False(Utils.AreEqual(null, y));
            Assert.False(Utils.AreEqual(x, null));
        }

        [Fact]
        [TestCategory("SkipInCI")]
        public void TestDownloadFromLocal()
        {
            var envVarOld = Environment.GetEnvironmentVariable(ResourceManagerUtils.CustomResourcesUrlEnvVariable);
            var resourcePathVarOld = Environment.GetEnvironmentVariable(Utils.CustomSearchDirEnvVariable);
            Environment.SetEnvironmentVariable(Utils.CustomSearchDirEnvVariable, null);

            var baseDir = GetOutputPath("resources");
            Assert.True(Uri.TryCreate(baseDir, UriKind.Absolute, out var baseDirUri), "Uri.TryCreate failed");
            Environment.SetEnvironmentVariable(ResourceManagerUtils.CustomResourcesUrlEnvVariable, baseDirUri.AbsoluteUri);
            var envVar = Environment.GetEnvironmentVariable(ResourceManagerUtils.CustomResourcesUrlEnvVariable);
            Assert.True(envVar == baseDirUri.AbsoluteUri);
            var path = DeleteOutputPath(Path.Combine("resources", "subdir"), "breast-cancer.txt");

            var bc = GetDataPath("breast-cancer.txt");
            File.Copy(bc, path);

            var saveToDir = GetOutputPath("copyto");
            DeleteOutputPath("copyto", "breast-cancer.txt");
            var sbOut = new StringBuilder();
            var env = new ConsoleEnvironment(42);
            using (var ch = env.Start("Downloading"))
            {
                try
                {
                    var t = ResourceManagerUtils.Instance.EnsureResourceAsync(env, ch, "subdir/breast-cancer.txt", "breast-cancer.txt", saveToDir, 1 * 60 * 1000);
                    t.Wait();

                    if (t.Result.ErrorMessage != null)
                        Fail(String.Format("Expected zero length error string. Received error: {0}", t.Result.ErrorMessage));
                    if (t.Status != TaskStatus.RanToCompletion)
                        Fail("Download did not complete succesfully");
                    if (!File.Exists(GetOutputPath("copyto", "breast-cancer.txt")))
                    {
                        Fail($"File '{GetOutputPath("copyto", "breast-cancer.txt")}' does not exist. " +
                            $"File was downloaded to '{t.Result.FileName}' instead." +
                            $"MICROSOFTML_RESOURCE_PATH is set to {Environment.GetEnvironmentVariable(Utils.CustomSearchDirEnvVariable)}");
                    }
                    Done();
                }
                finally
                {
                    // Set environment variable back to its old value.
                    Environment.SetEnvironmentVariable(ResourceManagerUtils.CustomResourcesUrlEnvVariable, envVarOld);
                    Environment.SetEnvironmentVariable(Utils.CustomSearchDirEnvVariable, resourcePathVarOld);
                }
            }
        }

        [Fact]
        [TestCategory("SkipInCI")]
        public void TestDownloadError()
        {
            var envVarOld = Environment.GetEnvironmentVariable(ResourceManagerUtils.CustomResourcesUrlEnvVariable);
            var timeoutVarOld = Environment.GetEnvironmentVariable(ResourceManagerUtils.TimeoutEnvVariable);
            var resourcePathVarOld = Environment.GetEnvironmentVariable(Utils.CustomSearchDirEnvVariable);
            Environment.SetEnvironmentVariable(Utils.CustomSearchDirEnvVariable, null);

            // Bad path.
            try
            {
                if (!Uri.TryCreate($@"\\ct01\public\{Guid.NewGuid()}\", UriKind.Absolute, out var badUri))
                    Fail("Uri could not be created");
                Environment.SetEnvironmentVariable(ResourceManagerUtils.CustomResourcesUrlEnvVariable, badUri.AbsoluteUri);
                var envVar = Environment.GetEnvironmentVariable(ResourceManagerUtils.CustomResourcesUrlEnvVariable);
                if (envVar != badUri.AbsoluteUri)
                    Fail("Environment variable not set properly");

                var saveToDir = GetOutputPath("copyto");
                DeleteOutputPath("copyto", "breast-cancer.txt");
                var sbOut = new StringBuilder();
                var sbErr = new StringBuilder();
                using (var outWriter = new StringWriter(sbOut))
                using (var errWriter = new StringWriter(sbErr))
                {
                    var env = new ConsoleEnvironment(42, outWriter: outWriter, errWriter: errWriter);
                    using (var ch = env.Start("Downloading"))
                    {
                        var t = ResourceManagerUtils.Instance.EnsureResourceAsync(env, ch, "breast-cancer.txt", "breast-cancer.txt", saveToDir, 10 * 1000);
                        t.Wait();

                        Log("Bad path");
                        Log($"out: {sbOut.ToString()}");
                        Log($"error: {sbErr.ToString()}");
                        if (t.Status != TaskStatus.RanToCompletion)
                            Fail("Download did not complete succesfully");
                        if (File.Exists(Path.Combine(saveToDir, "breast-cancer.txt")))
                            Fail($"File '{GetOutputPath("copyto", "breast-cancer.txt")}' should have been deleted.");
                    }
                }

                // Good path, bad file name.
                if (!Uri.TryCreate(GetDataPath("breast-cancer.txt")+"bad_addition", UriKind.Absolute, out var goodUri))
                    Fail("Uri could not be created");

                Environment.SetEnvironmentVariable(ResourceManagerUtils.CustomResourcesUrlEnvVariable, goodUri.AbsoluteUri);
                envVar = Environment.GetEnvironmentVariable(ResourceManagerUtils.CustomResourcesUrlEnvVariable);
                if (envVar != goodUri.AbsoluteUri)
                    Fail("Environment variable not set properly");

                DeleteOutputPath("copyto", "breast-cancer.txt");
                sbOut.Clear();
                sbErr.Clear();
                using (var outWriter = new StringWriter(sbOut))
                using (var errWriter = new StringWriter(sbErr))
                {
                    var env = new ConsoleEnvironment(42, outWriter: outWriter, errWriter: errWriter);

                    using (var ch = env.Start("Downloading"))
                    {
                        var t = ResourceManagerUtils.Instance.EnsureResourceAsync(env, ch, "breast-cancer1.txt", "breast-cancer.txt", saveToDir, 10 * 1000);
                        t.Wait();

                        Log("Good path, bad file name");
                        Log($"out: {sbOut.ToString()}");
                        Log($"error: {sbErr.ToString()}");

                        if (t.Status != TaskStatus.RanToCompletion)
                            Fail("Download did not complete succesfully");
                        if (File.Exists(Path.Combine(saveToDir, "breast-cancer.txt")))
                            Fail($"File '{GetOutputPath("copyto", "breast-cancer.txt")}' should have been deleted.");
                    }
                }

                // Bad url.
                if (!Uri.TryCreate("https://fake-website/fake-model.model/", UriKind.Absolute, out badUri))
                    Fail("Uri could not be created");

                Environment.SetEnvironmentVariable(ResourceManagerUtils.CustomResourcesUrlEnvVariable, badUri.AbsoluteUri);
                envVar = Environment.GetEnvironmentVariable(ResourceManagerUtils.CustomResourcesUrlEnvVariable);
                if (envVar != badUri.AbsoluteUri)
                    Fail("Environment variable not set properly");

                DeleteOutputPath("copyto", "ResNet_18_Updated.model");
                sbOut.Clear();
                sbErr.Clear();
                using (var outWriter = new StringWriter(sbOut))
                using (var errWriter = new StringWriter(sbErr))
                {
                    var env = new ConsoleEnvironment(42, outWriter: outWriter, errWriter: errWriter);
                    using (var ch = env.Start("Downloading"))
                    {
                        var fileName = "test_bad_url.model";
                        var t = ResourceManagerUtils.Instance.EnsureResourceAsync(env, ch, "Image/ResNet_18_Updated.model", fileName, saveToDir, 10 * 1000);
                        t.Wait();

                        Log("Bad url");
                        Log($"out: {sbOut.ToString()}");
                        Log($"error: {sbErr.ToString()}");

                        if (t.Status != TaskStatus.RanToCompletion)
                            Fail("Download did not complete succesfully");
                        if (File.Exists(Path.Combine(saveToDir, fileName)))
                            Fail($"File '{Path.Combine(saveToDir, fileName)}' should have been deleted.");
                    }
                }

                // Good url, bad page.
                if (!Uri.TryCreate("https://cnn.com/", UriKind.Absolute, out var cnn))
                    Fail("Uri could not be created");
                Environment.SetEnvironmentVariable(ResourceManagerUtils.CustomResourcesUrlEnvVariable, cnn.AbsoluteUri);
                envVar = Environment.GetEnvironmentVariable(ResourceManagerUtils.CustomResourcesUrlEnvVariable);
                if (envVar != cnn.AbsoluteUri)
                    Fail("Environment variable not set properly");

                DeleteOutputPath("copyto", "ResNet_18_Updated.model");
                sbOut.Clear();
                sbErr.Clear();
                using (var outWriter = new StringWriter(sbOut))
                using (var errWriter = new StringWriter(sbErr))
                {
                    var env = new ConsoleEnvironment(42, outWriter: outWriter, errWriter: errWriter);
                    using (var ch = env.Start("Downloading"))
                    {
                        var fileName = "test_cnn_page_does_not_exist.model";
                        var t = ResourceManagerUtils.Instance.EnsureResourceAsync(env, ch, "Image/ResNet_18_Updated.model", fileName, saveToDir, 10 * 1000);
                        t.Wait();

                        Log("Good url, bad page");
                        Log($"out: {sbOut.ToString()}");
                        Log($"error: {sbErr.ToString()}");

                        if (t.Status != TaskStatus.RanToCompletion)
                            Fail("Download did not complete succesfully");
#if !CORECLR
                        if (!sbErr.ToString().Contains("(404) Not Found"))
                            Fail($"Error message should contain '(404) Not Found. Instead: {sbErr.ToString()}");
#endif
                        if (File.Exists(Path.Combine(saveToDir, fileName)))
                            Fail($"File '{Path.Combine(saveToDir, fileName)}' should have been deleted.");
                    }
                }

                // Download from local, short time out.
#if CORECLR
                var path = Path.Combine(Path.GetDirectoryName(typeof(TestImageAnalyticsTransforms).Assembly.Location), "..", "AutoLoad");
#else
                var path = Path.GetDirectoryName(typeof(TestImageAnalyticsTransforms).Assembly.Location);
#endif

                Assert.True(Uri.TryCreate(path, UriKind.Absolute, out var baseDirUri));
                Environment.SetEnvironmentVariable(ResourceManagerUtils.CustomResourcesUrlEnvVariable, baseDirUri.AbsoluteUri);
                envVar = Environment.GetEnvironmentVariable(ResourceManagerUtils.CustomResourcesUrlEnvVariable);
                if (envVar != baseDirUri.AbsoluteUri)
                    Fail("Environment variable not set properly");

                Environment.SetEnvironmentVariable(ResourceManagerUtils.CustomResourcesUrlEnvVariable, null);
                envVar = Environment.GetEnvironmentVariable(ResourceManagerUtils.CustomResourcesUrlEnvVariable);
                if (envVar != null)
                    Fail("Environment variable not set properly");

                Environment.SetEnvironmentVariable(ResourceManagerUtils.TimeoutEnvVariable, "10");
                envVar = Environment.GetEnvironmentVariable(ResourceManagerUtils.TimeoutEnvVariable);
                if (envVar != "10")
                    Fail("Environment variable not set properly");

                DeleteOutputPath("copyto", "ResNet_18_Updated.model");
                sbOut.Clear();
                sbErr.Clear();
                using (var outWriter = new StringWriter(sbOut))
                using (var errWriter = new StringWriter(sbErr))
                {
                    var env = new ConsoleEnvironment(42, outWriter: outWriter, errWriter: errWriter);
                    using (var ch = env.Start("Downloading"))
                    {
                        var fileName = "test_short_timeout.model";
                        var t = ResourceManagerUtils.Instance.EnsureResourceAsync(env, ch, "Image/AlexNet_Updated.model", fileName, saveToDir, 10 * 1000);
                        t.Wait();

                        Log("Default url, short time out");
                        Log($"out: {sbOut.ToString()}");
                        Log($"error: {sbErr.ToString()}");

#if !CORECLR
                        if (!sbErr.ToString().Contains("Download timed out"))
                            Fail($"Error message should contain the string 'Download timed out'. Instead: {sbErr.ToString()}");
#endif
                        if (File.Exists(Path.Combine(saveToDir, fileName)))
                            Fail($"File '{Path.Combine(saveToDir, fileName)}' should have been deleted.");
                    }
                }
                Done();
            }
            finally
            {
                // Set environment variable back to its old value.
                Environment.SetEnvironmentVariable(ResourceManagerUtils.CustomResourcesUrlEnvVariable, envVarOld);
                Environment.SetEnvironmentVariable(ResourceManagerUtils.TimeoutEnvVariable, timeoutVarOld);
                Environment.SetEnvironmentVariable(Utils.CustomSearchDirEnvVariable, resourcePathVarOld);
            }
        }
    }
}
