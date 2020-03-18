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

[assembly: CollectionBehavior(DisableTestParallelization = true)]

namespace Microsoft.ML.Core.Tests.UnitTests
{
    public class TestResourceDownload : BaseTestBaseline
    {
        public TestResourceDownload(ITestOutputHelper helper)
            : base(helper)
        {
        }
        [Fact]
        [TestCategory("ResourceDownload")]
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
        [TestCategory("ResourceDownload")]
        public void TestDownloadError()
        {
            var envVarOld = Environment.GetEnvironmentVariable(ResourceManagerUtils.CustomResourcesUrlEnvVariable);
            var timeoutVarOld = Environment.GetEnvironmentVariable(ResourceManagerUtils.TimeoutEnvVariable);
            var resourcePathVarOld = Environment.GetEnvironmentVariable(Utils.CustomSearchDirEnvVariable);
            Environment.SetEnvironmentVariable(Utils.CustomSearchDirEnvVariable, null);

            // Bad local path.
            try
            {
                if (!Uri.TryCreate(GetDataPath("breast-cancer.txt"), UriKind.Absolute, out var badUri))
                    Fail("Uri could not be created");
                Environment.SetEnvironmentVariable(ResourceManagerUtils.CustomResourcesUrlEnvVariable, badUri.AbsoluteUri);
                var envVar = Environment.GetEnvironmentVariable(ResourceManagerUtils.CustomResourcesUrlEnvVariable);
                if (envVar != badUri.AbsoluteUri)
                    Fail("Environment variable not set properly");

                var saveToDir = GetOutputPath("copyto")+"badLocalPathAddition";
                if (Directory.Exists(saveToDir))
                    Fail("Bad local path should not exist.");
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

                // Good local path, bad file name.
                if (!Uri.TryCreate(GetDataPath("breast-cancer.txt") + "bad_addition", UriKind.Absolute, out var goodUri))
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
                if (!Uri.TryCreate(String.Format("http://aka.ms/mlnet/badurltest/{0}", Guid.NewGuid()), UriKind.Absolute, out badUri))
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
