using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
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
        public async Task TestDownloadError()
        {
            var envVarOld = Environment.GetEnvironmentVariable(ResourceManagerUtils.CustomResourcesUrlEnvVariable);
            var resourcePathVarOld = Environment.GetEnvironmentVariable(Utils.CustomSearchDirEnvVariable);
            Environment.SetEnvironmentVariable(Utils.CustomSearchDirEnvVariable, null);

            try
            {
                var envVar = Environment.GetEnvironmentVariable(ResourceManagerUtils.CustomResourcesUrlEnvVariable);
                var saveToDir = GetOutputPath("copyto");
                DeleteOutputPath("copyto", "breast-cancer.txt");
                var sbOut = new StringBuilder();
                var sbErr = new StringBuilder();

                // Bad url.
                if (!Uri.TryCreate("https://fake-website/fake-model.model/", UriKind.Absolute, out var badUri))
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
                        await Assert.ThrowsAsync<NotSupportedException>(() => ResourceManagerUtils.Instance.EnsureResourceAsync(env, ch, "Image/ResNet_18_Updated.model", fileName, saveToDir, 10 * 1000));

                        Log("Bad url");
                        Log($"out: {sbOut.ToString()}");
                        Log($"error: {sbErr.ToString()}");

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
                        await Assert.ThrowsAsync<NotSupportedException>(() => ResourceManagerUtils.Instance.EnsureResourceAsync(env, ch, "Image/ResNet_18_Updated.model", fileName, saveToDir, 10 * 1000));

                        Log("Good url, bad page");
                        Log($"out: {sbOut.ToString()}");
                        Log($"error: {sbErr.ToString()}");

                        if (File.Exists(Path.Combine(saveToDir, fileName)))
                            Fail($"File '{Path.Combine(saveToDir, fileName)}' should have been deleted.");
                    }
                }

                //Good url, good page
                Environment.SetEnvironmentVariable(ResourceManagerUtils.CustomResourcesUrlEnvVariable, envVarOld);
                envVar = Environment.GetEnvironmentVariable(ResourceManagerUtils.CustomResourcesUrlEnvVariable);
                if (envVar != envVarOld)
                    Fail("Environment variable not set properly");

                DeleteOutputPath("copyto", "sentiment.emd");
                sbOut.Clear();
                sbErr.Clear();
                using (var outWriter = new StringWriter(sbOut))
                using (var errWriter = new StringWriter(sbErr))
                {
                    var env = new ConsoleEnvironment(42, outWriter: outWriter, errWriter: errWriter);
                    using (var ch = env.Start("Downloading"))
                    {
                        var fileName = "sentiment.emd";
                        var t = ResourceManagerUtils.Instance.EnsureResourceAsync(env, ch, "text/Sswe/sentiment.emd", fileName, saveToDir, 1 * 60 * 1000);
                        var results = await t;

                        if (results.ErrorMessage != null)
                            Fail(String.Format("Expected zero length error string. Received error: {0}", results.ErrorMessage));
                        if (t.Status != TaskStatus.RanToCompletion)
                            Fail("Download did not complete succesfully");
                        if (!File.Exists(GetOutputPath("copyto", "sentiment.emd")))
                        {
                            Fail($"File '{GetOutputPath("copyto", "sentiment.emd")}' does not exist. " +
                                $"File was downloaded to '{results.FileName}' instead." +
                                $"MICROSOFTML_RESOURCE_PATH is set to {Environment.GetEnvironmentVariable(Utils.CustomSearchDirEnvVariable)}");
                        }
                    }
                }
                Done();
            }
            finally
            {
                // Set environment variable back to its old value.
                Environment.SetEnvironmentVariable(ResourceManagerUtils.CustomResourcesUrlEnvVariable, envVarOld);
                Environment.SetEnvironmentVariable(Utils.CustomSearchDirEnvVariable, resourcePathVarOld);

                if (File.Exists(GetOutputPath("copyto", "sentiment.emd")))
                    File.Delete(GetOutputPath("copyto", "sentiment.emd"));
            }
        }

        [Fact]
        [TestCategory("ResourceDownload")]
        [Trait("Category", "RunSpecificTest")]
        public void TestDatasetFileDownload()
        {
            int numberOfParallel = 15;
            int numberOfIterations = 20;

            var env = new ConsoleEnvironment(1);
            var fileList = new List<string> { "MSLRWeb10KTest240kRows.tsv", "MSLRWeb10KTrain720kRows.tsv", 
                "MSLRWeb10KValidate240kRows.tsv", "WikiDetoxAnnotated160kRows.tsv" };

            for (int j = 0; j < numberOfIterations; j++)
            {
                var tasks = new List<Task>();
                var paths = new List<string>();
                for (int i = 0; i < numberOfParallel; i++)
                {
                    string guid = Guid.NewGuid().ToString();
                    var path = Path.Combine(Path.GetTempPath(), "MLNET", guid);
                    paths.Add(path);

                    foreach (var file in fileList)
                    {
#pragma warning disable VSTHRD105 // Avoid method overloads that assume TaskScheduler.Current
                        tasks.Add(Task.Factory.StartNew(() => Download(env, "benchmarks/" + file, path, file)));
#pragma warning restore VSTHRD105 // Avoid method overloads that assume TaskScheduler.Current
                    }
                }

                Task.WaitAll(tasks.ToArray());
                Console.WriteLine($"Test: Finish download for {j}-th round.");
                Thread.Sleep(10 * 1000);
            }
        }

        [Fact]
        [TestCategory("ResourceDownload")]
        [Trait("Category", "RunSpecificTest")]
        public void TestMetaFileDownload()
        {
            int numberOfParallel = 15;
            int numberOfIterations = 20;

            var env = new ConsoleEnvironment(1);
            var fileList = new List<string> { "inception_v3.meta", "mobilenet_v2.meta", "resnet_v2_50_299.meta", "resnet_v2_101_299.meta" };

            for (int j = 0; j < numberOfIterations; j++)
            {
                var tasks = new List<Task>();
                var paths = new List<string>();
                for (int i = 0; i < numberOfParallel; i++)
                {
                    string guid = Guid.NewGuid().ToString();
                    var path = Path.Combine(Path.GetTempPath(), "MLNET", guid);
                    paths.Add(path);

                    foreach (var file in fileList)
                    {
#pragma warning disable VSTHRD105 // Avoid method overloads that assume TaskScheduler.Current
                        tasks.Add(Task.Factory.StartNew(() => Download(env, file, path, file)));
#pragma warning restore VSTHRD105 // Avoid method overloads that assume TaskScheduler.Current
                    }
                }

                Task.WaitAll(tasks.ToArray());
                Console.WriteLine($"Test: Finish download for {j}-th round.");
                Thread.Sleep(10*1000);
            }
        }

        private void Download(IHostEnvironment env, string url, string destDir, string destFileName)
        {
            if (destFileName == null)
                destFileName = url.Split(Path.DirectorySeparatorChar).Last();

            Directory.CreateDirectory(destDir);

            int timeout = 10 * 60 * 1000;
            using (var ch = env.Start("Test Download files"))
            {
                var ensureModel = ResourceManagerUtils.Instance.EnsureResourceAsync(env, ch, url, destFileName, destDir, timeout);
                ensureModel.Wait();
                var errorResult = ResourceManagerUtils.GetErrorMessage(out var errorMessage, ensureModel.Result);
                if (errorResult != null)
                {
                    Console.WriteLine($"Test: Dowload fail for {destDir}/{destFileName}");
                    return;
                }
            }

            if (File.Exists(Path.Combine(destDir, destFileName)))
            {
                File.Delete(Path.Combine(destDir, destFileName));
            }
        }
    }
}
