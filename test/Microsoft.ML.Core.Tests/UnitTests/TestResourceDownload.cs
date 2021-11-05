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

[assembly: CollectionBehavior(DisableTestParallelization = true)]

namespace Microsoft.ML.Core.Tests.UnitTests
{
    public class TestResourceDownload : BaseTestBaseline
    {
        public TestResourceDownload(ITestOutputHelper helper)
            : base(helper)
        {
        }

        [Fact(Skip = "Temporarily skipping while helix issues are resolved. Tracked in issue #5845")]
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
                            Fail("Download did not complete successfully");
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
    }
}
