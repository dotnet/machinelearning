// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Microsoft.DotNet.RemoteExecutor;
using Xunit;

namespace Microsoft.ML.Tokenizers.Tests
{
    public class Utils
    {
        public static async Task DownloadFile(string url, string fileName)
        {
            using (var client = new HttpClient() { Timeout = TimeSpan.FromMinutes(5) })
            {
                await RetryHelper.ExecuteAsync(async () =>
                {
                    var response = await client.GetAsync(url);
                    if (response.IsSuccessStatusCode)
                    {
                        var stream = await response.Content.ReadAsStreamAsync();
                        var fileInfo = new FileInfo(fileName);
                        using (var fileStream = fileInfo.OpenWrite())
                        {
                            await stream.CopyToAsync(fileStream);
                        }
                    }
                    else
                    {
                        throw new Exception("File not found");
                    }
                });
            }
        }

        public static void DeleteFile(string file)
        {
            if (File.Exists(file))
            {
                try
                {
                    File.Delete(file);
                }
                catch
                {
                }
            }
        }
        public static string CreateTemporaryFile(string extension) =>
             Path.Combine(Path.GetTempPath(), Path.ChangeExtension(Guid.NewGuid().ToString(), extension));

        public static string SaveEmbeddedResourceFile(string resourceName)
        {
            string fileName = CreateTemporaryFile("txt");
            using Stream fileStream = File.Create(fileName);
            typeof(Utils).Assembly.GetManifestResourceStream(resourceName)!.CopyTo(fileStream);
            return fileName;
        }

        // Runs the remote process in globalization-invariant mode. RuntimeConfigurationOptions would replace the test's
        // runtimeconfig.dev.json, which coverage builds need to find NuGet assemblies, so set the environment variable instead.
        public static RemoteInvokeOptions CreateInvariantGlobalizationOptions()
        {
            RemoteInvokeOptions options = new();
            options.StartInfo.Environment["DOTNET_SYSTEM_GLOBALIZATION_INVARIANT"] = "true";
            return options;
        }

        public static void AssertInvariantGlobalizationMode()
        {
            // string.Normalize returns its input unchanged when globalization-invariant mode is enabled.
            Assert.Equal("\u00E9", "\u00E9".Normalize(NormalizationForm.FormD));
        }
    }
}
