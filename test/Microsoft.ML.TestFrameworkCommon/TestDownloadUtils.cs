// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;

namespace Microsoft.ML.TestFrameworkCommon
{
    public static class TestDownloadUtils
    {
        public static void DownloadFile(string url, string filePath, TimeSpan timeout)
        {
            var directory = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(directory))
                Directory.CreateDirectory(directory);

            string temporaryFilePath = $"{filePath}.{Guid.NewGuid():N}.download";
            using (var client = new HttpClient { Timeout = Timeout.InfiniteTimeSpan })
            {
                RetryHelper.Execute(() =>
                {
                    try
                    {
                        if (File.Exists(filePath))
                            return;

                        using (var cancellationSource = new CancellationTokenSource(timeout))
                        using (var response = client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationSource.Token).GetAwaiter().GetResult())
                        {
                            response.EnsureSuccessStatusCode();
                            long? expectedLength = response.Content.Headers.ContentLength;
                            using (var contentStream = response.Content.ReadAsStreamAsync().GetAwaiter().GetResult())
                            using (var fileStream = new FileStream(temporaryFilePath, FileMode.Create, FileAccess.Write, FileShare.None))
                            {
                                contentStream.CopyToAsync(fileStream, 81920, cancellationSource.Token).GetAwaiter().GetResult();
                                if (expectedLength.HasValue && fileStream.Length != expectedLength.Value)
                                    throw new IOException($"Expected {expectedLength.Value} bytes from '{url}', but downloaded {fileStream.Length} bytes.");
                            }
                        }

                        try
                        {
                            File.Move(temporaryFilePath, filePath);
                        }
                        catch (IOException) when (File.Exists(filePath))
                        {
                            // Another caller completed the same download first.
                        }
                    }
                    finally
                    {
                        if (File.Exists(temporaryFilePath))
                            File.Delete(temporaryFilePath);
                    }
                }, backoffFunc: attempt => 10_000,
                    retryWhen: ex => ex is HttpRequestException || ex is TaskCanceledException || ex is IOException);
            }
        }
    }
}
