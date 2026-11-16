// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.RunTests;
using Microsoft.ML.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Core.Tests.UnitTests
{
    // DownloadStreamsBodyToDisk measures process-wide allocations, so this class must not run
    // alongside other test collections.
    [CollectionDefinition(nameof(ResourceDownloadNoParallelizationDefinition), DisableParallelization = true)]
    public class ResourceDownloadNoParallelizationDefinition { }

    /// <summary>
    /// Local, network-free tests for <see cref="ResourceManagerUtils.DownloadResource"/>.
    /// </summary>
    [Collection(nameof(ResourceDownloadNoParallelizationDefinition))]
    public class TestResourceDownloadBehavior : BaseTestBaseline
    {
        public TestResourceDownloadBehavior(ITestOutputHelper output)
            : base(output)
        {
        }

        [Fact]
        public async Task DownloadStreamsBodyToDisk()
        {
            const long payloadSize = 64 * 1024 * 1024;

            using (var server = new LocalPayloadServer(payloadSize))
            using (var workspace = new TempWorkspace())
            {
                string path = Path.Combine(workspace.Path, "payload.bin");
                var env = CreateEnvironment();

                using (var ch = env.Start("test"))
                {
#if NET
                    long allocatedBefore = GC.GetTotalAllocatedBytes(precise: true);
#endif
                    var clock = Stopwatch.StartNew();

                    Exception result = await ResourceManagerUtils.Instance.DownloadResource(
                        env, ch, new HttpClient(), new Uri(server.Url), path, "payload.bin", CancellationToken.None);

                    clock.Stop();

                    Assert.Null(result);
                    Assert.Equal(payloadSize, new FileInfo(path).Length);
                    Assert.Equal(1, server.Requests);

#if NET
                    // GC.GetTotalAllocatedBytes is not available on .NET Framework, so the allocation
                    // check only runs on .NET. The body must stream to disk: allocating the payload in
                    // memory would need >= 64 MB.
                    long allocated = GC.GetTotalAllocatedBytes(precise: true) - allocatedBefore;
                    Assert.True(
                        allocated < payloadSize / 2,
                        $"Expected the download to stream, but {allocated / (1024 * 1024)} MB were allocated for a {payloadSize / (1024 * 1024)} MB payload.");

                    Log($"64 MB in {clock.Elapsed.TotalSeconds:F2}s, {allocated / (1024 * 1024)} MB allocated.");
#else
                    Log($"64 MB in {clock.Elapsed.TotalSeconds:F2}s.");
#endif
                }
            }
        }

        [Fact]
        public async Task DownloadReturnsHttpExceptionForErrorStatus()
        {
            using (var server = new LocalPayloadServer(0, status: 404))
            using (var workspace = new TempWorkspace())
            {
                string path = Path.Combine(workspace.Path, "missing.bin");

                Exception result = await DownloadAsync(server, path);

                Assert.IsType<HttpRequestException>(result);
                Assert.False(File.Exists(path));
                Assert.Empty(Directory.GetFiles(workspace.Path, "temp-resource-*"));
            }
        }

        [Fact]
        public async Task DownloadThrowsWhenRedirectedToDefaultPage()
        {
            using (var server = new LocalPayloadServer(0, status: 302, location: "https://www.microsoft.com/?ref=aka"))
            using (var workspace = new TempWorkspace())
            {
                string path = Path.Combine(workspace.Path, "default-page.bin");
                var env = CreateEnvironment();

                using (var ch = env.Start("test"))
                using (var client = new HttpClient(new HttpClientHandler { AllowAutoRedirect = false }))
                {
                    await Assert.ThrowsAsync<NotSupportedException>(() =>
                        ResourceManagerUtils.Instance.DownloadResource(
                            env, ch, client, new Uri(server.Url), path, "default-page.bin", CancellationToken.None));
                }

                Assert.False(File.Exists(path));
            }
        }

        [Fact]
        public async Task DownloadReturnsCancellationWhenTokenIsCancelled()
        {
            using (var server = new LocalPayloadServer(1024 * 1024, delayMs: 500))
            using (var workspace = new TempWorkspace())
            {
                string path = Path.Combine(workspace.Path, "cancelled.bin");
                var env = CreateEnvironment();
                using (var cts = new CancellationTokenSource())
                {
                    cts.Cancel();

                    using (var ch = env.Start("test"))
                    {
                        Exception result = await ResourceManagerUtils.Instance.DownloadResource(
                            env, ch, new HttpClient(), new Uri(server.Url), path, "cancelled.bin", cts.Token);

                        Assert.IsAssignableFrom<OperationCanceledException>(result);
                    }
                }

                Assert.False(File.Exists(path));
                Assert.Empty(Directory.GetFiles(workspace.Path, "temp-resource-*"));
            }
        }

        [Fact]
        public async Task ConcurrentDownloadsOfSameFileDownloadOnce()
        {
            const long payloadSize = 8 * 1024 * 1024;

            // The server holds the body long enough for the second caller to arrive while the first
            // download is still running.
            using (var server = new LocalPayloadServer(payloadSize, delayMs: 800))
            using (var workspace = new TempWorkspace())
            {
                string path = Path.Combine(workspace.Path, "shared.bin");
                var env = CreateEnvironment();

                var downloads = new[]
                {
                    DownloadAsync(server, path),
                    DownloadAsync(server, path),
                };

                Exception[] results = await Task.WhenAll(downloads);

                Assert.All(results, r => Assert.Null(r));
                Assert.Equal(1, server.Requests);
                Assert.Equal(payloadSize, new FileInfo(path).Length);
                Assert.Empty(Directory.GetFiles(workspace.Path, "temp-resource-*"));
            }
        }

        private static ConsoleEnvironment CreateEnvironment()
            => new ConsoleEnvironment(seed: 42, outWriter: new StringWriter(), errWriter: new StringWriter());

        private static async Task<Exception> DownloadAsync(LocalPayloadServer server, string path)
        {
            var env = CreateEnvironment();
            using (var ch = env.Start("test"))
            {
                return await ResourceManagerUtils.Instance.DownloadResource(
                    env, ch, new HttpClient(), new Uri(server.Url), path, Path.GetFileName(path), CancellationToken.None);
            }
        }

        /// <summary>
        /// Minimal loopback HTTP server: serves a fixed-size payload, a status code, or a redirect.
        /// </summary>
        [Fact]
        public async Task DownloadReturnsIOExceptionImmediatelyForMissingDirectory()
        {
            using (var server = new LocalPayloadServer(1024))
            using (var workspace = new TempWorkspace())
            {
                // A permanent file system error must not be retried until the download timeout.
                string path = Path.Combine(workspace.Path, "missing", "payload.bin");
                var env = CreateEnvironment();

                using (var ch = env.Start("test"))
                {
                    var clock = Stopwatch.StartNew();

                    Exception result = await ResourceManagerUtils.Instance.DownloadResource(
                        env, ch, new HttpClient(), new Uri(server.Url), path, "payload.bin", CancellationToken.None);

                    Assert.IsAssignableFrom<IOException>(result);
                    Assert.True(clock.Elapsed < TimeSpan.FromSeconds(10), $"Took {clock.Elapsed.TotalSeconds:F1}s to report the error.");
                    Assert.Equal(0, server.Requests);
                }
            }
        }

        private sealed class LocalPayloadServer : IDisposable
        {
            private const int ChunkSize = 64 * 1024;

            private readonly HttpListener _listener = new HttpListener();
            private readonly CancellationTokenSource _cts = new CancellationTokenSource();
            private readonly Task _acceptLoop;
            private readonly long _payloadSize;
            private readonly int _status;
            private readonly string _location;
            private readonly int _delayMs;
            private int _requests;

            public string Url { get; }

            public int Requests => Volatile.Read(ref _requests);

            public LocalPayloadServer(long payloadSize, int status = 200, string location = null, int delayMs = 0)
            {
                _payloadSize = payloadSize;
                _status = status;
                _location = location;
                _delayMs = delayMs;

                int port = FindFreePort();
                Url = $"http://127.0.0.1:{port}/payload";
                _listener.Prefixes.Add($"http://127.0.0.1:{port}/");
                _listener.Start();
                _acceptLoop = Task.Run(AcceptLoopAsync);
            }

            private static int FindFreePort()
            {
                var probe = new TcpListener(IPAddress.Loopback, 0);
                probe.Start();
                int port = ((IPEndPoint)probe.LocalEndpoint).Port;
                probe.Stop();
                return port;
            }

            private async Task AcceptLoopAsync()
            {
                while (!_cts.IsCancellationRequested)
                {
                    HttpListenerContext context;
                    try
                    {
                        context = await _listener.GetContextAsync().ConfigureAwait(false);
                    }
                    catch
                    {
                        return;
                    }

                    _ = Task.Run(() => ServeAsync(context));
                }
            }

            private async Task ServeAsync(HttpListenerContext context)
            {
                Interlocked.Increment(ref _requests);

                var response = context.Response;
                try
                {
                    if (_status != 200)
                    {
                        response.StatusCode = _status;
                        if (_location != null)
                            response.RedirectLocation = _location;
                        response.OutputStream.Close();
                        return;
                    }

                    var buffer = new byte[ChunkSize];
                    for (int i = 0; i < buffer.Length; i++)
                        buffer[i] = (byte)'A';
                    response.ContentType = "application/octet-stream";
                    response.ContentLength64 = _payloadSize;

                    long sent = 0;
                    while (sent < _payloadSize && !_cts.IsCancellationRequested)
                    {
                        if (_delayMs > 0 && sent == 0)
                            await Task.Delay(_delayMs).ConfigureAwait(false);

                        int count = (int)Math.Min(ChunkSize, _payloadSize - sent);
                        await response.OutputStream.WriteAsync(buffer, 0, count).ConfigureAwait(false);
                        sent += count;
                    }

                    response.OutputStream.Close();
                }
                catch
                {
                    // Client cancelled or disconnected.
                }
                finally
                {
                    try { response.Close(); } catch { /* already closed */ }
                }
            }

            public void Dispose()
            {
                _cts.Cancel();
                try { _listener.Stop(); } catch { /* ignore */ }
                try { _acceptLoop.Wait(TimeSpan.FromSeconds(5)); } catch { /* ignore */ }
                _listener.Close();
                _cts.Dispose();
            }
        }

        private sealed class TempWorkspace : IDisposable
        {
            public string Path { get; }

            public TempWorkspace()
            {
                Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "mlnet-download-test-" + Guid.NewGuid().ToString("N"));
                Directory.CreateDirectory(Path);
            }

            public void Dispose()
            {
                try { Directory.Delete(Path, recursive: true); } catch { /* best effort */ }
            }
        }
    }
}
