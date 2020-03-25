// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Diagnostics;
using System.Linq;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.Extensions.Primitives;
using Microsoft.ML;

namespace Microsoft.Extensions.ML
{
    internal class UriModelLoader : ModelLoader, IDisposable
    {
        //TODO: This should be able to be removed for HeaderNames.ETag
        private const string ETagHeader = "ETag";
        private const int TimeoutMilliseconds = 60000;
        private readonly MLContext _context;
        private TimeSpan? _timerPeriod;
        private Uri _uri;
        private ITransformer _model;
        private ModelReloadToken _reloadToken;
        private Timer _reloadTimer;
        private readonly object _reloadTimerLock;
        private string _eTag;
        private readonly ILogger _logger;
        private readonly CancellationTokenSource _stopping;
        private bool _started;

        public UriModelLoader(IOptions<MLOptions> contextOptions, ILogger<UriModelLoader> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _context = contextOptions.Value?.MLContext;
            _reloadTimerLock = new object();
            _reloadToken = new ModelReloadToken();
            _stopping = new CancellationTokenSource();
            _started = false;
        }

        internal void Start(Uri uri, TimeSpan period)
        {
            _timerPeriod = period;
            _uri = uri;
            if (LoadModelAsync().ConfigureAwait(false).GetAwaiter().GetResult())
            {
                StartReloadTimer();
            }
            _started = true;
        }

        private void ReloadTimerTick(object state)
        {
            _ = Task.Run(async () =>
            {
                StopReloadTimer();

                await RunAsync();

                StartReloadTimer();
            });
        }

        internal bool IsStopping => _stopping.IsCancellationRequested;

        internal async Task RunAsync()
        {
            CancellationTokenSource cancellation = null;
            //TODO: Switch to ValueStopWatch
            var duration = Stopwatch.StartNew();
            try
            {
                cancellation = CancellationTokenSource.CreateLinkedTokenSource(_stopping.Token);
                cancellation.CancelAfter(TimeoutMilliseconds);
                Logger.UriReloadBegin(_logger, _uri);

                var eTagMatches = await MatchEtagAsync(_uri, _eTag);
                if (!eTagMatches)
                {
                    await LoadModelAsync();
                    var previousToken = Interlocked.Exchange(ref _reloadToken, new ModelReloadToken());
                    previousToken.OnReload();
                }

                Logger.UriReloadEnd(_logger, _uri, duration.Elapsed);
            }
            catch (OperationCanceledException) when (IsStopping)
            {
                // This is a cancellation - if the app is shutting down we want to ignore it.
            }
            catch (Exception ex)
            {
                Logger.UriReloadError(_logger, _uri, duration.Elapsed, ex);
            }
            finally
            {
                cancellation.Dispose();
            }
        }

        internal virtual async Task<bool> MatchEtagAsync(Uri uri, string eTag)
        {
            using (var client = new HttpClient())
            {
                var headRequest = new HttpRequestMessage(HttpMethod.Head, uri);
                var resp = await client.SendAsync(headRequest);

                return resp.Headers.GetValues(ETagHeader).First() == eTag;
            }
        }

        internal void StartReloadTimer()
        {
            lock (_reloadTimerLock)
            {
                if (_reloadTimer == null)
                {
                    _reloadTimer = new Timer(ReloadTimerTick, this, Convert.ToInt32(_timerPeriod.Value.TotalMilliseconds), Timeout.Infinite);
                }
            }
        }

        internal void StopReloadTimer()
        {
            lock (_reloadTimerLock)
            {
                _reloadTimer.Dispose();
                _reloadTimer = null;
            }
        }

        internal virtual async Task<bool> LoadModelAsync()
        {
            //TODO: We probably need some sort of retry policy for this.
            try
            {
                using (var client = new HttpClient())
                {
                    var resp = await client.GetAsync(_uri);
                    using (var stream = await resp.Content.ReadAsStreamAsync())
                    {
                        _model = _context.Model.Load(stream, out _);
                    }

                    if (resp.Headers.Contains(ETagHeader))
                    {
                        _eTag = resp.Headers.GetValues(ETagHeader).First();
                        return true;
                    }
                    return false;
                }
            }
            catch (Exception ex)
            {
                Logger.UriLoadError(_logger, _uri, ex);
                throw;
            }
        }

        public override ITransformer GetModel()
        {
            if (!_started) throw new InvalidOperationException("Start must be called on a ModelLoader before it can be used.");

            return _model;
        }

        public override IChangeToken GetReloadToken()
        {
            if (!_started) throw new InvalidOperationException("Start must be called on a ModelLoader before it can be used.");

            return _reloadToken;
        }

        public void Dispose()
        {
            _reloadTimer?.Dispose();
        }

        internal static class EventIds
        {
            public static readonly EventId UriReloadBegin = new EventId(100, "UriReloadBegin");
            public static readonly EventId UriReloadEnd = new EventId(101, "UriReloadEnd");
            public static readonly EventId UriReloadError = new EventId(102, "UriReloadError");
            public static readonly EventId UriLoadError = new EventId(103, "UriLoadError");
        }

        private static class Logger
        {
            private static readonly Action<ILogger, Uri, Exception> _uriReloadBegin = LoggerMessage.Define<Uri>(
                LogLevel.Debug,
                EventIds.UriReloadBegin,
                "URI reload '{uri}'");

            private static readonly Action<ILogger, Uri, double, Exception> _uriReloadEnd = LoggerMessage.Define<Uri, double>(
                LogLevel.Debug,
                EventIds.UriReloadEnd,
                "URI reload '{uri}' completed after {ElapsedMilliseconds}ms");

            private static readonly Action<ILogger, Uri, double, Exception> _uriReloadError = LoggerMessage.Define<Uri, double>(
                LogLevel.Error,
                EventIds.UriReloadError,
                "URI reload for {uri} threw an unhandled exception after {ElapsedMilliseconds}ms");

            private static readonly Action<ILogger, Uri, Exception> _uriLoadError = LoggerMessage.Define<Uri>(
                LogLevel.Error,
                EventIds.UriLoadError,
                "Error loading {uri}");

            public static void UriReloadBegin(ILogger logger, Uri uri)
            {
                _uriReloadBegin(logger, uri, null);
            }

            public static void UriReloadEnd(ILogger logger, Uri uri, TimeSpan duration)
            {
                _uriReloadEnd(logger, uri, duration.TotalMilliseconds, null);
            }

            public static void UriReloadError(ILogger logger, Uri uri, TimeSpan duration, Exception exception)
            {
                _uriReloadError(logger, uri, duration.TotalMilliseconds, exception);
            }

            public static void UriLoadError(ILogger logger, Uri uri, Exception exception)
            {
                _uriLoadError(logger, uri, exception);
            }
        }
    }
}
