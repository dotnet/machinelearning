// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.Extensions.Primitives;
using Microsoft.ML;

namespace Microsoft.Extensions.ML
{
    internal class FileModelLoader : ModelLoader, IDisposable
    {
        private readonly ILogger<FileModelLoader> _logger;
        private string _filePath;
        private FileSystemWatcher _watcher;
        private ModelReloadToken _reloadToken;
        private ITransformer _model;

        private readonly MLContext _context;

        private readonly object _lock;

        public FileModelLoader(IOptions<MLOptions> contextOptions, ILogger<FileModelLoader> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _context = contextOptions.Value?.MLContext ?? throw new ArgumentNullException(nameof(contextOptions));
            _lock = new object();
        }

        public void Start(string filePath, bool watchFile)
        {
            _filePath = filePath;
            _reloadToken = new ModelReloadToken();

            if (!File.Exists(filePath))
            {
                throw new ArgumentException($"The provided model file {filePath} doesn't exist.");
            }

            var directory = Path.GetDirectoryName(filePath);

            if (string.IsNullOrEmpty(directory))
            {
                directory = Directory.GetCurrentDirectory();
            }

            var file = Path.GetFileName(filePath);

            LoadModel();

            if (watchFile)
            {
                _watcher = new FileSystemWatcher(directory, file);
                _watcher.EnableRaisingEvents = true;
                _watcher.Changed += WatcherChanged;
            }
        }

        private void WatcherChanged(object sender, FileSystemEventArgs e)
        {
            var timer = Stopwatch.StartNew();

            try
            {
                Logger.FileReloadBegin(_logger, _filePath);

                var previousToken = Interlocked.Exchange(ref _reloadToken, new ModelReloadToken());
                lock (_lock)
                {
                    //TODO: We get here multiple times when you copy and paste a file
                    //because of the way file watchers work. Need to think through the
                    //ramifications.
                    LoadModel();
                    Logger.ReloadingFile(_logger, _filePath, timer.Elapsed);
                }
                previousToken.OnReload();
                timer.Stop();
                Logger.FileReloadEnd(_logger, _filePath, timer.Elapsed);
            }
            catch (OperationCanceledException)
            {
                // This is a cancellation - if the app is shutting down we want to ignore it.
            }
            catch (Exception ex)
            {
                Logger.FileReloadError(_logger, _filePath, timer.Elapsed, ex);
            }
        }

        public override IChangeToken GetReloadToken()
        {
            if (_reloadToken == null) throw new InvalidOperationException("Start must be called on a ModelLoader before it can be used.");
            return _reloadToken;
        }

        public override ITransformer GetModel()
        {
            if (_model == null) throw new InvalidOperationException("Start must be called on a ModelLoader before it can be used.");

            return _model;
        }

        private FileStream WaitForFile(string fullPath, FileMode mode, FileAccess access, FileShare share)
        {
            for (int numTries = 0; numTries < 100; numTries++)
            {
                FileStream fs = null;
                try
                {
                    fs = new FileStream(fullPath, mode, access, share);
                    return fs;
                }
                catch (IOException)
                {
                    if (fs != null)
                    {
                        fs.Dispose();
                    }
                    Thread.Sleep(50);
                }
            }

            return null;
        }

        //internal virtual for testing purposes.
        internal virtual void LoadModel()
        {
            var fs = WaitForFile(_filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
            if (fs == null)
                throw new IOException($"Model file {_filePath} still got locked after 5 seconds, fail to reload.");

            using (fs)
            {
                _model = _context.Model.Load(fs, out _);
            }
        }

        public void Dispose()
        {
            _watcher?.Dispose();
        }

        internal static class EventIds
        {
            public static readonly EventId FileReloadBegin = new EventId(100, "FileReloadBegin");
            public static readonly EventId FileReloadEnd = new EventId(101, "FileReloadEnd");
            public static readonly EventId FileReload = new EventId(102, "FileReload");
            public static readonly EventId FileReloadError = new EventId(103, nameof(FileReloadError));
        }

        private static class Logger
        {
            private static readonly Action<ILogger, string, Exception> _fileLoadBegin = LoggerMessage.Define<string>(
                LogLevel.Debug,
                EventIds.FileReloadBegin,
                "File reload for '{filePath}'");

            private static readonly Action<ILogger, string, double, Exception> _fileLoadEnd = LoggerMessage.Define<string, double>(
                LogLevel.Debug,
                EventIds.FileReloadEnd,
                "File reload for '{filePath}' completed after {ElapsedMilliseconds}ms");

            private static readonly Action<ILogger, string, double, Exception> _fileReloadError = LoggerMessage.Define<string, double>(
                LogLevel.Error,
                EventIds.FileReloadError,
                "File reload for '{filePath}' threw an unhandled exception after {ElapsedMilliseconds}ms");

            private static readonly Action<ILogger, string, double, Exception> _fileReLoad = LoggerMessage.Define<string, double>(
                LogLevel.Information,
                EventIds.FileReloadEnd,
                "Reloading file '{filePath}' completed after {ElapsedMilliseconds}ms");

            public static void FileReloadBegin(ILogger logger, string filePath)
            {
                _fileLoadBegin(logger, filePath, null);
            }

            public static void FileReloadEnd(ILogger logger, string filePath, TimeSpan duration)
            {
                _fileLoadEnd(logger, filePath, duration.TotalMilliseconds, null);
            }

            public static void FileReloadError(ILogger logger, string filePath, TimeSpan duration, Exception exception)
            {
                _fileReloadError(logger, filePath, duration.TotalMilliseconds, exception);
            }

            public static void ReloadingFile(ILogger logger, string filePath, TimeSpan duration)
            {
                _fileReLoad(logger, filePath, duration.TotalMilliseconds, null);
            }
        }

    }
}
