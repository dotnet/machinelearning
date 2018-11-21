// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime
{
    /// <summary>
    /// A file handle.
    /// </summary>
    public interface IFileHandle : IDisposable
    {
        /// <summary>
        /// Returns whether CreateWriteStream is expected to succeed. Typically, once
        /// CreateWriteStream has been called once, this will forever more return false.
        /// </summary>
        bool CanWrite { get; }

        /// <summary>
        /// Returns whether OpenReadStream is expected to succeed.
        /// </summary>
        bool CanRead { get; }

        /// <summary>
        /// Create a writable stream for this file handle.
        /// </summary>
        Stream CreateWriteStream();

        /// <summary>
        /// Open a readable stream for this file handle.
        /// </summary>
        Stream OpenReadStream();
    }

    /// <summary>
    /// A simple disk-based file handle.
    /// </summary>
    public sealed class SimpleFileHandle : IFileHandle
    {
        private readonly string _fullPath;

        // Exception context.
        private readonly IExceptionContext _ectx;

        private readonly object _lock;

        // Whether to delete the file when this is disposed.
        private readonly bool _autoDelete;

        // Whether this file has contents. This is false if the file needs CreateWriteStream to be
        // called (before OpenReadStream can be called).
        private bool _wrote;
        // If non-null, the active write stream. This should be disposed before the first OpenReadStream call.
        private Stream _streamWrite;

        // This contains the potentially active read streams. This is set to null once this file
        // handle has been disposed.
        private List<Stream> _streams;

        private bool IsDisposed { get { return _streams == null; } }

        public SimpleFileHandle(IExceptionContext ectx, string path, bool needsWrite, bool autoDelete)
        {
            Contracts.CheckValue(ectx, nameof(ectx));
            ectx.CheckNonEmpty(path, nameof(path));

            _ectx = ectx;
            _fullPath = Path.GetFullPath(path);

            _autoDelete = autoDelete;

            // The file has already been written to iff needsWrite is false.
            _wrote = !needsWrite;

            // REVIEW: Should this do some basic validation? Eg, for output files, ensure that
            // the directory exists (and perhaps even create an empty file); for input files, ensure
            // that the file exists (and perhaps even attempt to open it).

            _lock = new object();
            _streams = new List<Stream>();
        }

        public bool CanWrite
        {
            get { return !_wrote && !IsDisposed; }
        }

        public bool CanRead
        {
            get { return _wrote && !IsDisposed; }
        }

        public void Dispose()
        {
            lock (_lock)
            {
                if (IsDisposed)
                    return;

                Contracts.Assert(_streams != null);

                // REVIEW: Is it safe to dispose these streams? What if they are
                // being used on other threads? Does that matter?
                if (_streamWrite != null)
                {
                    try
                    {
                        _streamWrite.CloseEx();
                        _streamWrite.Dispose();
                    }
                    catch
                    {
                        // REVIEW: What should we do here?
                        Contracts.Assert(false, "Closing a SimpleFileHandle write stream failed!");
                    }
                    _streamWrite = null;
                }

                foreach (var stream in _streams)
                {
                    try
                    {
                        stream.CloseEx();
                        stream.Dispose();
                    }
                    catch
                    {
                        // REVIEW: What should we do here?
                        Contracts.Assert(false, "Closing a SimpleFileHandle read stream failed!");
                    }
                }

                _streams = null;
                Contracts.Assert(IsDisposed);

                if (_autoDelete)
                {
                    try
                    {
                        // Finally, delete the file.
                        File.Delete(_fullPath);
                    }
                    catch
                    {
                        // REVIEW: What should we do here?
                        Contracts.Assert(false, "Deleting a SimpleFileHandle physical file failed!");
                    }
                }
            }
        }

        private void CheckNotDisposed()
        {
            if (IsDisposed)
                throw _ectx.Except("SimpleFileHandle has already been disposed");
        }

        public Stream CreateWriteStream()
        {
            lock (_lock)
            {
                CheckNotDisposed();

                if (_wrote)
                    throw _ectx.Except("CreateWriteStream called multiple times on SimpleFileHandle");

                Contracts.Assert(_streamWrite == null);
                _streamWrite = new FileStream(_fullPath, FileMode.Create, FileAccess.Write);
                _wrote = true;
                return _streamWrite;
            }
        }

        public Stream OpenReadStream()
        {
            lock (_lock)
            {
                CheckNotDisposed();

                if (!_wrote)
                    throw _ectx.Except("SimpleFileHandle hasn't been written yet");

                if (_streamWrite != null)
                {
                    if (_streamWrite.CanWrite)
                        throw _ectx.Except("Write stream for SimpleFileHandle hasn't been disposed");
                    _streamWrite = null;
                }

                // Drop read streams that have already been disposed.
                _streams.RemoveAll(s => !s.CanRead);

                var stream = new FileStream(_fullPath, FileMode.Open, FileAccess.Read);
                _streams.Add(stream);
                return stream;
            }
        }
    }
}
