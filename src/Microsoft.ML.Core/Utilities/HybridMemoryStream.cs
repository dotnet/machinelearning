// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    using Conditional = System.Diagnostics.ConditionalAttribute;

    /// <summary>
    /// This is a read-write stream that, at or below a length threshold given in the constructor,
    /// works purely in memory, but if it ever grows larger than that point backs off to use the
    /// file system. This can be useful if we have intermediate operations that require streams.
    /// The temporary file will be destroyed if the object is properly disposed.
    /// </summary>
    public sealed class HybridMemoryStream : Stream
    {
        private MemoryStream _memStream;
        private Stream _overflowStream;
        private string _overflowPath;
        private readonly int _overflowBoundary;
        private const int _defaultMaxLen = 1 << 30;

        private bool _disposed;

        private Stream MyStream => _memStream ?? _overflowStream;

        private bool IsMemory => _memStream != null;

        public override long Position {
            get => MyStream.Position;
            set => Seek(value, SeekOrigin.Begin);
        }

        public override long Length => MyStream.Length;
        public override bool CanWrite => MyStream.CanWrite;
        public override bool CanSeek => MyStream.CanSeek;
        public override bool CanRead => MyStream.CanRead;

        /// <summary>
        /// Constructs an initially empty read-write stream. Once the number of
        /// bytes in the stream exceeds <paramref name="maxLen"/>,
        /// then we back off to disk.
        /// </summary>
        /// <param name="maxLen">The maximum length we will accomodate in memory</param>
        public HybridMemoryStream(int maxLen = _defaultMaxLen)
        {
            if (!(0 <= maxLen && maxLen <= Utils.ArrayMaxSize))
                throw Contracts.ExceptParam(nameof(maxLen), "must be in range [0,{0}]", Utils.ArrayMaxSize);
            _memStream = new MemoryStream();
            _overflowBoundary = maxLen;
            AssertInvariants();
        }

        /// <summary>
        /// A common usecase of the hybrid memory stream is to create a persistent
        /// readable (not necessarily writable) copy of a stream whose source is very
        /// transient and temporary. This utility method makes that creation of a copy
        /// somewhat easier.
        /// </summary>
        /// <param name="stream">A stream that can be opened</param>
        /// <param name="maxLen">The maximum length we will accomodate in memory</param>
        /// <returns>A readable copy of the data stream</returns>
        public static Stream CreateCache(Stream stream, int maxLen = _defaultMaxLen)
        {
            Contracts.CheckValue(stream, nameof(stream));
            Contracts.CheckParam(stream.CanRead, nameof(stream), "Cannot copy a stream we cannot read");
            if (!(0 <= maxLen && maxLen <= Utils.ArrayMaxSize))
                throw Contracts.ExceptParam(nameof(maxLen), "must be in range [0,{0}]", Utils.ArrayMaxSize);

            if (stream.CanSeek)
            {
                // If we can seek, then we can know the length ahead of time,
                // and return the less-overhead memory stream directly if appropriate.
                Contracts.CheckParam(stream.Position == 0, nameof(stream), "Should be at the head of the stream");
                long len = stream.Length;
                if (len <= maxLen)
                {
                    byte[] bytes = new byte[(int)len];
                    stream.ReadBlock(bytes, 0, bytes.Length);
                    return new MemoryStream(bytes, writable: false);
                }
            }
            var memStream = new HybridMemoryStream(maxLen);
            stream.CopyTo(memStream);
            memStream.Seek(0, SeekOrigin.Begin);
            return memStream;
        }

        [Conditional("DEBUG")]
        private void AssertInvariants()
        {
#if DEBUG
            if (_disposed)
            {
                Contracts.Assert(_memStream == null);
                Contracts.Assert(_overflowStream == null);
            }
            else
            {
                Contracts.Assert((_memStream == null) != (_overflowStream == null));
                Contracts.Assert(Length <= _overflowBoundary || _overflowStream != null);
            }
#endif
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing && !_disposed)
            {
                AssertInvariants();
                if (_memStream != null)
                {
                    _memStream.Dispose();
                    _memStream = null;
                }

                if (_overflowStream != null)
                {
                    var overflow = _overflowStream;
                    _overflowStream = null;
                    overflow.Dispose();
                    Contracts.AssertValue(_overflowPath);
                    _overflowPath = null;
                }
                _disposed = true;
                AssertInvariants();
                base.Dispose(disposing);
            }
        }

        public override void Close()
        {
            AssertInvariants();
            // The base Stream class Close will call Dispose(bool).
            base.Close();
        }

        public override void Flush()
        {
            AssertInvariants();
            MyStream?.Flush();
            AssertInvariants();
        }

        /// <summary>
        /// Creates the overflow stream if it does not exist, switching all content over to
        /// the file-based stream and disposing of the memory stream. If the overflow stream
        /// already exists, this method has no effect.
        /// </summary>
        private void EnsureOverflow()
        {
            AssertInvariants();
            Contracts.Check(!_disposed, "Stream already disposed");
            if (_overflowStream != null)
                return;
            Contracts.Assert(_memStream != null);
            // MemoryStreams return that they cannot read when they are closed.
            // The only way that stream would be closed is if we ourselves have
            // been closed.
            Contracts.Check(_memStream.CanRead, "attempt to perform operation on closed stream");

            Contracts.Assert(_overflowPath == null);
            _overflowPath = Path.GetTempFileName();
            _overflowStream = new FileStream(_overflowPath, FileMode.Open, FileAccess.ReadWrite,
                FileShare.None, bufferSize: 4096, FileOptions.DeleteOnClose);

            // The documentation is not clear on this point, but the source code for
            // memory stream makes clear that this buffer is exposable for a memory
            // stream constructed as we have.
            long pos = _memStream.Position;
            ArraySegment<byte> buffer;
            bool tmp = _memStream.TryGetBuffer(out buffer);
            Contracts.Assert(tmp, "TryGetBuffer failed in HybridMemoryStream");
            _overflowStream.Write(buffer.Array, buffer.Offset, buffer.Count);
            _memStream.Dispose();
            _memStream = null;

            _overflowStream.Seek(pos, SeekOrigin.Begin);

            AssertInvariants();
        }

        public override void SetLength(long value)
        {
            Contracts.CheckParam(0 <= value, nameof(value), "cannot be negative");
            AssertInvariants();
            Contracts.Check(!_disposed, "Stream already disposed");
            if (value > _overflowBoundary)
                EnsureOverflow();
            MyStream.SetLength(value);
            AssertInvariants();
        }

        public override long Seek(long offset, SeekOrigin origin)
        {
            AssertInvariants();
            Contracts.Check(!_disposed, "Stream already disposed");
            return MyStream.Seek(offset, origin);
        }

        public override int Read(byte[] buffer, int offset, int count)
        {
            AssertInvariants();
            Contracts.Check(!_disposed, "Stream already disposed");
            return MyStream.Read(buffer, offset, count);
        }

        public override void Write(byte[] buffer, int offset, int count)
        {
            AssertInvariants();
            Contracts.Check(!_disposed, "Stream already disposed");
            Contracts.CheckValue(buffer, nameof(buffer));
            Contracts.CheckParam(0 <= offset && offset <= buffer.Length, nameof(offset));
            Contracts.CheckParam(0 <= count && count <= buffer.Length - offset, nameof(count));
            if (IsMemory && _memStream.Position > _overflowBoundary - count)
                EnsureOverflow();
            MyStream.Write(buffer, offset, count);
            AssertInvariants();
        }

        public override int ReadByte()
        {
            AssertInvariants();
            Contracts.Check(!_disposed, "Stream already disposed");
            return MyStream.ReadByte();
        }

        public override void WriteByte(byte value)
        {
            AssertInvariants();
            Contracts.Check(!_disposed, "Stream already disposed");
            if (IsMemory && _memStream.Position >= _overflowBoundary)
                EnsureOverflow();
            MyStream.WriteByte(value);
            AssertInvariants();
        }
    }
}
