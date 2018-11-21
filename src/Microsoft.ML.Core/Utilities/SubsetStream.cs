// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    /// <summary>
    /// Returns a "view" stream, which appears to be a possibly truncated
    /// version of the stream. Reads on this containing stream will also
    /// advance the wrapped stream. If truncated, reads will not progress
    /// beyond the indicated length, and writes will fail beyond the
    /// indicated length. This stream supports seeking (and associated
    /// operations) if the underlying stream supports seeking, where it is
    /// supposed that the returned <c>SubsetStream</c> instance has position
    /// 0 during creation, corresponding to whatever the position of the
    /// enclosed stream was during creation, so this stream will act as an
    /// "offset" version of the enclosed stream. As this is intended to
    /// operate over a subset of a stream, during closing or disposal of the
    /// subset stream, the underlying stream will always remain open and
    /// undisposed.
    /// </summary>
    public sealed class SubsetStream : Stream
    {
        private readonly Stream _stream;
        // The position of the stream.
        private readonly long _offset;
        // Negative value if unbounded.
        private long _remaining;

        /// <summary>
        /// Construct the view stream.
        /// </summary>
        /// <param name="stream">The underlying stream</param>
        /// <param name="length">The maximum length this containing
        /// stream should appear to have, or null if unbounded</param>
        public SubsetStream(Stream stream, long? length = null)
        {
            Contracts.AssertValue(stream);
            Contracts.Assert(!length.HasValue || length >= 0);
            _stream = stream;
            _remaining = length.GetValueOrDefault(-1);

            try
            {
                _offset = stream.Position;
            }
            catch (NotSupportedException)
            {
                // Doesn't matter in this case anyway. Any operations depending
                // on _offset should fail, assuming this is a "proper" implementation
                // of stream.
                _offset = -1;
            }
        }

        public override int Read(byte[] buffer, int offset, int count)
        {
            if (_remaining < 0)
                return _stream.Read(buffer, offset, count);
            if (count > _remaining)
                count = (int)_remaining;
            int retval = _stream.Read(buffer, offset, count);
            _remaining -= retval;
            return retval;
        }

        public override void SetLength(long value)
        {
            _stream.SetLength(value + _offset);
        }

        public override long Seek(long offset, SeekOrigin origin)
        {
            long oldPosition = Position;
            long newPosition;
            if (origin == SeekOrigin.End)
                newPosition = _stream.Seek(offset + Length + _offset, SeekOrigin.Begin) - _offset;
            else
                newPosition = _stream.Seek(offset + _offset, origin) - _offset;
            if (_remaining >= 0)
                _remaining -= newPosition - oldPosition;
            return newPosition;
        }

        public override bool CanRead { get { return _stream.CanRead; } }
        public override bool CanWrite { get { return _stream.CanWrite; } }
        public override bool CanTimeout { get { return _stream.CanTimeout; } }
        public override bool CanSeek { get { return _stream.CanSeek; } }

        public override long Position
        {
            get { return _stream.Position - _offset; }
            set { Seek(value, SeekOrigin.Begin); }
        }

        public override long Length
        {
            get
            {
                // This may fail with a not supported operation, due to
                // the underlying stream not supporting these operations.
                // But that is fine as this is the expected behavior for
                // this code as well.
                if (_remaining < 0)
                    return _stream.Length - _offset;
                return Position + _remaining;
            }
        }

        public override void Flush()
        {
            _stream.Flush();
        }

        public override void Write(byte[] buffer, int offset, int count)
        {
            if (_remaining >= 0 && count > _remaining)
                throw Contracts.Except("cannot write {0} bytes to stream bounded at {1} bytes", count, _remaining);
            _stream.Write(buffer, offset, count);
            if (_remaining >= 0)
                _remaining -= count;
        }
    }
}
