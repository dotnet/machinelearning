// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Text;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    /// <summary>
    /// A readable <see cref="Stream"/> that is backed by a <see cref="TextReader"/>.
    /// Because text readers strip line breaks from the end of their lines, this
    /// compensates by inserting <c>\n</c> line feed characters at the end of every
    /// input line, including the last one.
    /// </summary>
    public sealed class TextReaderStream : Stream
    {
        private readonly TextReader _baseReader;
        private readonly Encoding _encoding;
        // Position in the stream.
        private long _position;

        private const int _charBlockSize = 1024;

        private string _line;
        private int _lineCur;

        private byte[] _buff;
        private int _buffCur;
        private int _buffLim;
        private bool _eof;

        public override bool CanRead => true;

        public override bool CanSeek => false;

        public override bool CanWrite => false;

        public override long Length
            => throw Contracts.ExceptNotSupp("Stream cannot determine length.");

        public override long Position
        {
            get => _position;
            set
            {
                if (value != Position)
                {
                    throw Contracts.ExceptNotSupp("Stream cannot seek.");
                }
            }
        }

        /// <summary>
        /// Create a stream wrapping the given text reader, using the <see cref="Encoding.UTF8"/>
        /// encoding.
        /// </summary>
        /// <param name="baseReader">the reader to wrap</param>
        public TextReaderStream(TextReader baseReader)
            : this(baseReader, Encoding.UTF8)
        {
        }

        /// <summary>
        /// Create a stream wrapping the given text reader, using the given encoding. The class
        /// assumes that the encoding is distributive, that is, the concatenation of the byte
        /// encodings of different strings, is a valid byte encoding of the single encoding of
        /// the concatenation of those strings. (I believe all standard encodings obey this
        /// property.)
        /// </summary>
        /// <param name="baseReader">The reader to wrap.</param>
        /// <param name="encoding">The encoding to use.</param>
        public TextReaderStream(TextReader baseReader, Encoding encoding)
        {
            _baseReader = baseReader;
            _encoding = encoding;

            _buff = new byte[Math.Max(_encoding.GetByteCount("\n"), _encoding.GetMaxByteCount(_charBlockSize))];
        }

        public override void Close()
        {
            _baseReader.Close();
            _eof = true;
        }

        protected override void Dispose(bool disposing)
        {
            _baseReader.Dispose();
            base.Dispose(disposing);
        }

        public override void Flush()
        {
            return;
        }

        /// <summary>
        /// A helper method that will either ensure that <see cref="_buffCur"/> is less
        /// than <see cref="_buffLim"/> (so there are at least some characters), or that
        /// <see cref="_eof"/> is set.
        /// </summary>
        private void EnsureBytes()
        {
            // If we're at the end of the file, or if there are still bytes available in
            // the buffer, just return.
            if (_eof || _buffCur < _buffLim)
                return;
            Contracts.Assert(_buffCur == _buffLim);
            // There are no bytes available in the buffer, and we have not reached the
            // end of the file. If we don't have a line pending, get one.
            if (_line == null)
            {
                Contracts.Assert(_lineCur == 0);
                _line = _baseReader.ReadLine();
                if (_line == null)
                {
                    _eof = true;
                    return;
                }
            }
            Contracts.AssertValue(_line);
            Contracts.Assert(_lineCur <= _line.Length);
            _buffCur = 0;
            if (_lineCur == _line.Length)
            {
                // There are no more chars, and we are at the end of a string. Encode the
                // linefeed, then set the line to null so we know, when we run out of bytes
                // on the linefeed character, that we have no line "pending."
                _buffLim = _encoding.GetBytes("\n", 0, 1, _buff, 0);
                // I am assuming it's impossible for an encoding to encode "\n" as 0 bytes. :)
                Contracts.Assert(0 < _buffLim && _buffLim <= _buff.Length);
                _line = null;
                _lineCur = 0;
                return;
            }
            // There are still more characters in _line to encode.
            int charCount = Math.Min(_line.Length - _lineCur, _charBlockSize);
            Contracts.Assert(charCount > 0);
            _buffLim = _encoding.GetBytes(_line, _lineCur, charCount, _buff, 0);
            Contracts.Assert(0 < _buffLim && _buffLim <= _buff.Length);
            _lineCur += charCount;
            Contracts.Assert(_lineCur <= _line.Length);
        }

        public override int Read(byte[] buffer, int offset, int count)
        {
            Contracts.CheckValue(buffer, nameof(buffer));
            Contracts.CheckParam(0 <= offset && offset <= buffer.Length, nameof(offset), "invalid for this sized array");
            Contracts.CheckParam(0 <= count && count <= buffer.Length - offset, nameof(count), "invalid for this sized array");
            int readCount = 0;
            while (readCount < count)
            {
                EnsureBytes();
                if (_eof)
                    break;
                Contracts.Assert(_buffCur < _buffLim);
                int toCopy = Math.Min(count - readCount, _buffLim - _buffCur);
                Contracts.Assert(toCopy > 0);
                Buffer.BlockCopy(_buff, _buffCur, buffer, offset, toCopy);
                offset += toCopy;
                readCount += toCopy;
                _buffCur += toCopy;
            }
            _position += readCount;
            return readCount;
        }

        public override int ReadByte()
        {
            EnsureBytes();
            Contracts.Assert(_eof || _buffCur < _buffLim);
            return _eof ? -1 : (int)_buff[_buffCur++];
        }

        public override long Seek(long offset, SeekOrigin origin)
            => throw Contracts.ExceptNotSupp("Stream cannot seek.");

        public override void Write(byte[] buffer, int offset, int count)
            => throw Contracts.ExceptNotSupp("Stream is not writable.");

        public override void SetLength(long value)
            => throw Contracts.ExceptNotSupp("Stream is not writable.");
    }
}