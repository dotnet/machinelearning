// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Runtime.Data.IO.Zlib
{
    public sealed class ZInflateStream : Stream
    {
        private readonly Stream _compressed;
        private readonly byte[] _buffer;

        private int _bufferUsed;
        private ZStream _zstrm;
        private bool _disposed;

        public override bool CanRead => true;

        public override bool CanSeek => false;

        public override bool CanWrite => false;

        public override long Length { get { throw Contracts.ExceptNotSupp(); } }

        public override long Position {
            get { throw Contracts.ExceptNotSupp(); }
            set { throw Contracts.ExceptNotSupp(); }
        }

        public ZInflateStream(Stream compressed, bool useZlibFormat = false)
        {
            Constants.RetCode ret;
            _compressed = compressed;
            _buffer = new byte[1 << 15];
            unsafe
            {
                fixed (ZStream* pZstream = &_zstrm)
                {
                    ret = Zlib.InflateInit2(pZstream, useZlibFormat ? Constants.MaxBufferSize : -Constants.MaxBufferSize);
                }
            }
            if (ret != Constants.RetCode.OK)
                throw Contracts.Except("Could not initialize zstream. Error code: {0}", ret);
            _bufferUsed = 0;
        }

        protected override void Dispose(bool disposing)
        {
            if (_disposed)
                return;
            _disposed = true;
            Constants.RetCode ret;
            unsafe
            {
                fixed (ZStream* pZstream = &_zstrm)
                {
                    ret = Zlib.inflateEnd(pZstream);
                }
            }
            base.Dispose(disposing);
            if (disposing)
            {
                GC.SuppressFinalize(this);
                if (ret != Constants.RetCode.OK)
                    throw Contracts.Except("Zlib inflateEnd failed with {0}", ret);
            }
        }

        ~ZInflateStream()
        {
            Dispose(false);
        }

        public override void Flush()
        {
        }

        public override int Read(byte[] buffer, int offset, int count)
        {
            Contracts.CheckValue(buffer, nameof(buffer));
            Contracts.CheckParamValue(offset >= 0, offset, nameof(offset), "Must be non-negative value");
            Contracts.CheckParamValue(offset < buffer.Length, offset, nameof(offset), "Must be greater than buffer length");
            Contracts.CheckParamValue(count >= 0, count, nameof(count), "Must be non-negative value");
            Contracts.CheckParamValue(count <= buffer.Length - offset, count, nameof(count),
                "Must or equal than difference between buffer length and offset");
            if (count == 0)
                return 0;
            unsafe
            {
                fixed (byte* pInput = &_buffer[0])
                fixed (byte* pOutput = &buffer[offset])
                {
                    return InternalRead(pInput, pOutput, count);
                }
            }
            throw Contracts.Except("Bad offset {0} and count {1} for length {2} buffer", offset, count, buffer.Length);
        }

        private unsafe int InternalRead(byte* pInput, byte* pOutput, int count)
        {
            Constants.RetCode ret;

            _zstrm.NextIn = pInput + _bufferUsed - _zstrm.AvailIn;
            _zstrm.NextOut = pOutput;
            _zstrm.AvailOut = (uint)count;
            do
            {
                if (_compressed != null && (_bufferUsed == 0 || _zstrm.AvailIn == 0))
                {
                    _bufferUsed = _compressed.Read(_buffer, 0, _buffer.Length);
                    _zstrm.AvailIn = (uint)_bufferUsed;
                    if (_bufferUsed == 0)
                        break;
                    _zstrm.NextIn = pInput;
                }
                else
                    _zstrm.NextIn = pInput + _bufferUsed - _zstrm.AvailIn;

                if (_zstrm.AvailIn == 0)
                    return 0;

                fixed (ZStream* pZstream = &_zstrm)
                {
                    ret = Zlib.inflate(pZstream, Constants.Flush.NoFlush);
                    if (!(ret == Constants.RetCode.StreamEnd || ret == Constants.RetCode.OK))
                        throw Contracts.Except($"{nameof(Zlib.inflate)} failed with {ret}");
                }
            } while (ret != Constants.RetCode.StreamEnd && _zstrm.AvailOut != 0);

            return count - (int)_zstrm.AvailOut;
        }

        public override long Seek(long offset, SeekOrigin origin)
        {
            throw Contracts.ExceptNotSupp();
        }

        public override void SetLength(long value)
        {
            throw Contracts.ExceptNotSupp();
        }

        public override void Write(byte[] buffer, int offset, int count)
        {
            throw Contracts.ExceptNotSupp();
        }
    }
}
