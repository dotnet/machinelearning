// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;

namespace Microsoft.ML.Runtime.Data.IO.Zlib
{
    public sealed class ZDeflateStream : Stream
    {
        private readonly Stream _compressed;
        private readonly byte[] _buffer;

        private ZStream _zstrm;
        private bool _disposed;

        public ZDeflateStream(Stream compressed, Constants.Level level = Constants.Level.BestCompression,
            Constants.Strategy strategy = Constants.Strategy.DefaultStrategy, int memLevel = 9,
            bool useZlibFormat = false, int windowBits = Constants.MaxBufferSize)
        {
            Constants.RetCode ret;
            _compressed = compressed;
            _buffer = new byte[1 << 15];
            unsafe
            {
                fixed (ZStream* pZstream = &_zstrm)
                {
                    ret = Zlib.DeflateInit2(pZstream, (int)level, 8, useZlibFormat ? windowBits : -windowBits, memLevel, strategy);
                }
            }
            if (ret != Constants.RetCode.OK)
                throw Contracts.Except("Could not initialize zstream. Error code: {0}", ret);
            _zstrm.AvailOut = (uint)_buffer.Length;
        }

        protected override void Dispose(bool disposing)
        {
            if (_disposed)
                return;
            _disposed = true;

            Constants.RetCode disposeRet = Constants.RetCode.StreamEnd;
            if (disposing)
            {
                unsafe
                {
                    fixed (byte* pOutput = _buffer)
                    fixed (ZStream* pZstream = &_zstrm)
                    {
                        pZstream->AvailIn = 0;
                        pZstream->NextIn = null;
                        pZstream->NextOut = pOutput + BufferUsed;
                        do
                        {
                            RefreshOutput(pOutput);
                            disposeRet = Zlib.deflate(pZstream, Constants.Flush.Finish);
                        } while (disposeRet == Constants.RetCode.OK);
                        if (disposeRet == Constants.RetCode.StreamEnd)
                        {
                            Flush();
                            _compressed.Flush();
                        }
                    }
                }
            }
            Constants.RetCode ret;
            unsafe
            {
                fixed (ZStream* pZstream = &_zstrm)
                {
                    ret = Zlib.deflateEnd(pZstream);
                }
            }
            base.Dispose(disposing);
            if (disposing)
            {
                GC.SuppressFinalize(this);
                if (disposeRet != Constants.RetCode.StreamEnd)
                    throw Contracts.Except("Zlib deflate failed with {0}", disposeRet);
                if (ret != Constants.RetCode.OK)
                    throw Contracts.Except("Zlib deflateEnd failed with {0}", ret);
            }
        }

        ~ZDeflateStream()
        {
            Dispose(false);
        }

        public override bool CanRead
        {
            get { return false; }
        }

        public override bool CanSeek
        {
            get { return false; }
        }

        public override bool CanWrite
        {
            get { return true; }
        }

        private int BufferUsed
        {
            get
            {
                Contracts.Assert(0 <= _zstrm.AvailOut);
                Contracts.Assert(_zstrm.AvailOut <= _buffer.Length);
                return _buffer.Length - (int)_zstrm.AvailOut;
            }
        }

        public override void Flush()
        {
            if (BufferUsed <= 0)
                return;
            _compressed.Write(_buffer, 0, BufferUsed);
            _zstrm.AvailOut = (uint)_buffer.Length;
        }

        public override long Length
        {
            get { throw Contracts.ExceptNotSupp(); }
        }

        public override long Position
        {
            get
            {
                throw Contracts.ExceptNotSupp();
            }
            set
            {
                throw Contracts.ExceptNotSupp();
            }
        }

        public override int Read(byte[] buffer, int offset, int count)
        {
            throw Contracts.ExceptNotImpl();
        }

        public override long Seek(long offset, SeekOrigin origin)
        {
            throw Contracts.ExceptNotImpl();
        }

        public override void SetLength(long value)
        {
            throw Contracts.ExceptNotImpl();
        }

        public override void Write(byte[] buffer, int offset, int count)
        {
            Contracts.CheckValue(buffer, nameof(buffer));
            Contracts.CheckParamValue(offset >= 0, offset, nameof(offset), "offset can't be negative value");
            Contracts.CheckParamValue(offset < buffer.Length, offset, nameof(offset), "offset can't be greater than buffer length");
            Contracts.CheckParamValue(count >= 0, count, nameof(count), "count can't be negative value");
            Contracts.CheckParamValue(count <= buffer.Length - offset, count, nameof(count),
                "count should be less or equal than difference between buffer length and offset");

            int length = buffer.Length;
            if (count == 0)
                return;
            unsafe
            {
                fixed (byte* pOutput = &_buffer[0])
                fixed (byte* pInput = &buffer[offset])
                {
                    RawWrite(pInput, pOutput, count);
                }
            }
        }

        /// <summary>
        /// Check zlib internal buffer and if it's full flush its results to compressed stream.
        /// </summary>
        /// <param name="pOutput">link internal buffer</param>
        private unsafe void RefreshOutput(byte* pOutput)
        {
#if DEBUG
            fixed (byte* bufferPointer = &_buffer[0])
            {
                Contracts.Assert(pOutput == bufferPointer);
            }
#endif
            if (_zstrm.AvailOut != 0)
                return;
            Flush();
            _zstrm.NextOut = pOutput;
        }

        private unsafe void RawWrite(byte* buffer, byte* pOutput, int count)
        {
#if DEBUG
            fixed (byte* bufferPointer = &_buffer[0])
            {
                Contracts.Assert(pOutput == bufferPointer);
            }
#endif
            Constants.RetCode ret;
            _zstrm.AvailIn = (uint)count;
            _zstrm.NextIn = buffer;
            _zstrm.NextOut = pOutput + BufferUsed;
            do
            {
                RefreshOutput(pOutput);
                fixed (ZStream* pZstream = &_zstrm)
                {
                    ret = Zlib.deflate(pZstream, Constants.Flush.NoFlush);
                }
                if (ret != Constants.RetCode.OK)
                {
                    throw Contracts.Except("Zlib.deflate failed with {0}", ret);
                }
            } while (_zstrm.AvailIn > 0);
            _zstrm.NextIn = null;
        }
    }
}
