// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using System.IO.Compression;

namespace Microsoft.ML.Trainers.FastTree.Internal
{
    public static class StreamExtensions
    {
        /// <summary>
        /// A stream class that suppresses the dispose signal
        /// </summary>
        private sealed class UndisposableStream : Stream
        {
            private readonly Stream _inner;

            public override bool CanRead => _inner.CanRead;
            public override bool CanSeek => _inner.CanSeek;
            public override bool CanWrite => _inner.CanWrite;

            public override long Position
            {
                get { return _inner.Position; }
                set { _inner.Position = value; }
            }

            public UndisposableStream(Stream inner)
            {
                _inner = inner;
            }

            public override void Flush()
            {
                _inner.Flush();
            }

            public override long Length => _inner.Length;

            public override int Read(byte[] buffer, int offset, int count)
            {
                return _inner.Read(buffer, offset, count);
            }

            public override long Seek(long offset, SeekOrigin origin)
            {
                return _inner.Seek(offset, origin);
            }

            public override void SetLength(long value)
            {
                _inner.SetLength(value);
            }

            public override void Write(byte[] buffer, int offset, int count)
            {
                _inner.Write(buffer, offset, count);
            }
        }

        /// <summary>
        /// Reads a compressed array of byte from the stream (written by WriteCompressed)
        /// </summary>
        /// <param name="stream">The stream to read from</param>
        /// <returns>The decompressed bytes</returns>
        public static byte[] ReadCompressed(this Stream stream)
        {
            BinaryReader reader = new BinaryReader(stream);
            int len = reader.ReadInt32();
            byte[] array = new byte[len];
            using (var ds = new DeflateStream(stream, CompressionMode.Decompress))
                ds.Read(array, 0, len);
            return array;
        }

        /// <summary>
        /// Writes an array of bytes to the stream with compression
        /// </summary>
        /// <param name="stream">Stream to write to</param>
        /// <param name="array">Array to write</param>
        /// <param name="offset">The byte offset into the array to write</param>
        /// <param name="count">The number of bytes from the array to write</param>
        public static void WriteCompressed(this Stream stream, byte[] array, int offset, int count)
        {
            // we don't want the DeflateStream to close the input stream
            // but we have to dispose the DeflateStream in order for it to flush (according to the documentation)
            // so wrap the input stream in an UndisposableStream
            stream = new UndisposableStream(stream);

            using (BinaryWriter writer = new BinaryWriter(stream))
                writer.Write(count);
            using (DeflateStream ds = new DeflateStream(stream, CompressionMode.Compress))
                ds.Write(array, offset, count);
        }

        /// <summary>
        /// Writes an array of bytes to the stream with compression
        /// </summary>
        /// <param name="stream">Stream to write to</param>
        /// <param name="array">Array to write</param>
        public static void WriteCompressed(this Stream stream, byte[] array)
        {
            stream.WriteCompressed(array, 0, array.Length);
        }
    }
}
