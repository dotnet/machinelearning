// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;

namespace Microsoft.ML.Runtime.FastTree.Internal
{
    internal struct BufferBlock
    {
        public byte[] Buffer;
        public int Offset;
        public int Length;
        public int Size;

        public static IEnumerable<BufferBlock> BlockLoader(Func<byte[], int, int, int> loader, int loadSize)
        {
            BufferBlock block = new BufferBlock();
            block.Buffer = new byte[loadSize * 2];
            block.Size = -1;
            block.Offset = 0;
            block.Length = 0;

            while (true)
            {
                // read message size
                if (block.Size < 0 && block.Length > sizeof(int))
                {
                    int p = block.Offset;
                    block.Size = block.Buffer.ToInt(ref p);
                }

                // check message completed
                if (block.Size > 0 && block.Size <= block.Length)
                {
                    yield return block;
                    block.Offset += block.Size;
                    block.Length -= block.Size;
                    block.Size = -1;
                }
                else
                {
                    // not enough space, shift to begin
                    if (block.Offset > 0 && block.Offset + block.Length + loadSize > block.Buffer.Length)
                    {
                        Array.Copy(block.Buffer, block.Offset, block.Buffer, 0, block.Length);
                        block.Offset = 0;
                    }

                    // still not enough space, resize buffer
                    if (block.Offset + block.Length + loadSize > block.Buffer.Length)
                    {
                        Array.Resize(ref block.Buffer, (block.Offset + block.Length + loadSize) * 2);
                    }

                    int receiveSize = loader(block.Buffer, block.Offset + block.Length, loadSize);

                    if (receiveSize <= 0)
                        break;

                    block.Length += receiveSize;
                }
            }
        }

        public static IEnumerable<BufferBlock> BlockSplitter(byte[] buffer, int offset)
        {
            BufferBlock block = new BufferBlock();
            block.Buffer = buffer;
            block.Size = -1;
            block.Offset = offset;
            block.Length = buffer.Length - offset;

            while (true)
            {
                // read message size
                if (block.Size < 0 && block.Length > sizeof(int))
                {
                    int p = block.Offset;
                    block.Size = block.Buffer.ToInt(ref p);
                }

                // check message completed
                if (block.Size > 0 && block.Size <= block.Length)
                {
                    yield return block;
                    block.Offset += block.Size;
                    block.Length -= block.Size;
                    block.Size = -1;
                }
                else
                {
                    break;
                }
            }
        }
    }

    internal static class CompressUtil
    {
        public static byte[] Compress(byte[] uncompressed, int offset, int length)
        {
            using (var ms = new MemoryStream())
            // Stream compressedStream = new ICSharpCode.SharpZipLib.BZip2.BZip2OutputStream(ms);
            using (var compressedStream = new GZipStream(ms, CompressionMode.Compress))
            {
                compressedStream.Write(uncompressed, offset, length);
                return ms.ToArray();
            }
        }

        public static IEnumerable<BufferBlock> DeCompress(byte[] compressed)
        {
            using (var ms = new MemoryStream(compressed))
            // Stream decompressedStream = new ICSharpCode.SharpZipLib.BZip2.BZip2InputStream(ms);
            using (var decompressedStream = new GZipStream(ms, CompressionMode.Decompress))
            {
                int readBlockSize = 1024 * 1024;
                foreach (BufferBlock block in BufferBlock.BlockLoader(decompressedStream.Read, readBlockSize))
                {
                    yield return block;
                }
            }
        }
    }

}
