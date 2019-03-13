// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using System.IO.Compression;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data.IO
{
    /// <summary>
    /// A code indicating the kind of compression. It is supposed that each kind of compression is totally
    /// sufficient to describe how a given stream should be decompressed.
    /// </summary>
    internal enum CompressionKind : byte
    {
        None = 0,    // No compression at all.
        Deflate = 1, // DEFLATE algorithm as in zlib's headerless/tailless compression.
        Default = Deflate
    }

    internal static class CompressionCodecExtension
    {
        /// <summary>
        /// Generate an appropriate wrapping compressing stream for the codec. This
        /// stream will be closable and disposable, without closing or disposing of
        /// the passed in stream. The scheme for compression is not in any way
        /// parameterizable.
        /// </summary>
        /// <param name="compression">The compression codec</param>
        /// <param name="stream">The stream to which compressed data will be written</param>
        /// <returns>A stream to which the user can write uncompressed data</returns>
        public static Stream CompressStream(this CompressionKind compression, Stream stream)
        {
            switch (compression)
            {
                case CompressionKind.None:
                    return new SubsetStream(stream);
                case CompressionKind.Deflate:
                    return new DeflateStream(stream, CompressionMode.Compress, true);
                default:
                    throw Contracts.Except("unrecognized compression codec {0}", compression);
            }
        }

        /// <summary>
        /// Generate an appropriate wrapping decompressing stream for the codec.
        /// </summary>
        /// <param name="compression">The compression codec</param>
        /// <param name="stream">The stream from which compressed data will be written</param>
        /// <returns>A stream from which the user can read uncompressed data</returns>
        public static Stream DecompressStream(this CompressionKind compression, Stream stream)
        {
            switch (compression)
            {
                case CompressionKind.None:
                    return new SubsetStream(stream);
                case CompressionKind.Deflate:
                    return new DeflateStream(stream, CompressionMode.Decompress, true);
                default:
                    throw Contracts.Except("unrecognized compression codec {0}", compression);
            }
        }
    }
}
