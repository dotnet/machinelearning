// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.IO.Compression;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Data.IO.Zlib;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data.IO
{
    /// <summary>
    /// A code indicating the kind of compression. It is supposed that each kind of compression is totally
    /// sufficient to describe how a given stream should be decompressed.
    /// </summary>
    public enum CompressionKind : byte
    {
        None = 0,    // No compression at all.
        Deflate = 1, // DEFLATE algorithm as in zlib's headerless/tailless compression.
        Default = Deflate
    }

    public static class CompressionCodecExtension
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

    /// <summary>
    /// A loadable class to parameterize compression.
    /// </summary>
    public abstract class Compression
    {
        public abstract CompressionKind Kind { get; }

        /// <summary>
        /// Generate an appropriate wrapping compressing stream for the codec. This
        /// stream will be closable and disposable, without closing or disposing of
        /// the passed in stream. The scheme for compression is parameterized by the
        /// <see cref="Compression"/> instance.
        /// </summary>
        /// <param name="stream">The stream to which compressed data will be written</param>
        /// <returns>A stream to which the user can write uncompressed data</returns>
        public virtual Stream Open(Stream stream)
        {
            return Kind.CompressStream(stream);
        }

        // Named with "Impl" suffix since otherwise it was difficult to disambiguate
        // with other identifiers.
        public sealed class NoneImpl : Compression
        {
            public override CompressionKind Kind { get { return CompressionKind.None; } }
        }

        public sealed class ZlibImpl : Compression
        {
            public abstract class ArgumentsBase
            {
                [Argument(ArgumentType.AtMostOnce, HelpText = "Level of compression from 0 to 9", ShortName = "c")]

                public int? CompressionLevel = 9;

                [Argument(ArgumentType.AtMostOnce, HelpText = "Window bits from 8 to 15, higher values enable more useful run length encodings", ShortName = "w")]
                public int WindowBits = 15;

                [Argument(ArgumentType.AtMostOnce, HelpText = "Level of memory from 1 to 9, with higher values using more memory but enabling better, faster compression", ShortName = "m")]
                public int MemoryLevel = 9;

                [Argument(ArgumentType.AtMostOnce, HelpText = "Compression strategy to employ", ShortName = "s")]
                public Constants.Strategy Strategy = Constants.Strategy.DefaultStrategy;
            }

            public sealed class DeflateArguments : ArgumentsBase
            {
            }

            public sealed class ZlibArguments : ArgumentsBase
            {
            }

            public override CompressionKind Kind
            {
                get { return CompressionKind.Deflate; }
            }

            private readonly int _windowBits;
            private readonly Constants.Level _level;
            private readonly bool _isDeflate;
            private readonly int _memoryLevel;
            private readonly Constants.Strategy _strategy;

            private ZlibImpl(ArgumentsBase args, bool isDeflate)
            {
                Contracts.CheckUserArg(args.CompressionLevel == null ||
                          (0 <= args.CompressionLevel && args.CompressionLevel <= 9),
                          nameof(args.CompressionLevel), "Must be in range 0 to 9 or null");
                Contracts.CheckUserArg(8 <= args.WindowBits && args.WindowBits <= 15, nameof(args.WindowBits), "Must be in range 8 to 15");
                Contracts.CheckUserArg(1 <= args.MemoryLevel && args.MemoryLevel <= 9, nameof(args.MemoryLevel), "Must be in range 1 to 9");
                Contracts.CheckUserArg(Enum.IsDefined(typeof(Constants.Strategy), args.Strategy), nameof(args.Strategy), "Value was not defined");

                if (args.CompressionLevel == null)
                    _level = Constants.Level.DefaultCompression;
                else
                    _level = (Constants.Level)args.CompressionLevel;
                Contracts.Assert(Enum.IsDefined(typeof(Constants.Level), _level));
                _windowBits = args.WindowBits;
                _isDeflate = isDeflate;
                _memoryLevel = args.MemoryLevel;
                _strategy = args.Strategy;
            }

            public ZlibImpl(DeflateArguments args)
                : this(args, isDeflate: true)
            {
                Contracts.Assert(Kind == CompressionKind.Deflate);
            }

            public ZlibImpl(ZlibArguments args)
                : this(args, isDeflate: false)
            {
               // Contracts.Assert(Kind == CompressionKind.Zlib);
            }

            public override Stream Open(Stream stream)
            {
                return new ZDeflateStream(stream, _level, _strategy, _memoryLevel, !_isDeflate, _windowBits);
            }
        }
    }
}
