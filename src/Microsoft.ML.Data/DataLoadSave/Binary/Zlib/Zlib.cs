// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.InteropServices;
using System.Security;

namespace Microsoft.ML.Runtime.Data.IO.Zlib
{
    internal static class Zlib
    {
        public const string DllPath = "zlib.dll";

        [DllImport(DllPath), SuppressUnmanagedCodeSecurity]
        private static unsafe extern Constants.RetCode deflateInit2_(ZStream* strm, int level, int method, int windowBits,
            int memLevel, Constants.Strategy strategy, byte* version, int streamSize);

        [DllImport(DllPath), SuppressUnmanagedCodeSecurity]
        private static unsafe extern Constants.RetCode inflateInit2_(ZStream* strm, int windowBits, byte* version, int streamSize);

        [DllImport(DllPath), SuppressUnmanagedCodeSecurity]
        private static unsafe extern byte* zlibVersion();

        [DllImport(DllPath), SuppressUnmanagedCodeSecurity]
        public static unsafe extern Constants.RetCode deflateEnd(ZStream* strm);

        [DllImport(DllPath), SuppressUnmanagedCodeSecurity]
        public static unsafe extern Constants.RetCode deflate(ZStream* strm, Constants.Flush flush);

        public static unsafe Constants.RetCode DeflateInit2(ZStream* strm, int level, int method, int windowBits,
            int memLevel, Constants.Strategy strategy)
        {
            return deflateInit2_(strm, level, method, windowBits, memLevel, strategy, zlibVersion(), sizeof(ZStream));
        }

        public static unsafe Constants.RetCode InflateInit2(ZStream* strm, int windowBits)
        {
            return inflateInit2_(strm, windowBits, zlibVersion(), sizeof(ZStream));
        }

        [DllImport(DllPath), SuppressUnmanagedCodeSecurity]
        public static unsafe extern Constants.RetCode inflate(ZStream* strm, Constants.Flush flush);

        [DllImport(DllPath), SuppressUnmanagedCodeSecurity]
        public static unsafe extern Constants.RetCode inflateEnd(ZStream* strm);
    }

    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct ZStream
    {
        /// <summary>
        /// Pointer to input buffer. Zlib inflate and deflate routine consumes data from this buffer.
        /// </summary>
        public byte* NextIn;
        /// <summary>
        /// Number of bytes available at next_in.
        /// </summary>
        public uint AvailIn;
        /// <summary>
        /// Total number of input bytes read so far.
        /// </summary>
        public uint TotalIn;

        /// <summary>
        /// Pointer to output buffer. Zlib inflate and deflate routine produce output to this location.
        /// </summary>
        public byte* NextOut;
        /// <summary>
        /// Remaining free space at next_out.
        /// </summary>
        public uint AvailOut;
        /// <summary>
        /// Total number of bytes output so far.
        /// </summary>
        public uint TotalOut;

        /// <summary>
        /// Last error message, NULL if no error.
        /// </summary>
        public byte* Msg;
        /// <summary>
        /// Internal state struct.
        /// </summary>
        public IntPtr State;

        /// <summary>
        /// Used to allocate the internal state.
        /// </summary>
        public IntPtr Zalloc;
        /// <summary>
        /// Used to free the internal state.
        /// </summary>
        public IntPtr Zfree;
        /// <summary>
        /// Private data object passed to zalloc and zfree.
        /// </summary>
        public IntPtr Opaque;

        /// <summary>
        /// Best guess about the data type: binary or text.
        /// </summary>
        public int DataType;
        /// <summary>
        /// Adler32 value of the uncompressed data.
        /// </summary>
        public uint Adler;
        /// <summary>
        /// Reserved for future use.
        /// </summary>
        public uint Reserved;
    }
}
