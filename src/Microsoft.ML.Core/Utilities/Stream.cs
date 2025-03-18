// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Internal.Utilities
{
    internal static partial class Utils
    {
        public static void CloseEx(this Stream stream)
        {
            if (stream == null)
                return;
            stream.Close();
        }

        public static void CloseEx(this TextWriter writer)
        {
            if (writer == null)
                return;
            writer.Close();
        }

        /// <summary>
        /// Similar to Stream.CopyTo but takes a length rather than assuming copy to end.  Returns amount copied.
        /// </summary>
        /// <param name="source">Source stream to copy from</param>
        /// <param name="destination">Destination stream to copy to</param>
        /// <param name="length">Number of bytes to copy</param>
        /// <param name="bufferSize">Size of buffer to use when copying, default is 81920 to match that of Stream</param>
        /// <returns>number of bytes copied</returns>
        public static long CopyRange(this Stream source, Stream destination, long length, int bufferSize = 81920)
        {
            // should use ArrayPool once we can take that dependency
            byte[] buffer = new byte[bufferSize];
            int read;
            long remaining = length;
            while (remaining != 0 &&
                   (read = source.Read(buffer, 0, (int)Math.Min(buffer.Length, remaining))) != 0)
            {
                destination.Write(buffer, 0, read);
                remaining -= read;
            }

            return length - remaining;
        }

        public static void WriteBoolByte(this BinaryWriter writer, bool x)
        {
            Contracts.AssertValue(writer);
            writer.Write((byte)(x ? 1 : 0));
        }

        /// <summary>
        /// Writes a length prefixed span of ints.
        /// </summary>
        public static void WriteIntArray(this BinaryWriter writer, ReadOnlySpan<int> values)
        {
            Contracts.AssertValue(writer);

            writer.Write(values.Length);
            for (int i = 0; i < values.Length; i++)
                writer.Write(values[i]);
        }

        /// <summary>
        /// Writes a span of ints without the length prefix.
        /// </summary>
        public static void WriteIntsNoCount(this BinaryWriter writer, ReadOnlySpan<int> values)
        {
            Contracts.AssertValue(writer);

            for (int i = 0; i < values.Length; i++)
                writer.Write(values[i]);
        }

        /// <summary>
        /// Writes a length prefixed span of uints.
        /// </summary>
        public static void WriteUIntArray(this BinaryWriter writer, ReadOnlySpan<uint> values)
        {
            Contracts.AssertValue(writer);

            writer.Write(values.Length);
            for (int i = 0; i < values.Length; i++)
                writer.Write(values[i]);
        }

        /// <summary>
        /// Writes a span of uints without the length prefix.
        /// </summary>
        public static void WriteUIntsNoCount(this BinaryWriter writer, ReadOnlySpan<uint> values)
        {
            Contracts.AssertValue(writer);

            for (int i = 0; i < values.Length; i++)
                writer.Write(values[i]);
        }

        /// <summary>
        /// Writes a length prefixed array of bytes.
        /// </summary>
        public static void WriteByteArray(this BinaryWriter writer, byte[] values)
        {
            Contracts.AssertValue(writer);
            Contracts.AssertValueOrNull(values);

            if (values == null)
            {
                writer.Write(0);
                return;
            }

            writer.Write(values.Length);

            // This method doesn't write the length of the binary array.
            writer.Write(values);
        }

        /// <summary>
        /// Writes a length prefixed span of bytes.
        /// </summary>
        public static void WriteByteArray(this BinaryWriter writer, ReadOnlySpan<byte> values)
        {
            Contracts.AssertValue(writer);

            writer.Write(values.Length);

            for (int i = 0; i < values.Length; i++)
                writer.Write(values[i]);
        }

        /// <summary>
        /// Writes a length prefixed array of bytes.
        /// </summary>
        public static void WriteByteArray(this BinaryWriter writer, byte[] values, int count)
        {
            Contracts.AssertValue(writer);
            Contracts.AssertValueOrNull(values);
            Contracts.Assert(0 <= count && count <= Utils.Size(values));

            writer.Write(count);
            writer.Write(values, 0, count);
        }

        /// <summary>
        /// Writes an array of bytes without the length prefix.
        /// </summary>
        public static void WriteBytesNoCount(this BinaryWriter writer, byte[] values, int count)
        {
            Contracts.AssertValue(writer);
            Contracts.AssertValueOrNull(values);
            Contracts.Assert(0 <= count && count <= Utils.Size(values));

            writer.Write(values, 0, count);
        }

        /// <summary>
        /// Writes a length prefixed span of Floats.
        /// </summary>
        public static void WriteSingleArray(this BinaryWriter writer, ReadOnlySpan<float> values)
        {
            Contracts.AssertValue(writer);

            writer.Write(values.Length);
            for (int i = 0; i < values.Length; i++)
                writer.Write(values[i]);
        }

        /// <summary>
        /// Writes a length prefixed array of Floats.
        /// </summary>
        public static void WriteSingleArray(this BinaryWriter writer, IEnumerable<float> values, int count)
        {
            Contracts.AssertValue(writer);
            Contracts.AssertValue(values);

            writer.Write(count);
            int cv = 0;
            foreach (var val in values)
            {
                Contracts.Assert(cv < count);
                writer.Write(val);
                cv++;
            }
            Contracts.Assert(cv == count);
        }

        /// <summary>
        /// Writes a span of Floats without the length prefix.
        /// </summary>
        public static void WriteSinglesNoCount(this BinaryWriter writer, ReadOnlySpan<float> values)
        {
            Contracts.AssertValue(writer);

            for (int i = 0; i < values.Length; i++)
                writer.Write(values[i]);
        }

        /// <summary>
        /// Writes a length prefixed span of doubles.
        /// </summary>
        public static void WriteDoubleArray(this BinaryWriter writer, ReadOnlySpan<double> values)
        {
            Contracts.AssertValue(writer);

            writer.Write(values.Length);
            for (int i = 0; i < values.Length; i++)
                writer.Write(values[i]);
        }

        /// <summary>
        /// Writes a span of doubles without the length prefix.
        /// </summary>
        public static void WriteDoublesNoCount(this BinaryWriter writer, ReadOnlySpan<double> values)
        {
            Contracts.AssertValue(writer);

            for (int i = 0; i < values.Length; i++)
                writer.Write(values[i]);
        }

        /// <summary>
        /// Writes a length prefixed span of bools as bytes with 0/1 values.
        /// </summary>
        public static void WriteBoolByteArray(this BinaryWriter writer, ReadOnlySpan<bool> values)
        {
            Contracts.AssertValue(writer);

            writer.Write(values.Length);
            for (int i = 0; i < values.Length; i++)
                writer.Write(values[i] ? (byte)1 : (byte)0);
        }

        /// <summary>
        /// Writes a span of bools as bytes with 0/1 values, without the length prefix.
        /// </summary>
        public static void WriteBoolBytesNoCount(this BinaryWriter writer, ReadOnlySpan<bool> values)
        {
            Contracts.AssertValue(writer);

            for (int i = 0; i < values.Length; i++)
                writer.Write(values[i] ? (byte)1 : (byte)0);
        }

        /// <summary>
        /// Writes a length prefixed span of chars.
        /// </summary>
        public static void WriteCharArray(this BinaryWriter writer, ReadOnlySpan<char> values)
        {
            Contracts.AssertValue(writer);

            writer.Write(values.Length);
            for (int i = 0; i < values.Length; i++)
                writer.Write((short)values[i]);
        }

        /// <summary>
        /// Writes a length prefixed array of packed bits.
        /// </summary>
        public static void WriteBitArray(this BinaryWriter writer, BitArray arr)
        {
            var numBits = Utils.Size(arr);
            writer.Write(numBits);
            if (numBits > 0)
            {
                var numBytes = (numBits + 7) / 8;
                var bytes = new byte[numBytes];
                arr.CopyTo(bytes, 0);
                writer.Write(bytes, 0, bytes.Length);
            }
        }

        public static long WriteSByteStream(this BinaryWriter writer, IEnumerable<sbyte> e)
        {
            long c = 0;
            foreach (var v in e)
            {
                writer.Write(v);
                c++;
            }
            return c;
        }

        public static long WriteByteStream(this BinaryWriter writer, IEnumerable<byte> e)
        {
            long c = 0;
            foreach (var v in e)
            {
                writer.Write(v);
                c++;
            }
            return c;
        }

        public static long WriteIntStream(this BinaryWriter writer, IEnumerable<int> e)
        {
            long c = 0;
            foreach (var v in e)
            {
                writer.Write(v);
                c++;
            }
            return c;
        }

        public static long WriteUIntStream(this BinaryWriter writer, IEnumerable<uint> e)
        {
            long c = 0;
            foreach (var v in e)
            {
                writer.Write(v);
                c++;
            }
            return c;
        }

        public static long WriteShortStream(this BinaryWriter writer, IEnumerable<short> e)
        {
            long c = 0;
            foreach (var v in e)
            {
                writer.Write(v);
                c++;
            }
            return c;
        }

        public static long WriteUShortStream(this BinaryWriter writer, IEnumerable<ushort> e)
        {
            long c = 0;
            foreach (var v in e)
            {
                writer.Write(v);
                c++;
            }
            return c;
        }

        public static long WriteLongStream(this BinaryWriter writer, IEnumerable<long> e)
        {
            long c = 0;
            foreach (var v in e)
            {
                writer.Write(v);
                c++;
            }
            return c;
        }

        public static long WriteULongStream(this BinaryWriter writer, IEnumerable<long> e)
        {
            long c = 0;
            foreach (var v in e)
            {
                writer.Write(v);
                c++;
            }
            return c;
        }

        public static long WriteSingleStream(this BinaryWriter writer, IEnumerable<float> e)
        {
            long c = 0;
            foreach (var v in e)
            {
                writer.Write(v);
                c++;
            }
            return c;
        }

        public static long WriteDoubleStream(this BinaryWriter writer, IEnumerable<double> e)
        {
            long c = 0;
            foreach (var v in e)
            {
                writer.Write(v);
                c++;
            }
            return c;
        }

        public static long WriteStringStream(this BinaryWriter writer, IEnumerable<string> e)
        {
            long c = 0;
            foreach (var v in e)
            {
                writer.Write(v);
                c++;
            }
            return c;
        }

        /// <summary>
        /// Writes what Microsoft calls a UTF-7 encoded number in the binary reader and
        /// writer string methods. For non-negative integers this is equivalent to LEB128
        /// (see https://en.wikipedia.org/wiki/LEB128).
        /// </summary>
        public static void WriteLeb128Int(this BinaryWriter writer, ulong value)
        {
            // Copied from the internal source code for Write7BitEncodedInt()
            while (value >= 0x80)
            {
                writer.Write((byte)(value | 0x80));
                value >>= 7;
            }
            writer.Write((byte)value);
        }

        /// <summary>
        /// The number of bytes that would be written if one were to attempt to write
        /// the value in LEB128.
        /// </summary>
        public static int Leb128IntLength(ulong value)
        {
            int len = 1;
            while (value >= 0x80)
            {
                len++;
                value >>= 7;
            }
            return len;
        }

        public static long FpCur(this BinaryWriter writer)
        {
            return writer.BaseStream.Position;
        }

        public static void Seek(this BinaryWriter writer, long fp)
        {
            writer.BaseStream.Position = fp;
        }

        public static long FpCur(this BinaryReader reader)
        {
            return reader.BaseStream.Position;
        }

        public static void Seek(this BinaryReader reader, long fp)
        {
            reader.BaseStream.Position = fp;
        }

        public static bool ReadBoolByte(this BinaryReader reader)
        {
            byte b = reader.ReadByte();
            Contracts.CheckDecode(b <= 1);
            return b != 0;
        }

        public static float ReadFloat(this BinaryReader reader)
        {
            return reader.ReadSingle();
        }

        public static float[] ReadFloatArray(this BinaryReader reader)
        {
            Contracts.AssertValue(reader);

            int size = reader.ReadInt32(); // reading trusted capacity value from data stream
            Contracts.CheckDecode(size >= 0);
            return ReadFloatArray(reader, size);
        }

        public static float[] ReadFloatArray(this BinaryReader reader, int size)
        {
            Contracts.AssertValue(reader);
            Contracts.Assert(size >= 0);

            if (size == 0)
                return null;
            var values = new float[size];

            ReadBinaryDataIntoSpan(reader, values.AsSpan());

            return values;
        }

        public static void ReadFloatArray(this BinaryReader reader, float[] array, int start, int count)
        {
            Contracts.AssertValue(reader);
            Contracts.AssertValue(array);
            Contracts.Assert(0 <= start && start < array.Length);
            Contracts.Assert(0 < count && count <= array.Length - start);

            ReadBinaryDataIntoSpan(reader, array.AsSpan(start, count));
        }

        public static float[] ReadSingleArray(this BinaryReader reader)
        {
            return reader.ReadFloatArray();
        }

        public static float[] ReadSingleArray(this BinaryReader reader, int size)
        {
            return reader.ReadFloatArray(size);
        }

        public static double[] ReadDoubleArray(this BinaryReader reader)
        {
            Contracts.AssertValue(reader);

            int size = reader.ReadInt32(); // reading trusted capacity value from data stream
            Contracts.CheckDecode(size >= 0);
            return ReadDoubleArray(reader, size);
        }

        public static double[] ReadDoubleArray(this BinaryReader reader, int size)
        {
            Contracts.AssertValue(reader);
            Contracts.Assert(size >= 0);
            if (size == 0)
                return null;
            var values = new double[size];

            ReadBinaryDataIntoSpan(reader, values.AsSpan());

            return values;
        }

        public static int[] ReadIntArray(this BinaryReader reader)
        {
            Contracts.AssertValue(reader);

            int size = reader.ReadInt32(); // reading trusted capacity value from data stream
            Contracts.CheckDecode(size >= 0);
            return ReadIntArray(reader, size);
        }

        public static int[] ReadIntArray(this BinaryReader reader, int size)
        {
            Contracts.AssertValue(reader);
            Contracts.Assert(size >= 0);

            if (size == 0)
                return null;
            var values = new int[size];

            ReadBinaryDataIntoSpan(reader, values.AsSpan());

            return values;
        }

        public static uint[] ReadUIntArray(this BinaryReader reader)
        {
            Contracts.AssertValue(reader);

            int size = reader.ReadInt32(); // reading trusted capacity value from data stream
            Contracts.CheckDecode(size >= 0);
            return ReadUIntArray(reader, size);
        }

        public static uint[] ReadUIntArray(this BinaryReader reader, int size)
        {
            Contracts.AssertValue(reader);
            Contracts.Assert(size >= 0);

            if (size == 0)
                return null;
            var values = new uint[size];

            ReadBinaryDataIntoSpan(reader, values.AsSpan());

            return values;
        }

        public static long[] ReadLongArray(this BinaryReader reader)
        {
            Contracts.AssertValue(reader);

            int size = reader.ReadInt32(); // reading trusted capacity value from data stream
            Contracts.CheckDecode(size >= 0);
            return ReadLongArray(reader, size);
        }

        public static long[] ReadLongArray(this BinaryReader reader, int size)
        {
            Contracts.AssertValue(reader);
            Contracts.Assert(size >= 0);

            if (size == 0)
                return null;
            var values = new long[size];

            ReadBinaryDataIntoSpan(reader, values.AsSpan());

            return values;
        }

        public static bool[] ReadBoolArray(this BinaryReader reader)
        {
            Contracts.AssertValue(reader);

            int size = reader.ReadInt32(); // reading trusted capacity value from data stream
            Contracts.CheckDecode(size >= 0);
            return ReadBoolArray(reader, size);
        }

        public static bool[] ReadBoolArray(this BinaryReader reader, int size)
        {
            Contracts.AssertValue(reader);
            Contracts.Assert(size >= 0);

            if (size == 0)
                return null;
            var values = new bool[size];

            // It is in general not safe to populate a bool[] with untrusted input.
            // The call below assumes the input stream is trusted.
            ReadBinaryDataIntoSpan(reader, values.AsSpan());

            return values;
        }

        public static char[] ReadCharArray(this BinaryReader reader)
        {
            Contracts.AssertValue(reader);

            int size = reader.ReadInt32(); // reading trusted capacity value from data stream
            Contracts.CheckDecode(size >= 0);
            return ReadCharArray(reader, size);
        }

        public static char[] ReadCharArray(this BinaryReader reader, int size)
        {
            Contracts.AssertValue(reader);
            Contracts.Assert(size >= 0);

            if (size == 0)
                return null;
            var values = new char[size];

            ReadBinaryDataIntoSpan(reader, values.AsSpan());

            return values;
        }

        public static byte[] ReadByteArray(this BinaryReader reader)
        {
            Contracts.AssertValue(reader);

            int size = reader.ReadInt32(); // reading trusted capacity value from data stream
            Contracts.CheckDecode(size >= 0);
            return ReadByteArray(reader, size);
        }

        public static byte[] ReadByteArray(this BinaryReader reader, int size)
        {
            Contracts.AssertValue(reader);
            Contracts.Assert(size >= 0);

            if (size == 0)
                return null;
            var bytes = reader.ReadBytes(size);
            Contracts.CheckDecode(bytes.Length == size);
            return bytes;
        }

        public static BitArray ReadBitArray(this BinaryReader reader)
        {
            int numBits = reader.ReadInt32(); // reading trusted capacity value from data stream
            Contracts.CheckDecode(numBits >= 0);
            if (numBits == 0)
                return null;
            var numBytes = (numBits + 7) / 8; // trusted capacity value expected not to integer overflow
            var bytes = reader.ReadByteArray(numBytes);
            var returnArray = new BitArray(bytes);
            returnArray.Length = numBits;
            return returnArray;
        }

        private static unsafe void ReadBinaryDataIntoSpan<T>(BinaryReader reader, Span<T> destination) where T : unmanaged
        {
            Contracts.AssertValue(reader);
            Contracts.Assert(!destination.IsEmpty);

            // There are two considerations here. First, we want to keep all temporary arrays (even pooled arrays)
            // under some threshold size. Second, when we project the Span<T> to bytes, we need to do it in chunks,
            // as trying to project the entire span at once will lead to integer overflow if the byte length
            // exceeds int.MaxLength.

            const int maxChunkSizeInBytes = 70 * 1024;
            int maxChunkSizeInElements = maxChunkSizeInBytes / sizeof(T);
            Contracts.Assert(maxChunkSizeInElements > 0, "Unexpectedly large T.");

            // Rent a byte[] instead of a T[] to allow reuse of buffers across different types T.
            byte[] rentedArray = ArrayPool<byte>.Shared.Rent(maxChunkSizeInElements * sizeof(T));
            try
            {
                while (!destination.IsEmpty)
                {
                    int numElementsToReadThisChunk = Math.Min(maxChunkSizeInElements, destination.Length);
                    int rentedArrayOffset = 0;
                    int numBytesRemainingToReadThisChunk = numElementsToReadThisChunk * sizeof(T); // n.b. not necessarily populating the entire rented array

                    do
                    {
                        int numBytesReadJustNow = reader.Read(rentedArray, rentedArrayOffset, numBytesRemainingToReadThisChunk);
                        rentedArrayOffset += numBytesReadJustNow;
                        numBytesRemainingToReadThisChunk -= numBytesReadJustNow;
                    } while (numBytesRemainingToReadThisChunk > 0);

                    // Copy the rented array to the destination span (projected as bytes).
                    // This projection as bytes is safe as long as T is a primitive numeric type (integers, floats).
                    // Avoid projecting the Span<byte> as a Span<T> to avoid potential alignment issues.
                    rentedArray.AsSpan(0, numElementsToReadThisChunk * sizeof(T)).CopyTo(MemoryMarshal.AsBytes(destination.Slice(0, numElementsToReadThisChunk)));
                    destination = destination.Slice(numElementsToReadThisChunk);
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(rentedArray);
            }
        }

        /// <summary>
        /// If this return it will read exactly length bytes, and unlike the
        /// regular read method fails if it cannot.
        /// </summary>
        /// <param name="s">The stream</param>
        /// <param name="buff">The buffer into which to write the data.</param>
        /// <param name="offset">The offset of the output array into which to write.</param>
        /// <param name="length">The number of bytes to read.</param>
        public static void ReadBlock(this Stream s, byte[] buff, int offset, int length)
        {
            int pos = 0;
            int read;
            while (pos != length)
            {
                read = s.Read(buff, offset + pos, length - pos);
                Contracts.CheckIO(read > 0, "Unexpected failure to read");
                pos += read;
            }
        }

        /// <summary>
        /// If this return it will try to read exactly length bytes.
        /// </summary>
        /// <param name="s">The stream</param>
        /// <param name="buff">The buffer into which to write the data.</param>
        /// <param name="offset">The offset of the output array into which to write.</param>
        /// <param name="length">The number of bytes to read.</param>
        public static int TryReadBlock(this Stream s, byte[] buff, int offset, int length)
        {
            int pos = 0;
            int read = -1;
            while (pos != length && read != 0)
            {
                read = s.Read(buff, offset + pos, length - pos);
                pos += read;
            }

            return pos;
        }

        /// <summary>
        /// Reads a LEB128 encoded unsigned integer.
        /// </summary>
        public static ulong ReadLeb128Int(this BinaryReader reader)
        {
            // Copied from the internal source code for Read7BitEncodedInt()
            ulong value = 0;
            int shift = 0;
            byte b;
            do
            {
                // ReadByte handles end of stream cases for us.
                b = reader.ReadByte();
                if (shift == 9 * 7 && b > 0x01)
                    throw Contracts.ExceptDecode("LEB128 encoded integer exceeded expected length");
                value |= (((ulong)(b & 0x7F)) << shift);
                shift += 7;
            } while ((b & 0x80) != 0);
            return value;
        }

        private static Encoding _utf8NoBom;

        /// <summary>
        /// A convenience method to open a stream writer, by default with no-BOM UTF-8 encoding,
        /// buffer size of 1K, and the stream left open.
        /// </summary>
        public static StreamWriter OpenWriter(Stream stream, Encoding encoding = null, int bufferSize = 1024, bool leaveOpen = true)
        {
            Contracts.CheckValue(stream, nameof(stream));
            Contracts.CheckParam(0 < bufferSize, nameof(bufferSize), "buffer size must be positive");
            if (encoding == null)
            {
                // Even though the StreamWriter default encoding is BOM-less UTF8, note
                // that Encoding.UTF8 indicates we should write with a BOM!!
                if (_utf8NoBom == null)
                    Interlocked.CompareExchange(ref _utf8NoBom, new UTF8Encoding(false), null);
                encoding = _utf8NoBom;
            }
            return new StreamWriter(stream, encoding, bufferSize: bufferSize, leaveOpen: leaveOpen);
        }

#if !CORECLR // REVIEW: Remove this once we're on a .Net version that has it as an instance method.
        /// <summary>
        /// This extension method assumes that the origin was zero.
        /// </summary>
        public static bool TryGetBuffer(this MemoryStream mem, out ArraySegment<byte> buffer)
        {
            try
            {
                var bytes = mem.GetBuffer();
                buffer = new ArraySegment<byte>(bytes, 0, (int)mem.Length);
                return true;
            }
            catch (UnauthorizedAccessException)
            {
                buffer = default(ArraySegment<byte>);
                return false;
            }
        }
#endif
        // REVIEW: need to plumb IExceptionContext into the method.
        /// <summary>
        /// Checks that the directory of the file name passed in already exists.
        /// This is meant to be called before calling an API that creates the file,
        /// so the file need not exist.
        /// </summary>
        /// <param name="file">An absolute or relative file path, or null to skip the check
        /// (useful for optional user parameters)</param>
        /// <param name="userArgument">The user level parameter name, as exposed by the command line help</param>
        public static void CheckOptionalUserDirectory(string file, string userArgument)
        {
            if (string.IsNullOrWhiteSpace(file))
                return;

            // We can't check for URI directories.
            if (Uri.IsWellFormedUriString(file, UriKind.Absolute))
                return;

            string dir;
#pragma warning disable MSML_ContractsNameUsesNameof
            try
            {
                // Relative paths are interpreted as local.
                dir = Path.GetDirectoryName(Path.GetFullPath(file));
            }
            catch (NotSupportedException exc)
            {
                throw Contracts.ExceptUserArg(userArgument, exc.Message);
            }
            catch (PathTooLongException exc)
            {
                throw Contracts.ExceptUserArg(userArgument, exc.Message);
            }
            catch (System.Security.SecurityException exc)
            {
                throw Contracts.ExceptUserArg(userArgument, exc.Message);
            }
            if (!Directory.Exists(dir))
                throw Contracts.ExceptUserArg(userArgument, "Cannot find directory '{0}'.", dir);
        }
#pragma warning restore MSML_ContractsNameUsesNameof
    }
}
