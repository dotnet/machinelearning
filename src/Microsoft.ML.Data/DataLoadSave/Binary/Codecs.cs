// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;

namespace Microsoft.ML.Data.IO
{
    internal sealed partial class CodecFactory
    {
        /// <summary>
        /// A convenient base class for value writers.
        /// </summary>
        private abstract class ValueWriterBase<T> : IValueWriter<T>, IDisposable
        {
            protected readonly CodecFactory Factory;
            protected Stream Stream;
            protected BinaryWriter Writer;

            protected bool Disposed => Writer == null;

            public ValueWriterBase(CodecFactory factory, Stream stream)
            {
                Contracts.AssertValue(stream);
                Contracts.AssertValue(factory);
                Factory = factory;
                Stream = stream;
                Writer = Factory.OpenBinaryWriter(Stream);
            }

            public virtual void Dispose()
            {
                if (!Disposed)
                {
                    Writer.Dispose();
                    Writer = null;
                    Stream = null;
                }
            }

            public abstract void Write(in T value);

            public virtual void Write(ReadOnlySpan<T> values)
            {
                // Basic un-optimized reference implementation.
                for (int i = 0; i < values.Length; ++i)
                    Write(in values[i]);
            }

            public abstract void Commit();

            public abstract long GetCommitLengthEstimate();
        }

        /// <summary>
        /// A convenient base class for value readers.
        /// </summary>
        private abstract class ValueReaderBase<T> : IValueReader<T>, IDisposable
        {
            protected readonly CodecFactory Factory;
            protected Stream Stream;
            protected BinaryReader Reader;

            protected bool Disposed => Reader == null;

            public ValueReaderBase(CodecFactory factory, Stream stream)
            {
                Contracts.AssertValue(stream);
                Contracts.AssertValue(factory);
                Factory = factory;
                Stream = stream;
                Reader = Factory.OpenBinaryReader(Stream);
            }

            public virtual void Dispose()
            {
                if (!Disposed)
                {
                    Reader.Dispose();
                    Reader = null;
                    Stream = null;
                }
            }

            public abstract void MoveNext();

            public abstract void Get(ref T value);

            public virtual void Read(T[] values, int index, int count)
            {
                Contracts.Assert(0 <= index && index <= Utils.Size(values));
                Contracts.Assert(0 <= count && count <= Utils.Size(values) - index);
                // Basic un-optimized reference implementation.
                for (int i = 0; i < count; ++i)
                {
                    MoveNext();
                    Get(ref values[i + index]);
                }
            }
        }

        /// <summary>
        /// A simple codec is useful for those types with no parameterizations.
        /// </summary>
        private abstract class SimpleCodec<T> : IValueCodec<T>
        {
            protected readonly CodecFactory Factory;

            public ColumnType Type { get; }

            // For these basic types, the class name will do perfectly well.
            public virtual string LoadName => typeof(T).Name;

            public SimpleCodec(CodecFactory factory, ColumnType type)
            {
                Contracts.AssertValue(factory);
                Contracts.AssertValue(type);
                Contracts.Assert(type.RawType == typeof(T));
                Factory = factory;
                Type = type;
            }

            public bool GetCodec(Stream definitionStream, out IValueCodec codec)
            {
                codec = this;
                return true;
            }

            public int WriteParameterization(Stream stream)
            {
                // The simple codecs do not have any sort of dimensionality or subtypes, so they write nothing.
                return 0;
            }

            public abstract IValueWriter<T> OpenWriter(Stream stream);

            public abstract IValueReader<T> OpenReader(Stream stream, int items);
        }

        /// <summary>
        /// This codec is for use with types that have <c>UnsafeTypeOps</c> operations defined.
        /// Generally, this corresponds to numeric types that can be safely blitted.
        /// </summary>
        private sealed class UnsafeTypeCodec<T> : SimpleCodec<T> where T : struct
        {
            // *** Binary block format ***
            // Packed bytes of little-endian values.

            private readonly UnsafeTypeOps<T> _ops;

            // Gatekeeper to ensure T is a type that is supported by UnsafeTypeCodec.
            // Throws an exception if T is neither a TimeSpan nor a NumberType.
            private static ColumnType UnsafeColumnType(Type type)
            {
                return type == typeof(TimeSpan) ? (ColumnType)TimeSpanType.Instance : NumberType.FromType(type);
            }

            public UnsafeTypeCodec(CodecFactory factory)
                : base(factory, UnsafeColumnType(typeof(T)))
            {
                _ops = UnsafeTypeOpsFactory.Get<T>();
            }

            public override IValueWriter<T> OpenWriter(Stream stream)
            {
                return new Writer(this, stream);
            }

            public override IValueReader<T> OpenReader(Stream stream, int items)
            {
                return new Reader(this, stream, items);
            }

            private sealed class Writer : ValueWriterBase<T>
            {
                private readonly byte[] _buffer;
                private readonly UnsafeTypeOps<T> _ops;
                private long _numWritten;

                public Writer(UnsafeTypeCodec<T> codec, Stream stream)
                    : base(codec.Factory, stream)
                {
                    _buffer = new byte[1 << 15];
                    _ops = codec._ops;
                }

                public override void Write(in T value)
                {
                    _ops.Write(value, Writer);
                    _numWritten++;
                }

                public override void Write(ReadOnlySpan<T> values)
                {
                    int count = values.Length;
                    _ops.Apply(values, ptr =>
                    {
                        // REVIEW: In some future work we will want to avoid needless copies by
                        // seeing if this is a stream that can work over IntPtr writes or reads.
                        int byteLength = count * _ops.Size;
                        while (byteLength > 0)
                        {
                            int sublen = Math.Min(byteLength, _buffer.Length);
                            Marshal.Copy(ptr, _buffer, 0, sublen);
                            Stream.Write(_buffer, 0, sublen);
                            ptr += sublen;
                            byteLength -= sublen;
                        }
                    });
                    _numWritten += count;
                }

                public override void Commit()
                {
                    // No state or structure to flush. This does nothing.
                }

                public override long GetCommitLengthEstimate()
                {
                    return _ops.Size * _numWritten;
                }
            }

            private sealed class Reader : ValueReaderBase<T>
            {
                private readonly byte[] _buffer;
                private readonly UnsafeTypeOps<T> _ops;
                private int _remaining;
                private T _value;

                public Reader(UnsafeTypeCodec<T> codec, Stream stream, int items)
                    : base(codec.Factory, stream)
                {
                    _buffer = new byte[1 << 15];
                    _ops = codec._ops;
                    _remaining = items;
                }

                public override void MoveNext()
                {
                    Contracts.Assert(_remaining > 0);
                    _value = _ops.Read(Reader);
                    _remaining--;
                }

                public override void Get(ref T value)
                {
                    value = _value;
                }

                public override void Read(T[] values, int index, int count)
                {
                    Contracts.Assert(0 <= index && index <= Utils.Size(values));
                    Contracts.Assert(0 <= count && count <= Utils.Size(values) - index);
                    Contracts.Assert(_remaining >= count);
                    _ops.Apply(values, ptr =>
                    {
                        int offset = index * _ops.Size;
                        int byteLength = count * _ops.Size;
                        ptr += offset;
                        while (byteLength > 0)
                        {
                            int sublen = Math.Min(byteLength, _buffer.Length);
                            Stream.ReadBlock(_buffer, 0, sublen);
                            Marshal.Copy(_buffer, 0, ptr, sublen);
                            ptr += sublen;
                            byteLength -= sublen;
                        }
                    });
                    _remaining -= count;
                }
            }
        }

        private sealed class TextCodec : SimpleCodec<ReadOnlyMemory<char>>
        {
            private const int LengthMask = unchecked((int)0x7FFFFFFF);

            public override string LoadName
            {
                get { return "TextSpan"; }
            }

            // *** Binary block format ***
            // int: Number of entries.
            // int[entries]: The non-decreasing end-boundary character index array, with high bit set for "missing" values.
            // string: The UTF-8 encoded string, with the standard LEB128 byte-length preceeding it.

            public TextCodec(CodecFactory factory)
                : base(factory, TextType.Instance)
            {
            }

            public override IValueWriter<ReadOnlyMemory<char>> OpenWriter(Stream stream)
            {
                return new Writer(this, stream);
            }

            public override IValueReader<ReadOnlyMemory<char>> OpenReader(Stream stream, int items)
            {
                return new Reader(this, stream, items);
            }

            private sealed class Writer : ValueWriterBase<ReadOnlyMemory<char>>
            {
                private StringBuilder _builder;
                private List<int> _boundaries;

                public Writer(TextCodec codec, Stream stream)
                    : base(codec.Factory, stream)
                {
                    _builder = new StringBuilder();
                    _boundaries = new List<int>();
                }

                public override void Write(in ReadOnlyMemory<char> value)
                {
                    Contracts.Check(_builder != null, "writer was already committed");
                    _builder.AppendMemory(value);
                    _boundaries.Add(_builder.Length);
                }

                public override void Commit()
                {
                    Contracts.Check(_builder != null, "writer already committed");

                    Writer.Write(_boundaries.Count); // Write the number of entries.
                    Writer.WriteIntStream(_boundaries); // Write the entries end boundaries, in character counts.
                    Writer.Write(_builder.ToString());
                    _builder.Clear();
                    _builder = null;
                }

                public override long GetCommitLengthEstimate()
                {
                    // This is an estimate, only exact if the number of input
                    // characters takes one byte in the UTF-8 encoding.
                    return sizeof(int) * (1 + (long)_boundaries.Count) + _builder.Length;
                }
            }

            private sealed class Reader : ValueReaderBase<ReadOnlyMemory<char>>
            {
                private readonly int _entries;
                private readonly int[] _boundaries;
                private int _index;
                private string _text;

                public Reader(TextCodec codec, Stream stream, int items)
                    : base(codec.Factory, stream)
                {
                    _entries = Reader.ReadInt32();
                    Contracts.CheckDecode(_entries == items);
                    _index = -1;
                    _boundaries = new int[_entries + 1];
                    int bPrev = 0;
                    for (int i = 1; i < _boundaries.Length; ++i)
                    {
                        int b = _boundaries[i] = Reader.ReadInt32();
                        Contracts.CheckDecode(b >= (bPrev & LengthMask) || (b & LengthMask) == (bPrev & LengthMask));
                        bPrev = b;
                    }
                    _text = Reader.ReadString();
                    Contracts.CheckDecode(_text.Length == (_boundaries[_entries] & LengthMask));
                }

                public override void MoveNext()
                {
                    Contracts.Check(++_index < _entries, "reader already read all values");
                }

                public override void Get(ref ReadOnlyMemory<char> value)
                {
                    Contracts.Assert(_index < _entries);
                    int b = _boundaries[_index + 1];
                    int start = _boundaries[_index] & LengthMask;
                    if (b >= 0)
                        value = _text.AsMemory().Slice(start, (b & LengthMask) - start);
                    else
                    {
                        //For backward compatiblity when NA values existed, treat them
                        //as empty string.
                        value = ReadOnlyMemory<char>.Empty;
                    }
                }
            }
        }

        /// <summary>
        /// This is a boolean code that reads from a form that serialized
        /// 1 bit per value. The old encoding (implemented by a different codec)
        /// uses 2 bits per value so NA values can be supported.
        /// </summary>
        private sealed class BoolCodec : SimpleCodec<bool>
        {
            // *** Binary block format ***
            // Packed bits.

            public BoolCodec(CodecFactory factory)
                : base(factory, BoolType.Instance)
            {
            }

            public override string LoadName
            {
                get { return typeof(bool).Name; }
            }

            public override IValueWriter<bool> OpenWriter(Stream stream)
            {
                return new Writer(this, stream);
            }

            private sealed class Writer : ValueWriterBase<bool>
            {
                // Pack 8 values into 8 bits.
                private byte _currentBits;
                private long _numWritten;
                private byte _currentIndex;

                public Writer(BoolCodec codec, Stream stream)
                    : base(codec.Factory, stream)
                {
                }

                public override void Write(in bool value)
                {
                    Contracts.Assert(0 <= _currentIndex && _currentIndex < 8);

                    _numWritten++;
                    if (value)
                        _currentBits |= (byte)(1 << _currentIndex);

                    _currentIndex++;
                    if (_currentIndex == 8)
                    {
                        Writer.Write(_currentBits);
                        _currentBits = 0;
                        _currentIndex = 0;
                    }
                }

                // REVIEW: More efficient array writers are certainly possible.

                public override long GetCommitLengthEstimate()
                {
                    return 4 * (((_numWritten - 1) >> 4) + 1);
                }

                public override void Commit()
                {
                    if (_currentIndex > 0)
                    {
                        Writer.Write(_currentBits);
                        _currentBits = 0;
                        _currentIndex = 0;
                    }
                }
            }

            public override IValueReader<bool> OpenReader(Stream stream, int items)
            {
                return new Reader(this, stream, items);
            }

            private sealed class Reader : ValueReaderBase<bool>
            {
                private byte _currentBits;
                private int _currentIndex;
                private int _remaining;

                public Reader(BoolCodec codec, Stream stream, int items)
                    : base(codec.Factory, stream)
                {
                    _remaining = items;
                    _currentIndex = -1;
                }

                public override void MoveNext()
                {
                    Contracts.Assert(0 < _remaining, "already consumed all values");
                    --_remaining;
                    if ((_currentIndex = (_currentIndex + 1) & 7) == 0)
                        _currentBits = Reader.ReadByte();
                    else
                        _currentBits >>= 1;
                }

                public override void Get(ref bool value)
                {
                    Contracts.Assert(0 <= _currentIndex, "have not moved in");
                    Contracts.Assert(_currentIndex < 8);
                    value = (_currentBits & 1) != 0;
                }
            }
        }

        private sealed class OldBoolCodec : SimpleCodec<bool>
        {
            // *** Binary block format ***
            // Pack 16 values into 32 bits, with 00 for false, 01 for true and 10 for NA.

            public OldBoolCodec(CodecFactory factory)
                : base(factory, BoolType.Instance)
            {
            }

            public override IValueWriter<bool> OpenWriter(Stream stream)
            {
                Contracts.Assert(false, "This older form only supports reading");
                throw Contracts.ExceptNotSupp("Writing single bit booleans no longer supported");
            }

            public override IValueReader<bool> OpenReader(Stream stream, int items)
            {
                return new Reader(this, stream, items);
            }

            private sealed class Reader : ValueReaderBase<bool>
            {
                private int _currentBits;
                private int _currentSlot;
                private int _remaining;

                public Reader(OldBoolCodec codec, Stream stream, int items)
                    : base(codec.Factory, stream)
                {
                    _remaining = items;
                    _currentSlot = -1;
                }

                public override void MoveNext()
                {
                    Contracts.Assert(0 < _remaining, "already consumed all values");
                    --_remaining;
                    if ((_currentSlot = (_currentSlot + 1) & 0x0F) == 0)
                        _currentBits = Reader.ReadInt32();
                    else
                        _currentBits = (int)((uint)_currentBits >> 2);
                }

                public override void Get(ref bool value)
                {
                    Contracts.Assert(0 <= _currentSlot, "have not moved in");
                    Contracts.Assert(_currentSlot < 16);
                    switch (_currentBits & 0x3)
                    {
                        case 0x0:
                            value = false;
                            break;
                        case 0x1:
                            value = true;
                            break;
                        case 0x2:
                            value = false;
                            break;
                        default:
                            throw Contracts.ExceptDecode("Invalid bit pattern in BoolCodec");
                    }
                }
            }
        }

        private sealed class DateTimeCodec : SimpleCodec<DateTime>
        {
            public DateTimeCodec(CodecFactory factory)
                : base(factory, DateTimeType.Instance)
            {
            }

            public override IValueWriter<DateTime> OpenWriter(Stream stream)
            {
                return new Writer(this, stream);
            }

            public override IValueReader<DateTime> OpenReader(Stream stream, int items)
            {
                return new Reader(this, stream, items);
            }

            private sealed class Writer : ValueWriterBase<DateTime>
            {
                private long _numWritten;

                public Writer(DateTimeCodec codec, Stream stream)
                    : base(codec.Factory, stream)
                {
                }

                public override void Write(in DateTime value)
                {
                    Writer.Write(value.Ticks);
                    _numWritten++;
                }

                public override void Commit()
                {
                    // No state or structure to flush. This does nothing.
                }

                public override long GetCommitLengthEstimate()
                {
                    return _numWritten * sizeof(long);
                }
            }

            private sealed class Reader : ValueReaderBase<DateTime>
            {
                private int _remaining;
                private DateTime _value;

                public Reader(DateTimeCodec codec, Stream stream, int items)
                    : base(codec.Factory, stream)
                {
                    _remaining = items;
                }

                public override void MoveNext()
                {
                    Contracts.Assert(_remaining > 0, "already consumed all values");

                    var ticks = Reader.ReadInt64();
                    _value = new DateTime(ticks == long.MinValue ? default : ticks);
                    _remaining--;
                }

                public override void Get(ref DateTime value)
                {
                    value = _value;
                }
            }
        }

        private sealed class DateTimeOffsetCodec : SimpleCodec<DateTimeOffset>
        {
            private readonly MadeObjectPool<long[]> _longBufferPool;
            private readonly MadeObjectPool<short[]> _shortBufferPool;

            public DateTimeOffsetCodec(CodecFactory factory)
                : base(factory, DateTimeOffsetType.Instance)
            {
                _longBufferPool = new MadeObjectPool<long[]>(() => null);
                _shortBufferPool = new MadeObjectPool<short[]>(() => null);
            }

            public override IValueWriter<DateTimeOffset> OpenWriter(Stream stream)
            {
                return new Writer(this, stream);
            }

            public override IValueReader<DateTimeOffset> OpenReader(Stream stream, int items)
            {
                return new Reader(this, stream, items);
            }

            private sealed class Writer : ValueWriterBase<DateTimeOffset>
            {
                private List<short> _offsets;
                private List<long> _ticks;

                public Writer(DateTimeOffsetCodec codec, Stream stream)
                    : base(codec.Factory, stream)
                {
                    _offsets = new List<short>();
                    _ticks = new List<long>();
                }

                public override void Write(in DateTimeOffset value)
                {
                    Contracts.Assert(_offsets != null, "writer was already committed");

                    _ticks.Add(value.DateTime.Ticks);

                    //DateTimeOffset exposes its offset as a TimeSpan, but internally it uses short and in minutes.
                    //https://github.com/dotnet/coreclr/blob/9499b08eefd895158c3f3c7834e185a73619128d/src/System.Private.CoreLib/shared/System/DateTimeOffset.cs#L51-L53
                    //https://github.com/dotnet/coreclr/blob/9499b08eefd895158c3f3c7834e185a73619128d/src/System.Private.CoreLib/shared/System/DateTimeOffset.cs#L286-L292
                    //From everything online(ISO8601, RFC3339, SQL Server doc, the offset supports the range -14 to 14 hours, and only supports minute precision.
                    _offsets.Add((short)(value.Offset.TotalMinutes));
                }

                public override void Commit()
                {
                    Contracts.Assert(_offsets != null, "writer was already committed");
                    Contracts.Assert(Utils.Size(_offsets) == Utils.Size(_ticks));

                    Writer.WriteShortStream(_offsets); // Write the offsets.
                    Writer.WriteLongStream(_ticks); // Write the tick values.
                    _offsets = null;
                    _ticks = null;
                }

                public override long GetCommitLengthEstimate()
                {
                    return (long)_offsets.Count * (sizeof(long) + sizeof(short));
                }
            }

            private sealed class Reader : ValueReaderBase<DateTimeOffset>
            {
                private readonly DateTimeOffsetCodec _codec;

                private readonly int _entries;
                private short[] _offsets;
                private long[] _ticks;
                private int _index;
                private bool _disposed;

                public Reader(DateTimeOffsetCodec codec, Stream stream, int items)
                    : base(codec.Factory, stream)
                {
                    _codec = codec;
                    _entries = items;
                    _index = -1;

                    _offsets = _codec._shortBufferPool.Get();
                    Utils.EnsureSize(ref _offsets, _entries, false);
                    for (int i = 0; i < _entries; i++)
                        _offsets[i] = Reader.ReadInt16();

                    _ticks = _codec._longBufferPool.Get();
                    Utils.EnsureSize(ref _ticks, _entries, false);
                    for (int i = 0; i < _entries; i++)
                        _ticks[i] = Reader.ReadInt64();
                }

                public override void MoveNext()
                {
                    Contracts.Assert(!_disposed);
                    Contracts.Check(++_index < _entries, "reader already read all values");
                }

                public override void Get(ref DateTimeOffset value)
                {
                    Contracts.Assert(!_disposed);
                    var ticks = _ticks[_index];
                    var offset = _offsets[_index];
                    value = new DateTimeOffset(new DateTime(ticks == long.MinValue ? default : ticks), new TimeSpan(0, offset == short.MinValue ? default : offset, 0));
                }

                public override void Dispose()
                {
                    if (!_disposed)
                    {
                        _codec._shortBufferPool.Return(_offsets);
                        _codec._longBufferPool.Return(_ticks);
                        _offsets = null;
                        _ticks = null;
                        _disposed = true;
                    }
                    base.Dispose();
                }
            }
        }

        private sealed class VBufferCodec<T> : IValueCodec<VBuffer<T>>
        {
            // *** Binary block format ***
            // int: Number of vectors.
            // int: If positive the length of all encoded vectors, or 0 if they are of variable size.
            // int[numVectors] (optional): If vectors of variable size, this array contains the lengths.
            // int[numVectors]: Counts of active elements per vector, or -1 if the array is dense.
            // int: Total number of indices. This must equal the sum of all non-negative elements of the counts array.
            // int[numIndices]: The packed array of indices for all the vbuffers.
            // <Values>: The packed sequence of values for all the vbuffers, written using the inner value codec's scheme.

            private readonly CodecFactory _factory;
            private readonly VectorType _type;
            // The codec for the internal elements.
            private readonly IValueCodec<T> _innerCodec;
            private readonly MadeObjectPool<T[]> _bufferPool;
            private readonly MadeObjectPool<int[]> _intBufferPool;

            public string LoadName { get { return "VBuffer"; } }

            public ColumnType Type { get { return _type; } }

            public VBufferCodec(CodecFactory factory, VectorType type, IValueCodec<T> innerCodec)
            {
                Contracts.AssertValue(factory);
                Contracts.AssertValue(type);
                Contracts.AssertValue(innerCodec);
                Contracts.Assert(type.RawType == typeof(VBuffer<T>));
                Contracts.Assert(innerCodec.Type == type.ItemType);
                _factory = factory;
                _type = type;
                _innerCodec = innerCodec;
                _bufferPool = new MadeObjectPool<T[]>(() => null);
                if (typeof(T) == typeof(int))
                    _intBufferPool = _bufferPool as MadeObjectPool<int[]>;
                else
                    _intBufferPool = new MadeObjectPool<int[]>(() => null);
            }

            public int WriteParameterization(Stream stream)
            {
                int total = _factory.WriteCodec(stream, _innerCodec);
                int count = _type.Dimensions.Length;
                total += sizeof(int) * (1 + count);
                using (BinaryWriter writer = _factory.OpenBinaryWriter(stream))
                {
                    writer.Write(count);
                    for (int i = 0; i < count; i++)
                        writer.Write(_type.Dimensions[i]);
                }
                return total;
            }

            public IValueWriter<VBuffer<T>> OpenWriter(Stream stream)
            {
                return new Writer(this, stream);
            }

            public IValueReader<VBuffer<T>> OpenReader(Stream stream, int items)
            {
                return new Reader(this, stream, items);
            }

            private sealed class Writer : ValueWriterBase<VBuffer<T>>
            {
                private readonly int _size;
                private readonly List<int> _lengths;
                private readonly List<int> _counts;
                private readonly List<int> _indices;

                private MemoryStream _valuesStream;
                private IValueWriter<T> _valueWriter;

                private bool FixedLength { get { return _size > 0; } }

                public Writer(VBufferCodec<T> codec, Stream stream)
                    : base(codec._factory, stream)
                {
                    _size = codec._type.Size;
                    _lengths = FixedLength ? null : new List<int>();
                    _counts = new List<int>();
                    _indices = new List<int>();

                    _valuesStream = Factory._memPool.Get();
                    _valueWriter = codec._innerCodec.OpenWriter(_valuesStream);
                }

                public override void Commit()
                {
                    Contracts.Check(_valuesStream != null, "writer already committed");

                    _valueWriter.Commit();
                    _valueWriter.Dispose();
                    _valueWriter = null;

                    // The number of vectors.
                    Writer.Write(_counts.Count);
                    // The fixed length of vectors, 0 if not fixed length.
                    if (FixedLength)
                    {
                        // If fixed length, we just write the length.
                        Writer.Write(_size);
                    }
                    else
                    {
                        // If not fixed length, still write the single length and skip the length
                        // if they happen to all be the same size. Else write 0.
                        int len = _lengths.Count == 0 ? 0 : _lengths[0];
                        for (int i = 1; i < _lengths.Count; ++i)
                        {
                            if (len != _lengths[i])
                            {
                                len = 0;
                                break;
                            }
                        }
                        Writer.Write(len);
                        if (len == 0)
                            Writer.WriteIntStream(_lengths);
                    }
                    // Write the counts.
                    Writer.WriteIntStream(_counts);
                    // Write the number of indices.
                    Writer.Write(_indices.Count);
                    // Write the indices.
                    Writer.WriteIntStream(_indices);

                    // Write the values. The way we created the memory stream, TryGetBuffer should not fail.
                    ArraySegment<byte> buffer;
                    bool tmp = _valuesStream.TryGetBuffer(out buffer);
                    Contracts.Assert(tmp, "TryGetBuffer failed in VBufferCodec!");
                    Stream.Write(buffer.Array, buffer.Offset, buffer.Count);

                    Factory._memPool.Return(ref _valuesStream);
                }

                public override long GetCommitLengthEstimate()
                {
                    long structureLength = sizeof(int) * (2 + (long)Utils.Size(_lengths) + _counts.Count + 1 + _indices.Count);
                    return structureLength + _valueWriter.GetCommitLengthEstimate();
                }

                public override void Write(in VBuffer<T> value)
                {
                    Contracts.Check(_valuesStream != null, "writer already committed");
                    if (FixedLength)
                    {
                        if (value.Length != _size)
                            throw Contracts.Except("Length mismatch: expected {0} slots but got {1}", _size, value.Length);
                    }
                    else
                        _lengths.Add(value.Length);
                    // REVIEW: In the non-fixed length case we can still check that the
                    // length is a multiple of the product of the non-zero tail sizes of the type.
                    var valueValues = value.GetValues();
                    if (value.IsDense)
                    {
                        _counts.Add(-1);
                        _valueWriter.Write(valueValues);
                    }
                    else
                    {
                        _counts.Add(valueValues.Length);
                        if (valueValues.Length > 0)
                        {
                            var valueIndices = value.GetIndices();
                            for (int i = 0; i < valueIndices.Length; i++)
                                _indices.Add(valueIndices[i]);
                            _valueWriter.Write(valueValues);
                        }
                    }
                }
            }

            private sealed class Reader : ValueReaderBase<VBuffer<T>>
            {
                private readonly VBufferCodec<T> _codec;

                // The fixed size of the vectors returned, with the same semantics as ColumnType.VectorSize,
                // that is, if zero, the vectors are *not* of fixed size. Even if the ColumnType happens
                // to be of non-fixed size, this may have a positive value if the vectors just so happen
                // to have equal length.
                private readonly int _size;
                // The number of vectors.
                private readonly int _numVectors;
                // This will be non-null only if the vectors are not of fixed size. In such a case, it is of
                // length equal to the number of vectors, with each element holding the length of the vectors.
                private readonly int[] _lengths;
                // This is of length equal to the number of vectors, and holds the count of values for the
                // sparse vbuffer (in which case it is non-negative), or -1 if the vector is dense.
                private readonly int[] _counts;
                // The packed array of indices. This may be null, if there are no indices, that is, all vectors
                // are dense.
                private readonly int[] _indices;
                // The packed array of values. This will be equal to the count of values in all arrays.
                private readonly T[] _values;

                private bool _disposed;
                // The current index of the vector. This will index _counts, and if applicable _lengths. This
                // will be incremented by one on movenext.
                private int _vectorIndex;
                // The current offset into the _indices array, for the current vector index. This will be
                // incremented on movenext by the current (soon to be previous) vector's number of specified
                // indices (0 if the vector is dense, or sparse with no values).
                private int _indicesOffset;
                // The current offset into the _values array, for the current vector index. This will be
                // incremented on movenext by the current (soon to be previous) vector's "count" of number
                // of specified values.
                private int _valuesOffset;

                private bool FixedLength { get { return _size > 0; } }

                public Reader(VBufferCodec<T> codec, Stream stream, int items)
                    : base(codec._factory, stream)
                {
                    _codec = codec;

                    // The number of vectors.
                    _numVectors = Reader.ReadInt32();
                    Contracts.CheckDecode(_numVectors == items);

                    // The length of all those vectors.
                    _size = Reader.ReadInt32();
                    if (codec._type.IsKnownSize)
                        Contracts.CheckDecode(codec._type.Size == _size);
                    else
                        Contracts.CheckDecode(_size >= 0);
                    if (!FixedLength)
                        _lengths = ReadIntArray(_numVectors);

                    // The counts of all such vectors.
                    _counts = ReadIntArray(_numVectors);
                    int numIndices = Reader.ReadInt32();
                    Contracts.CheckDecode(numIndices >= 0);
                    _indices = ReadIntArray(numIndices);

                    // Validate the number of indices
                    int totalItems = 0;
                    for (int i = 0, ii = 0; i < _numVectors; ++i)
                    {
                        int count = _counts[i];
                        int len = FixedLength ? _size : _lengths[i];
                        Contracts.CheckDecode(len >= 0);
                        if (count < 0) // dense
                        {
                            Contracts.CheckDecode(count == -1);
                            count = len;
                            totalItems += count;
                        }
                        else // sparse
                        {
                            Contracts.CheckDecode(count < len);
                            numIndices += count;
                            totalItems += count;

                            // Check the correctness of the indices.
                            int prev = -1;
                            count += ii;
                            for (int j = ii; j < count; j++)
                            {
                                Contracts.CheckDecode(prev < _indices[j]);
                                prev = _indices[j];
                            }
                            ii = count;
                            Contracts.CheckDecode(prev < len);
                        }
                    }

                    // Get a buffer.
                    var values = codec._bufferPool.Get();
                    Utils.EnsureSize(ref values, totalItems, false);
                    using (var reader = codec._innerCodec.OpenReader(stream, totalItems))
                        reader.Read(values, 0, totalItems);
                    _values = values;
                    _vectorIndex = -1;
                }

                public override void Dispose()
                {
                    if (!_disposed)
                    {
                        _codec._bufferPool.Return(_values);
                        _codec._intBufferPool.Return(_counts);
                        _codec._intBufferPool.Return(_indices);
                        if (_lengths != null)
                            _codec._intBufferPool.Return(_lengths);
                        _disposed = true;
                    }
                    base.Dispose();
                }

                private int[] ReadIntArray(int count)
                {
                    int[] values = _codec._intBufferPool.Get();
                    Utils.EnsureSize(ref values, count, false);
                    for (int i = 0; i < count; ++i)
                        values[i] = Reader.ReadInt32();
                    return values;
                }

                public override void MoveNext()
                {
                    Contracts.Assert(_vectorIndex < _numVectors - 1, "already consumed all vectors");
                    if (_vectorIndex >= 0)
                    {
                        // We are on a vector. Skip to the next vector.
                        int count = _counts[_vectorIndex];
                        if (count < 0)
                        {
                            _valuesOffset += FixedLength ? _size : _lengths[_vectorIndex];
                        }
                        else
                        {
                            _indicesOffset += count;
                            _valuesOffset += count;
                        }
                    }
                    _vectorIndex++;
                }

                public override void Get(ref VBuffer<T> value)
                {
                    Contracts.Assert(_vectorIndex >= 0, "have not moved in");
                    int length = FixedLength ? _size : _lengths[_vectorIndex];
                    int count = _counts[_vectorIndex];

                    if (count < 0)
                    {
                        // dense
                        var editor = VBufferEditor.Create(ref value, length);
                        if (length > 0)
                        {
                            _values.AsSpan(_valuesOffset, length)
                                .CopyTo(editor.Values);
                        }
                        value = editor.Commit();
                    }
                    else
                    {
                        // sparse
                        var editor = VBufferEditor.Create(ref value, length, count);
                        if (count > 0)
                        {
                            _values.AsSpan(_valuesOffset, count)
                                .CopyTo(editor.Values);
                            _indices.AsSpan(_indicesOffset, count)
                                .CopyTo(editor.Indices);
                        }
                        value = editor.Commit();
                    }
                }
            }
        }

        private bool GetVBufferCodec(Stream definitionStream, out IValueCodec codec)
        {
            // The first value in the definition stream will be the internal codec.
            IValueCodec innerCodec;
            if (!TryReadCodec(definitionStream, out innerCodec))
            {
                codec = default(IValueCodec);
                return false;
            }
            // From this internal codec, get the VBuffer type of the codec we will return.
            var itemType = innerCodec.Type as PrimitiveType;
            Contracts.CheckDecode(itemType != null);
            // Following the internal type definition is the dimensions.
            VectorType type;
            using (BinaryReader reader = OpenBinaryReader(definitionStream))
            {
                var dims = reader.ReadIntArray();
                if (Utils.Size(dims) > 0)
                {
                    foreach (int d in dims)
                        Contracts.CheckDecode(d >= 0);
                    type = new VectorType(itemType, dims);
                }
                else
                {
                    // In prior times, in the case where the VectorType was of single rank, *and* of unknown length,
                    // then the vector type would be considered to have a dimension count of 0, for some reason.
                    // This can no longer occur, but in the case where we read an older file we have to account for
                    // the fact that nothing may have been written.
                    type = new VectorType(itemType);
                }
            }
            // Next create the vbuffer codec.
            Type codecType = typeof(VBufferCodec<>).MakeGenericType(itemType.RawType);
            codec = (IValueCodec)Activator.CreateInstance(codecType, this, type, innerCodec);
            return true;
        }

        private bool GetVBufferCodec(VectorType type, out IValueCodec codec)
        {
            ColumnType itemType = type.ItemType;
            // First create the element codec.
            IValueCodec innerCodec;
            if (!TryGetCodec(itemType, out innerCodec))
            {
                codec = default(IValueCodec);
                return false;
            }
            // Next create the vbuffer codec.
            Type codecType = typeof(VBufferCodec<>).MakeGenericType(itemType.RawType);
            codec = (IValueCodec)Activator.CreateInstance(codecType, this, type, innerCodec);
            return true;
        }

        private sealed class KeyCodecOld<T> : IValueCodec<T>
        {
            // *** Binary block format ***
            // Identical to UnsafeTypeCodec, packed bytes of little-endian values.

            private readonly CodecFactory _factory;
            private readonly KeyType _type;
            // We rely on a more basic value codec to do the actual saving and loading.
            private readonly IValueCodec<T> _innerCodec;

            public string LoadName { get { return "Key"; } }

            public ColumnType Type { get { return _type; } }

            public KeyCodecOld(CodecFactory factory, KeyType type, IValueCodec<T> innerCodec)
            {
                Contracts.AssertValue(factory);
                Contracts.AssertValue(type);
                Contracts.AssertValue(innerCodec);
                Contracts.Assert(type.RawType == typeof(T));
                Contracts.Assert(innerCodec.Type.RawType == type.RawType);
                _factory = factory;
                _type = type;
                _innerCodec = innerCodec;
            }

            public int WriteParameterization(Stream stream)
            {
                int total = _factory.WriteCodec(stream, _innerCodec);
                using (BinaryWriter writer = _factory.OpenBinaryWriter(stream))
                {
                    writer.Write(_type.Count);
                    total += sizeof(ulong);
                }
                return total;
            }

            // REVIEW: There is something a little bit troubling here. If someone, say,
            // produces a column on KeyType(I4, 4) and then returns 10 as a value in
            // that column, that's obviously a violation of the type, and lots of things
            // downstream may complain, but it is a "valid" cursor in that it produces values
            // and does not throw. So from that perspective of the codecs and their users being
            // common and indifferent carriers, it's not clear tha these codecs should take on the
            // responsibility for validating the input. On the *other* hand, if we know that we
            // wrote valid data, when reading it back from a stream should we not take advantage
            // of this, to validate the correctness of the decoding? On the other other hand, is
            // validating the correctness for the decoding of things like streams any less urgent?

            public IValueWriter<T> OpenWriter(Stream stream)
            {
                return _innerCodec.OpenWriter(stream);
            }

            public IValueReader<T> OpenReader(Stream stream, int items)
            {
                return _innerCodec.OpenReader(stream, items);
            }
        }

        private bool GetKeyCodecOld(Stream definitionStream, out IValueCodec codec)
        {
            // The first value in the definition stream will be the internal codec.
            IValueCodec innerCodec;
            if (!TryReadCodec(definitionStream, out innerCodec))
            {
                codec = default;
                return false;
            }
            // Construct the key type.
            var itemType = innerCodec.Type as PrimitiveType;
            Contracts.CheckDecode(itemType != null);
            Contracts.CheckDecode(KeyType.IsValidDataType(itemType.RawType));
            KeyType type;
            using (BinaryReader reader = OpenBinaryReader(definitionStream))
            {
                bool contiguous = reader.ReadBoolByte();
                ulong min = reader.ReadUInt64();
                int count = reader.ReadInt32();

                // Since we no longer support the notion of min != 0 or non contiguous values we throw in that case.
                Contracts.CheckDecode(min == 0);
                Contracts.CheckDecode(0 <= count);
                Contracts.CheckDecode((ulong)count <= itemType.GetRawKind().ToMaxInt());
                Contracts.CheckDecode(contiguous);

                // Since we removed the notion of unknown cardinality (count == 0), we map to the maximum value.
                if (count == 0)
                    type = new KeyType(itemType.RawType, itemType.RawType.ToMaxInt());
                else
                    type = new KeyType(itemType.RawType, count);
            }
            // Next create the key codec.
            Type codecType = typeof(KeyCodecOld<>).MakeGenericType(itemType.RawType);
            codec = (IValueCodec)Activator.CreateInstance(codecType, this, type, innerCodec);
            return true;
        }

        private bool GetKeyCodecOld(ColumnType type, out IValueCodec codec)
        {
            if (!(type is KeyType))
                throw Contracts.ExceptParam(nameof(type), "type must be a key type");
            // Create the internal codec the key codec will use to do the actual reading/writing.
            IValueCodec innerCodec;
            if (!TryGetCodec(NumberType.FromKind(type.GetRawKind()), out innerCodec))
            {
                codec = default;
                return false;
            }
            // Next create the key codec.
            Type codecType = typeof(KeyCodecOld<>).MakeGenericType(type.RawType);
            codec = (IValueCodec)Activator.CreateInstance(codecType, this, type, innerCodec);
            return true;
        }

        private sealed class KeyCodec<T> : IValueCodec<T>
        {
            // *** Binary block format ***
            // Identical to UnsafeTypeCodec, packed bytes of little-endian values.

            private readonly CodecFactory _factory;
            private readonly KeyType _type;
            // We rely on a more basic value codec to do the actual saving and loading.
            private readonly IValueCodec<T> _innerCodec;

            public string LoadName { get { return "Key2"; } }

            public ColumnType Type { get { return _type; } }

            public KeyCodec(CodecFactory factory, KeyType type, IValueCodec<T> innerCodec)
            {
                Contracts.AssertValue(factory);
                Contracts.AssertValue(type);
                Contracts.AssertValue(innerCodec);
                Contracts.Assert(type.RawType == typeof(T));
                Contracts.Assert(innerCodec.Type.RawType == type.RawType);
                _factory = factory;
                _type = type;
                _innerCodec = innerCodec;
            }

            public int WriteParameterization(Stream stream)
            {
                int total = _factory.WriteCodec(stream, _innerCodec);
                using (BinaryWriter writer = _factory.OpenBinaryWriter(stream))
                {
                    writer.Write(_type.Count);
                    total += sizeof(ulong);
                }
                return total;
            }

            // REVIEW: There is something a little bit troubling here. If someone, say,
            // produces a column on KeyType(I4, 4) and then returns 10 as a value in
            // that column, that's obviously a violation of the type, and lots of things
            // downstream may complain, but it is a "valid" cursor in that it produces values
            // and does not throw. So from that perspective of the codecs and their users being
            // common and indifferent carriers, it's not clear tha these codecs should take on the
            // responsibility for validating the input. On the *other* hand, if we know that we
            // wrote valid data, when reading it back from a stream should we not take advantage
            // of this, to validate the correctness of the decoding? On the other other hand, is
            // validating the correctness for the decoding of things like streams any less urgent?

            public IValueWriter<T> OpenWriter(Stream stream)
            {
                return _innerCodec.OpenWriter(stream);
            }

            public IValueReader<T> OpenReader(Stream stream, int items)
            {
                return _innerCodec.OpenReader(stream, items);
            }
        }

        private bool GetKeyCodec(Stream definitionStream, out IValueCodec codec)
        {
            // The first value in the definition stream will be the internal codec.
            IValueCodec innerCodec;
            if (!TryReadCodec(definitionStream, out innerCodec))
            {
                codec = default;
                return false;
            }
            // Construct the key type.
            var itemType = innerCodec.Type as PrimitiveType;
            Contracts.CheckDecode(itemType != null);
            Contracts.CheckDecode(KeyType.IsValidDataType(itemType.RawType));
            KeyType type;
            using (BinaryReader reader = OpenBinaryReader(definitionStream))
            {
                ulong count = reader.ReadUInt64();

                Contracts.CheckDecode(0 < count);
                Contracts.CheckDecode(count <= itemType.RawType.ToMaxInt());

                type = new KeyType(itemType.RawType, count);
            }
            // Next create the key codec.
            Type codecType = typeof(KeyCodec<>).MakeGenericType(itemType.RawType);
            codec = (IValueCodec)Activator.CreateInstance(codecType, this, type, innerCodec);
            return true;
        }

        private bool GetKeyCodec(ColumnType type, out IValueCodec codec)
        {
            if (!(type is KeyType))
                throw Contracts.ExceptParam(nameof(type), "type must be a key type");
            // Create the internal codec the key codec will use to do the actual reading/writing.
            IValueCodec innerCodec;
            if (!TryGetCodec(NumberType.FromType(type.RawType), out innerCodec))
            {
                codec = default;
                return false;
            }
            // Next create the key codec.
            Type codecType = typeof(KeyCodec<>).MakeGenericType(type.RawType);
            codec = (IValueCodec)Activator.CreateInstance(codecType, this, type, innerCodec);
            return true;
        }
    }
}
