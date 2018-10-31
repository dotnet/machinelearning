// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;

namespace Microsoft.ML.Runtime.Data.IO
{
    /// <summary>
    /// A value codec encapsulates implementations capable of writing and reading data of some
    /// type to and from streams. The idea is that one creates a codec using <c>TryGetCodec</c>
    /// on the appropriate <c>ColumnType</c>, then opens multiple writers to write blocks of data
    /// to some stream. The idea is that each writer or reader is called on some "managable chunk"
    /// of data.
    ///
    /// Codecs should be thread safe, though the readers and writers they spawn do not need to
    /// be thread safe.
    /// </summary>
    internal interface IValueCodec
    {
        /// <summary>
        /// This is the codec's identifying name. This is utilized both by the codec factory's
        /// <c>WriteTypeDescription</c> and <c>TryGetCodec</c>, for persisting and recovering
        /// the codec, respectively.
        /// </summary>
        string LoadName { get; }

        /// <summary>
        /// Writes the codec parameterization to the stream. (The parameterization
        /// is the third part of the codec type description, after the name, and length
        /// of the parameterization.)
        /// </summary>
        /// <returns>The number of bytes written to the stream</returns>
        int WriteParameterization(Stream stream);

        /// <summary>
        /// The column type for this codec.
        /// </summary>
        ColumnType Type { get; }
    }

    /// <summary>
    /// The generic value codec.
    /// </summary>
    /// <typeparam name="T">The type for which we can spawn readers and writers.
    /// Note that <c>Type.RawType == typeof(T)</c>.</typeparam>
    internal interface IValueCodec<T> : IValueCodec
    {
        /// <summary>
        /// Returns a writer for this codec, capable of writing a series of values to a block
        /// starting at the current position of the indicated writable stream.
        /// </summary>
        IValueWriter<T> OpenWriter(Stream stream);

        /// <summary>
        /// Returns a reader for this codec, capable of reading a series of values to a block
        /// starting at the current position of the indicated readable stream.
        /// </summary>
        /// <param name="stream">Stream on which we open reader.</param>
        /// <param name="items">The number of items expected to be encoded in the block
        /// starting from the current position of the stream. Implementors should, if
        /// possible, throw if it seems if the block contains a different number of
        /// elements.</param>
        IValueReader<T> OpenReader(Stream stream, int items);
    }

    internal interface IValueWriter : IDisposable
    {
        /// <summary>
        /// Finishes writing to the stream. No further values should be written using the
        /// <c>Write</c> methods. Note that failure to commit does not leave the stream in
        /// a defined state: something or nothing could have already been written to the
        /// stream, and the writer has no facilities to "rewind" whatever writes it may
        /// have performed.
        /// </summary>
        void Commit();

        /// <summary>
        /// Returns an estimate of the total length that would be written to the stream
        /// were we to commit right now. This may be called very often in some circumstances,
        /// so implementors should optimize for speed over accuracy.
        /// </summary>
        long GetCommitLengthEstimate();
    }

    /// <summary>
    /// A value writer on a particular type. The intent is that implementors of this will
    /// be spawned from an <seealso cref="IValueCodec"/>, its write methods called some
    /// number of times to write to the stream, and then <c>Commit</c> will be called when
    /// all values have been written, the stream now being at the end of the written block.
    ///
    /// The intended usage of the value writers is that blocks are composed of some small
    /// number of values (perhaps a few thousand), the idea being that a block is something
    /// that should easily fit in main memory, both for reading and writing. Some writers
    /// take advantage of this to organize their values for more efficient reading.
    /// </summary>
    internal interface IValueWriter<T> : IValueWriter
    {
        /// <summary>
        /// Writes a single value to the writer.
        /// </summary>
        void Write(in T value);

        /// <summary>
        /// Writes an array of values. This should be equivalent to writing each element
        /// singly, though possibly more efficient than such a naive implementation.
        /// </summary>
        void Write(T[] values, int index, int count);
    }

    /// <summary>
    /// A value reader on a particular type. As with writers, implementors of this will be
    /// spawned form an <seealso cref="IValueCodec"/>. Its read methods will be called some
    /// number of times to read from the stream. The read methods should be used to read
    /// precisely the same number of times as was written to the block. if you read more,
    /// then the values returned past the last will be undefined, and in either case the
    /// stream will be left in an undefined state. Implementors may optionally complain in
    /// such a case, but many will not, so outside knowledge should be used by the user
    /// to ensure bad behavior does not happen. (For example, if you have a writer that
    /// just writes packed binary values with no descriptive information, the corresponding
    /// read will have no ability to tell when it is supposed to "end.")
    /// </summary>
    internal interface IValueReader<T> : IDisposable
    {
        /// <summary>
        /// Moves to the next element.
        /// </summary>
        void MoveNext();

        /// <summary>
        /// Gets the current element.
        /// </summary>
        void Get(ref T value);

        /// <summary>
        /// Reads into an array of values. This should be roughly equivalent to calling <c>MoveNext</c>
        /// then <c>Get</c> into an array on each element singly, though possibly more efficient than
        /// such a naive implementation. It may also diverge from that, in that <c>Get</c>'s behavior
        /// before the next <c>MoveNext</c> is undefined when this function is called.
        /// </summary>
        void Read(T[] values, int index, int count);
    }
}
