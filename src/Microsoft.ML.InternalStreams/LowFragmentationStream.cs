// owner: rragno

#define LZMA_PLAIN
#define UNBUFFERED

using System;
using System.IO;
using System.Collections;
using System.Collections.Specialized;
using System.Text;
using System.Threading;
using System.Runtime.InteropServices;
//using System.Xml;
//using System.Data;
//using System.Data.OleDb;

namespace Microsoft.ML.Runtime.Internal.IO
{
    // Should this start out very small for tiny files?

    /// <summary>
    /// This is a writable FileStream that reduces the fragmentation by
    /// extending the output in chunks.
    /// </summary>
    public class LowFragmentationStream : FileStream
    {
        #region Static Settings

        private static long _minExtension = 50L * 1024L * 1024L;
        private static long _maxExtension = 2L * 1024L * 1024L * 1024L;
        private static float _extensionFactor = 0.5F;

        /// <summary>
        /// Get or set the minimum size increment by which a file will be extended, in bytes.
        /// Defaults to 50MB.
        /// </summary>
        public static long MinExtension
        {
            get { return _minExtension; }
            set
            {
                if (value <= 0)
                    value = 50L * 1024L * 1024L;
                _minExtension = value;
            }
        }

        /// <summary>
        /// Get or set the maximum size increment by which a file will be extended, in bytes.
        /// Defaults to 2GB.
        /// </summary>
        public static long MaxExtension
        {
            get { return _maxExtension; }
            set
            {
                if (value <= 0)
                    value = 2L * 1024L * 1024L * 1024L;
                _maxExtension = value;
            }
        }

        /// <summary>
        /// Get or set the factor by which a file wll be extended when needed.
        /// Defaults to 0.5, and must be greater than 0.0 and less than or equal to 1.0.
        /// </summary>
        public static float ExtensionFactor
        {
            get { return _extensionFactor; }
            set
            {
                if (value <= 0.0F)
                    value = 0.5F;
                if (value > 1.0F)
                    value = 1.0F;
                _extensionFactor = value;
            }
        }

        #endregion

        /// <summary>
        /// This is really equal to base.Length while writing, but base.Length
        /// is less efficient.
        /// </summary>
        private long _extendedLength;
        /// <summary>
        /// If positive, this represents the true length of the file.
        /// Otherwise, that is the Position value.
        /// </summary>
        private long _length;
        /// <summary>
        /// Hopefully, this is always positive, and is the cached Position for efficiency.
        /// </summary>
        private long _position;
        private bool _closed = false;

        /// <summary>
        /// Construct a writable Stream outputting to the given file.
        /// </summary>
        /// <param name="fileName">the name of the file to write to</param>
        ///
        /// <exception cref="System.IO.IOException">An I/O error occurs</exception>
        /// <exception cref="System.Security.SecurityException">The caller does not have the required permission.</exception>
        /// <exception cref="System.IO.DirectoryNotFoundException">The specified path is invalid, such as being on an unmapped drive.</exception>
        /// <exception cref="System.UnauthorizedAccessException">The access needed is not permitted by the operating system for the specified <paramref name="fileName" />.</exception>
        /// <exception cref="System.IO.PathTooLongException">The specified path, file name, or both exceed the system-defined maximum length.</exception>
        public LowFragmentationStream(string fileName)
            : this(fileName, false)
        {
        }
        /// <summary>
        /// Construct a writable Stream outputting to the given file.
        /// </summary>
        /// <param name="fileName">the name of the file to write to</param>
        /// <param name="bufferSize">the size of the buffer to use, in bytes</param>
        ///
        /// <exception cref="System.ArgumentOutOfRangeException"><paramref name="bufferSize" /> is negative or zero.</exception>
        /// <exception cref="System.IO.IOException">An I/O error occurs</exception>
        /// <exception cref="System.Security.SecurityException">The caller does not have the required permission.</exception>
        /// <exception cref="System.IO.DirectoryNotFoundException">The specified path is invalid, such as being on an unmapped drive.</exception>
        /// <exception cref="System.UnauthorizedAccessException">The access needed is not permitted by the operating system for the specified <paramref name="fileName" />.</exception>
        /// <exception cref="System.IO.PathTooLongException">The specified path, file name, or both exceed the system-defined maximum length.</exception>
        public LowFragmentationStream(string fileName, int bufferSize)
            : this(fileName, false, bufferSize)
        {
        }
        /// <summary>
        /// Construct a writable Stream outputting to the given file.
        /// </summary>
        /// <param name="fileName">the name of the file to write to</param>
        /// <param name="append">if true, append to the file; otherwise, overwrite</param>
        ///
        /// <exception cref="System.IO.FileNotFoundException">Append is specified, and the file cannot be found.</exception>
        /// <exception cref="System.IO.IOException">An I/O error occurs</exception>
        /// <exception cref="System.Security.SecurityException">The caller does not have the required permission.</exception>
        /// <exception cref="System.IO.DirectoryNotFoundException">The specified path is invalid, such as being on an unmapped drive.</exception>
        /// <exception cref="System.UnauthorizedAccessException">The access needed is not permitted by the operating system for the specified <paramref name="fileName" />.</exception>
        /// <exception cref="System.IO.PathTooLongException">The specified path, file name, or both exceed the system-defined maximum length.</exception>
        public LowFragmentationStream(string fileName, bool append)
            : this(fileName, append, 64 * 1024)
        {
        }
        /// <summary>
        /// Construct a writable Stream outputting to the given file.
        /// </summary>
        /// <param name="fileName">the name of the file to write to</param>
        /// <param name="append">if true, append to the file; otherwise, overwrite</param>
        /// <param name="bufferSize">the size of the buffer to use, in bytes</param>
        ///
        /// <exception cref="System.ArgumentOutOfRangeException"><paramref name="bufferSize" /> is negative or zero.</exception>
        /// <exception cref="System.IO.FileNotFoundException">Append is specified, and the file cannot be found.</exception>
        /// <exception cref="System.IO.IOException">An I/O error occurs</exception>
        /// <exception cref="System.Security.SecurityException">The caller does not have the required permission.</exception>
        /// <exception cref="System.IO.DirectoryNotFoundException">The specified path is invalid, such as being on an unmapped drive.</exception>
        /// <exception cref="System.UnauthorizedAccessException">The access needed is not permitted by the operating system for the specified <paramref name="fileName" />.</exception>
        /// <exception cref="System.IO.PathTooLongException">The specified path, file name, or both exceed the system-defined maximum length.</exception>
        public LowFragmentationStream(string fileName, bool append, int bufferSize)
            : base(fileName, append ? FileMode.Append : FileMode.Create, FileAccess.Write, FileShare.Read, bufferSize)
        {
            if (append)
            {
                _length = base.Length;
                _position = _length;
                _extendedLength = _length;
                Extend();
            }
            else
            {
                _length = -1;
                _position = 0;
                _extendedLength = 0;
                Extend();
            }
        }

        #region Allocation

        /// <summary>
        /// Set the desired file allocation, preferably without changing the end of file.
        /// </summary>
        /// <param name="length">the size to set, in bytes</param>
        /// <param name="useNT">if true, use the low-level undocumented NT API to allocate, without setting the length</param>
        private void SetAllocation(long length, bool useNT)
        {
            // using NT calls seems unreliable, beyond being undocumented, unsupported, and
            // likely platform dependant.
            //
            // Sometimes, it works perfectly. Other times, it generates roughly the same
            // number of fragments as not doing anything (but can vary). Other times, it
            // generates 10 times as many fragments as not doing anything.
            //
            // It does, at least, leave the file without extra allocated space at all times.

#if !NT_ALLOCATE
            if (!useNT)
            {
                base.SetLength(length);
            }
            else
            {
#endif
                //FILE_ALLOCATION_INFORMATION allocInfo = new FILE_ALLOCATION_INFORMATION(length);
                long allocInfo = length;
                IOUtil.Win32.IO_STATUS_BLOCK status = new IOUtil.Win32.IO_STATUS_BLOCK();  // = IO_STATUS_BLOCK.NullBlock;
                IOUtil.Win32.NtSetInformationFile(
                    base.SafeFileHandle.DangerousGetHandle(),
                    ref status,
                    ref allocInfo,
                    8, //sizeof(FILE_ALLOCATION_INFORMATION),
                    IOUtil.Win32.FILE_INFORMATION_CLASS.FileAllocationInformation);
                // FILE_INFORMATION_CLASS.FileEndOfFileInformation);
#if !NT_ALLOCATE
            }
#endif
        }
        #endregion

        /// <summary>
        /// Remove allocated space that was not written to.
        /// </summary>
        /// <param name="fileName">the file to trim</param>
        /// <returns>true if the file is resized; false otherwise</returns>
        /// <remarks>
        /// This is not normally needed. The LowFragmentationStream trims itself upon finalization.
        /// However, if the runtime is terminated abruptly, it is possible for a file to be left
        /// with unused space. In that case, this method will remove the unused space.
        /// </remarks>
        /// <exception cref="System.IO.FileNotFoundException">Append is specified, and the file cannot be found.</exception>
        /// <exception cref="System.IO.IOException">An I/O error occurs</exception>
        /// <exception cref="System.Security.SecurityException">The caller does not have the required permission.</exception>
        /// <exception cref="System.IO.DirectoryNotFoundException">The specified path is invalid, such as being on an unmapped drive.</exception>
        /// <exception cref="System.UnauthorizedAccessException">The access needed is not permitted by the operating system for the specified <paramref name="fileName" />.</exception>
        /// <exception cref="System.IO.PathTooLongException">The specified path, file name, or both exceed the system-defined maximum length.</exception>
        public static bool TrimOverextension(string fileName)
        {
            long length = -1;
            using (FileStream s = new FileStream(fileName, FileMode.Open, FileAccess.Read))
            {
                byte[] buffer = new byte[1024 * 1024];
                s.Seek(-1, SeekOrigin.End);
                int lastByte = s.ReadByte();
                //Console.WriteLine("last: " + lastByte);
                if (lastByte == 0)
                {
                    long end = s.Length;
                    while (length < 0 && end > 0)
                    {
                        end -= buffer.Length;
                        if (end < 0)
                            end = 0;
                        s.Seek(end, SeekOrigin.Begin);
                        int count = s.Read(buffer, 0, buffer.Length);
                        for (int c = count - 1; c >= 0; c--)
                        {
                            if (buffer[c] != 0)
                            {
                                length = end + c;
                                break;
                            }
                        }
                    }
                }
            }
            if (length < 0)
                return false;
            IOUtil.ResizeFile(fileName, length);
            return true;
        }

        private void Extend()
        {
            // really, using NtSetInformationFile might be better, because it will
            // automatically truncate, but that is ugly and not really supported... ***
            // Also, it seems to cause random fragmentation, unpredictably.
            long pos = Position;
            long ext = (long)(pos * _extensionFactor);
            if (ext < _minExtension)
            {
                ext = _minExtension;
            }
            else if (ext > _maxExtension)
            {
                ext = _maxExtension;
            }
            // this calls out to the system:
            //long len = base.Length;
            long len = _extendedLength;
            _extendedLength = pos + ext;
            if (_extendedLength > len)
            {
                try
                {
                    // attempt to keep the over-allocation down, at the beginning
                    bool useNT = len < _minExtension;
                    SetAllocation(_extendedLength, useNT);
                }
                catch
                {
                    //Console.WriteLine("extend failed!");
                    // ignore? This could mean there is not enough space on disk!
                    // If we leave the extendedLength set high, this instance will
                    // harmlessly believe it has preallocated and just allow the writes
                    // to extend normally... (of course, we could retry at a smaller
                    // increment...)
                    try
                    {
                        ext = pos + _minExtension;
                        base.SetLength(ext);
                        _extendedLength = ext;
                    }
                    catch
                    {
                        // give up
                    }
                }
            }
            else
            {
                //extendedLength = len;
            }
            //Seek(pos, SeekOrigin.Begin);
        }

        private void Truncate()
        {
            long len = _length;
            if (len <= 0)
            {
                len = Position;
            }
            if (len != base.Length)
            {
                base.SetLength(len);
            }
            _extendedLength = len;
        }

        /// <summary>
        /// Clean up the stream - truncate the file length as needed.
        /// </summary>
        ~LowFragmentationStream()
        {
            try
            {
                Truncate();
                //Flush(true);
            }
            catch
            {
                // ignore
            }
        }

        /// <summary>
        /// Close the stream.
        /// </summary>
        public override void Close()
        {
            if (_closed)
                return;
            try
            {
                Truncate();
            }
            catch
            {
                // ignore
            }
            base.Close();
            _closed = true;
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Release the resources used by this instance.
        /// </summary>
        /// <param name="disposing">true if disposing</param>
        protected override void Dispose(bool disposing)
        {
            //Truncate();
            Flush(true);
            base.Dispose(disposing);
        }

        /// <summary>
        /// Write any pending data, without truncating.
        /// </summary>
        /// <remarks>
        /// Note that this does not truncate the file!
        /// </remarks>
        public override void Flush()
        {
            // should this truncate? ***
            Flush(false);
        }
        /// <summary>
        /// Write any pending data and optionally truncate.
        /// </summary>
        /// <param name="truncate">if true, also truncate the file; if false, do not.</param>
        public new void Flush(bool truncate)
        {
            try
            {
                base.Flush();
                if (truncate)
                {
                    Truncate();
                    base.Flush();
                }
            }
            catch
            {
                // ignore??
            }
        }

        /// <summary>
        /// Extend the capacity to be at least a certain number of bytes.
        /// </summary>
        /// <param name="length">the number of bytes to allocate</param>
        /// <remarks>
        /// This is useful when the approximate or exact size of the output is known.
        /// It will not affect the final length of the file, but it can help with efficiency
        /// and fragmentation.
        /// </remarks>
        public void Reserve(long length)
        {
            //if (length > base.Length)
            if (length > _extendedLength)
            {
                base.SetLength(length);
                _extendedLength = length;
            }
        }

        #region Reading
        //		/// <summary>
        //		/// Read data into the buffer.
        //		/// </summary>
        //		/// <param name="buffer">the buffer to place the data in</param>
        //		/// <param name="offset">the starting index in buffer</param>
        //		/// <param name="count">the maximum number of bytes to read</param>
        //		/// <returns>the number of bytes read</returns>
        //		public override int Read(byte[] buffer, int offset, int count)
        //		{
        //			return base.Read(buffer, offset, count);
        //		}
        //		
        //		/// <summary>
        //		/// Read a single byte.
        //		/// </summary>
        //		/// <returns>the byte read, or a negative number if end of file</returns>
        //		public override int ReadByte()
        //		{
        //			return base.ReadByte();
        //		}
        //
        //		/// <summary>
        //		/// Read data into the buffer.
        //		/// </summary>
        //		/// <param name="buffer">the buffer to place the data in</param>
        //		/// <param name="offset">the starting index in buffer</param>
        //		/// <param name="count">the maximum number of bytes to read</param>
        //		/// <param name="callback">the callback to use</param>
        //		/// <param name="state">the state to use for the callback</param>
        //		/// <returns>the number of bytes read</returns>
        //		/// <exception cref="IOException">Read is positioned out of bounds.</exception>
        //		public override IAsyncResult BeginRead(byte[] buffer, int offset, int count, AsyncCallback callback, object state)
        //		{
        //			return base.BeginRead(buffer, offset, count, callback, state);
        //		}
        //
        //		/// <summary>
        //		/// End an asynchronous read.
        //		/// </summary>
        //		/// <param name="asyncResult">the result</param>
        //		/// <returns>number of bytes read</returns>
        //		public override int EndRead(IAsyncResult asyncResult)
        //		{
        //			return base.EndRead(asyncResult);
        //		}
        #endregion

        /// <summary>
        /// <para>Sets the current position of this stream to the given value.</para>
        /// </summary>
        /// <param name="offset">The byte number to seek to.</param>
        /// <returns>
        /// <para>The new position in the stream.</para>
        /// </returns>
        /// <exception cref="System.IO.IOException">An I/O error occurs.</exception>
        /// <exception cref="System.ArgumentException">Attempted seeking before the beginning of the stream.</exception>
        /// <exception cref="System.ObjectDisposedException">Methods were called after the stream was closed.</exception>
        public long Seek(long offset)
        {
            return Seek(offset, SeekOrigin.Begin);
        }
        /// <summary>
        /// <para>Sets the current position of this stream to the given value.</para>
        /// </summary>
        /// <param name="offset">The point relative to <paramref name="origin" /> to seek to, in bytes.</param>
        /// <param name="origin">Specifies the beginning, the end, or the current position as a reference point for <paramref name="origin" /> , using a value of type <see cref="T:System.IO.SeekOrigin" /> .</param>
        /// <returns>
        /// <para>The new position in the stream.</para>
        /// </returns>
        /// <exception cref="System.IO.IOException">An I/O error occurs.</exception>
        /// <exception cref="System.ArgumentException">Attempted seeking before the beginning of the stream.</exception>
        /// <exception cref="System.ObjectDisposedException">Methods were called after the stream was closed.</exception>
        public override long Seek(long offset, SeekOrigin origin)
        {
            switch (origin)
            {
            case SeekOrigin.Begin:
                break;
            case SeekOrigin.Current:
                offset = Position + offset;
                break;
            case SeekOrigin.End:
                offset = (_length > 0 ? _length : Position) + offset;
                break;
            }

            if (offset != Position)
            {
                Flush(true);
                if (_length < 0 && offset < Position)
                {
                    // now need to record the current length, since it is not Position:
                    _length = Position;
                }
            }
            _position = base.Seek(offset, SeekOrigin.Begin);
            return _position;
        }

        /// <summary>
        /// Set the file length - avoid using this manually except for truncation,
        /// since the size is normally automatically extended.
        /// </summary>
        /// <param name="value">the length to set it to, in bytes</param>
        /// <remarks>
        /// To simply ensure a predicted length, use <see cref="Reserve"/> instead.
        /// </remarks>
        public override void SetLength(long value)
        {
            base.SetLength(value);
            if (Position < value)
            {
                // must remember the true length...
                //Console.WriteLine("SetLength setting length...");
                _length = value;
            }
            else
            {
                // length now moves with position...
                _length = -1;
                _position = base.Position;
            }
            _extendedLength = value;
        }

        #region Writing

        /// <summary>
        /// Write data from the buffer.
        /// </summary>
        /// <param name="buffer">the buffer to read the data from</param>
        /// <param name="offset">the starting index in buffer</param>
        /// <param name="count">the maximum number of bytes to write</param>
        public override void Write(byte[] buffer, int offset, int count)
        {
            // what if it fails?
            //try
            //{
            long next = Position + count;
            if (_length > 0 && next >= _length)
            {
                _length = -1;
            }
            if (next >= _extendedLength)
            {
                Extend();
            }
            _position = -1;
            base.Write(buffer, offset, count);
            _position = next;
            //}
            //catch
            //{
            //    position = base.Position;
            //}
        }

        /// <summary>
        /// Write a single byte.
        /// </summary>
        /// <param name="value">the byte to write</param>
        public override void WriteByte(byte value)
        {
            // what if it fails?
            //try
            //{
            long next = Position + 1;
            if (_length > 0 && next >= _length)
            {
                _length = -1;
            }
            if (next >= _extendedLength)
            {
                Extend();
            }
            _position = -1;
            base.WriteByte(value);
            _position = next;
            //}
            //catch
            //{
            //    position = base.Position;
            //}
        }

        /// <summary>
        /// Begins an asynchronous write.
        /// </summary>
        /// <param name="buffer">The buffer to write data to.</param>
        /// <param name="offset">The zero based byte offset at which to begin writing.</param>
        /// <param name="count">The maximum number of bytes to write.</param>
        /// <param name="callback">The method to be called when the asynchronous write operation is completed.</param>
        /// <param name="stateObject">A user-provided object that distinguishes this particular asynchronous write request from other requests.</param>
        /// <returns>
        /// An <see cref="System.IAsyncResult" /> that references the asynchronous write.
        /// </returns>
        /// <exception cref="System.NotSupportedException">The stream does not support writing.</exception>
        /// <exception cref="System.ObjectDisposedException">The stream is closed.</exception>
        /// <exception cref="System.IO.IOException">An I/O error occurs.</exception>
        public override IAsyncResult BeginWrite(byte[] buffer, int offset, int count, AsyncCallback callback, object stateObject)
        {
            // we could wrap the state object...
            // just call this +1, in case less is written?
            long next = Position + 1;
            if (_length > 0 && next >= _length)
            {
                // should we do this, without EndWrite being called?
                _length = -1;
            }
            if (Position + count >= _extendedLength)
            {
                Extend();
            }
            // we just don't know how much will really be read...
            _position = -1;
            return base.BeginWrite(buffer, offset, count, callback, stateObject);
        }

        /// <summary>
        /// Ends an asynchronous write, blocking until the I/O operation has completed.
        /// </summary>
        /// <param name="asyncResult">The pending asynchronous I/O request.</param>
        /// <exception cref="System.ArgumentNullException"><paramref name="asyncResult" /> is <see langword="null" /> .</exception>
        /// <exception cref="System.ArgumentException">This <see cref="System.IAsyncResult" /> object was not created by calling <see cref="M:System.IO.Stream.BeginWrite(System.Byte[],System.Int32,System.Int32,System.AsyncCallback,System.Object)" /> on this class.</exception>
        /// <exception cref="System.InvalidOperationException"><see cref="System.IO.FileStream.EndWrite(System.IAsyncResult)" /> is called multiple times.</exception>
        public override void EndWrite(IAsyncResult asyncResult)
        {
            // should we sync the position here?
            _position = Position;
            if (_length > 0)
            {
                if (Position >= _length)
                {
                    _length = -1;
                }
            }
            base.EndWrite(asyncResult);
        }

        #endregion

        /// <summary>
        /// Get whether the stream can read.
        /// </summary>
        public override bool CanRead
        {
            get
            {
                // just pass through?
                return base.CanRead;
            }
        }

        //		/// <summary>
        //		/// Get whether the stream can seek - true, but it really cannot.
        //		/// (CanSeek is stupidly checked by the FileStream Position property
        //		/// on get, not just set.)
        //		/// </summary>
        /// <summary>
        /// Get whether the stream can seek.
        /// </summary>
        public override bool CanSeek
        {
            get
            {
                // do we need to force this to true, or can we rely on it?
                //return true;
                return base.CanSeek;
            }
        }

        /// <summary>
        /// Get whether the stream can write.
        /// </summary>
        public override bool CanWrite
        {
            get
            {
                // pass through...
                //return true;
                return base.CanWrite;
            }
        }

        /// <summary>
        /// Get the length of the file.
        /// </summary>
        public override long Length
        {
            get
            {
                if (_length > 0)
                    return _length;
                return Position;
            }
        }

        /// <summary>
        /// Get or set the position in the file.
        /// </summary>
        public override long Position
        {
            get
            {
                // the efficiency concern is over VerifyOSHandlePosition when the handle is exposed...
                //return base.Position;
                if (_position < 0)
                    _position = base.Position;
                return _position;
            }
            set
            {
                //throw new NotSupportedException("LowFragmentationStream cannot seek");
                Seek(value);
            }
        }
    }
}
