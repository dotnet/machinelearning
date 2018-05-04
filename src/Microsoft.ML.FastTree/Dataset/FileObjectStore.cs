// -----------------------------------------------------------------------
// <copyright file="FileObjectStore.cs" company="Microsoft">
//     Copyright (C) All Rights Reserved
// </copyright>
// -----------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Linq;
using System.Runtime.Serialization;
using System.Threading;

namespace Microsoft.ML.Runtime.FastTree.Internal
{
#if !NO_STORE
    /// <summary>
    /// A basic object store that uses generic class for serialization
    /// </summary>
    /// <typeparam name="T">Class implementing IFormatter interface</typeparam>
    public class FileObjectStore<T> where T : IFormatter, new()
    {
        /// <summary>
        /// A dictionary containing all the file store instances
        /// </summary>
        private static Dictionary<string, FileObjectStore<T>> instances = new Dictionary<string, FileObjectStore<T>>();

        /// <summary>
        /// The memory mapped file used to get objects from the file store
        /// </summary>
        private MemoryMappedFile objectCacheMemoryMappedFile;

        /// <summary>
        /// The file stream used to write objects to the file store
        /// </summary>
        private FileStream objectCacheFileStream;

        /// <summary>
        /// The formatter object used to serialize and deserialize the file store objects
        /// </summary>
        private T formatter;

        /// <summary>
        /// The file name backing up the object store
        /// </summary>
        private string fileStreamName = null;

        /// <summary>
        /// A reader/writer lock to synchronize access to the store
        /// </summary>
        private ReaderWriterLockSlim cacheLock = new ReaderWriterLockSlim();

        /// <summary>
        /// Gets the current size of the object store
        /// </summary>
        public long Size { get; private set; }

        /// <summary>
        /// Gets the file store initialization state
        /// </summary>
        public bool Initialized
        {
            get
            {
                return this.objectCacheMemoryMappedFile != null || this.objectCacheFileStream != null;
            }
        }

        /// <summary>
        /// Gets object store default instance. The default instance is the first object store created in the process
        /// </summary>
        /// <returns>Null if no object store has been created. Otherwise returns the object store default instance</returns>
        public static FileObjectStore<T> GetDefaultInstance()
        {
            return GetInstance(null);
        }

        /// <summary>
        /// Returns an object store by passing an instance name.
        /// </summary>
        /// <param name="instanceName">Object store instance name. If null is passed, then it returns the default object store</param>
        /// <returns>The file object store with the instance name. Null if instance name does not exist</returns>
        public static FileObjectStore<T> GetInstance(string instanceName)
        {
            FileObjectStore<T> fileObjectStore = null;

            lock (instances)
            {
                if (string.IsNullOrEmpty(instanceName) && instances.Count > 0)
                {
                    fileObjectStore = instances.ElementAt(0).Value;
                }
                else if (!string.IsNullOrEmpty(instanceName) && !instances.TryGetValue(instanceName, out fileObjectStore))
                {
                    fileObjectStore = new FileObjectStore<T>();
                    fileObjectStore.InitializeAsFileStream(instanceName);
                    instances.Add(instanceName, fileObjectStore);
                }

                return fileObjectStore;
            }
        }

        /// <summary>
        /// Prevents further changes in the object store. Needs to be called before reading objects from the store
        /// </summary>
        public void SealObjectStore()
        {
            if (this.objectCacheFileStream != null)
            {
                this.objectCacheFileStream.Close();
            }

            this.objectCacheMemoryMappedFile = MemoryMappedFile.CreateFromFile(this.fileStreamName, FileMode.OpenOrCreate);
        }

        /// <summary>
        /// Writes an object to the object store
        /// </summary>
        /// <param name="offset">Offset of the object in the store</param>
        /// <param name="source">Object to write to the store</param>
        /// <returns>Size of the object in the store. 0 if object store has not been initialized</returns>
        public long WriteObject(ref long offset, object source)
        {
            long size = 0;
            Stream viewStream = null;

            this.cacheLock.EnterWriteLock();

            if (!this.Initialized)
            {
                return 0;
            }

            try
            {
                offset = this.Size;

                viewStream = (this.objectCacheMemoryMappedFile != null) ? this.objectCacheMemoryMappedFile.CreateViewStream() : (Stream)this.objectCacheFileStream;
                viewStream.Position = offset;
                this.formatter.Serialize(viewStream, source);
                size = viewStream.Length - offset;

                this.Size += size;
            }
            finally
            {
                this.cacheLock.ExitWriteLock();

                if (viewStream != null && viewStream is MemoryMappedViewStream)
                {
                    viewStream.Dispose();
                }
            }

            return size;
        }

        /// <summary>
        /// Reads an object from the object store
        /// </summary>
        /// <param name="offset">Offset of the object in the store</param>
        /// <param name="size">Size of the object in the store</param>
        /// <returns>The object from the store</returns>
        public object ReadObject(long offset, long size)
        {
            try
            {
                this.cacheLock.EnterReadLock();

                if (this.objectCacheMemoryMappedFile == null)
                {
                    return null;
                }

                using (Stream viewStream = this.objectCacheMemoryMappedFile.CreateViewStream(offset, size))
                {
                    return this.formatter.Deserialize(viewStream);
                }
            }
            finally
            {
                this.cacheLock.ExitReadLock();
            }
        }

        /// <summary>
        /// Close the object store. Deletes associated object store file
        /// </summary>
        public void Close()
        {
            this.cacheLock.EnterWriteLock();

            if (this.objectCacheMemoryMappedFile != null)
            {
                this.objectCacheMemoryMappedFile.Dispose();
                this.objectCacheMemoryMappedFile = null;
            }

            if (this.objectCacheFileStream != null)
            {
                this.objectCacheFileStream.Close();

                if (File.Exists(this.fileStreamName))
                {
                    File.Delete(this.fileStreamName);
                }
            }

            this.cacheLock.ExitWriteLock();
        }

        /// <summary>
        /// Initializes file stream used as store
        /// </summary>
        /// <param name="fileName">Object store file name</param>
        private void InitializeAsFileStream(string fileName)
        {
            this.formatter = new T();
            this.fileStreamName = fileName;

            // Initialize event handlers to properly delete the files in case of unexpected shutdown
            AppDomain.CurrentDomain.UnhandledException += this.CurrentDomain_UnhandledException;
            AppDomain.CurrentDomain.DomainUnload += this.CurrentDomain_ProcessExit;
            AppDomain.CurrentDomain.ProcessExit += this.CurrentDomain_ProcessExit;
            Console.CancelKeyPress += this.Console_CancelKeyPress;

            this.objectCacheFileStream = new FileStream(this.fileStreamName, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.ReadWrite);
        }

        #region Process shutdown event handlers

        /// <summary>
        /// Process exit and Domain Unload event handler
        /// </summary>
        /// <param name="sender">The object raising the event</param>
        /// <param name="e">The event arguments</param>
        private void CurrentDomain_ProcessExit(object sender, EventArgs e)
        {
            this.Close();
        }

        /// <summary>
        /// Cancel key press event handler
        /// </summary>
        /// <param name="sender">The object raising the event</param>
        /// <param name="e">The event arguments</param>
        private void Console_CancelKeyPress(object sender, ConsoleCancelEventArgs e)
        {
            this.Close();
            Environment.Exit(-1);
        }

        /// <summary>
        /// AppDomain unhandled exception event handler
        /// </summary>
        /// <param name="sender">The object raising the event</param>
        /// <param name="e">The event arguments</param>
        private void CurrentDomain_UnhandledException(object sender, UnhandledExceptionEventArgs e)
        {
            this.Close();
            throw Contracts.Except(e.ExceptionObject as Exception, "Unhandled Exception detected");
        }
        #endregion
    }

    /// <summary>
    /// This class is used to enable serialization of IntArray objects
    /// </summary>
    public class IntArrayFormatter : IFormatter
    {
        /// <summary>
        /// Gets or sets the Binder property
        /// </summary>
        public SerializationBinder Binder
        {
            get
            {
                throw Contracts.ExceptNotImpl();
            }

            set
            {
                throw Contracts.ExceptNotImpl();
            }
        }

        /// <summary>
        /// Gets or sets the Streaming context property
        /// </summary>
        public StreamingContext Context
        {
            get
            {
                throw Contracts.ExceptNotImpl();
            }

            set
            {
                throw Contracts.ExceptNotImpl();
            }
        }

        /// <summary>
        /// Gets or sets the Surrogate selector property
        /// </summary>
        public ISurrogateSelector SurrogateSelector
        {
            get
            {
                throw Contracts.ExceptNotImpl();
            }

            set
            {
                throw Contracts.ExceptNotImpl();
            }
        }

        /// <summary>
        /// Deserializes an object from the input stream
        /// </summary>
        /// <param name="serializationStream">The serialization stream input</param>
        /// <returns>The IntArray object</returns>
        public object Deserialize(Stream serializationStream)
        {
            byte[] objectBuffer = BufferPoolManager.TakeBuffer<byte>((int)serializationStream.Length);

            int bytesRead = 0;
            int currentOffset = 0;
            const int BufferSize = 409600;

            while (currentOffset < serializationStream.Length)
            {
                bytesRead = serializationStream.Read(objectBuffer, currentOffset, Math.Min(BufferSize, (int)serializationStream.Length - currentOffset));
                currentOffset += bytesRead;
            }

            IntArray a = this.Deserialize(objectBuffer);

            BufferPoolManager.ReturnBuffer(ref objectBuffer);

            return a;
        }

        /// <summary>
        /// Serializes an object into an input stream
        /// </summary>
        /// <param name="serializationStream">The input stream to write the serialized object into</param>
        /// <param name="graph">The object to serialize</param>
        public void Serialize(Stream serializationStream, object graph)
        {
            byte[] buffer = this.Serialize((IntArray)graph);
            serializationStream.Write(buffer, 0, buffer.Length);
        }

        /// <summary>
        /// Serializes an IntArray into a byte array
        /// </summary>
        /// <param name="input">The IntArray object</param>
        /// <returns>The serialized byte array object</returns>
        private byte[] Serialize(IntArray input)
        {
            int position = 0;
            byte[] buffer = new byte[input.SizeInBytes()];

            input.ToByteArray(buffer, ref position);

            return buffer;
        }

        /// <summary>
        /// Deserializes an IntArray object from a byte array
        /// </summary>
        /// <param name="buffer">The byte array object representation</param>
        /// <returns>The IntArray object</returns>
        private IntArray Deserialize(byte[] buffer)
        {
            int position = 0;
            return IntArray.New(buffer, ref position, true);
        }
    }
#endif
}
