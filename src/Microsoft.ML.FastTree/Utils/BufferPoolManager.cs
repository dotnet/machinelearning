// -----------------------------------------------------------------------
// <copyright file="BufferPoolManager.cs" company="Microsoft Corporation">
//     Copyright (C) All Rights Reserved
// </copyright>
// -----------------------------------------------------------------------

using Microsoft.ML.Runtime;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace Microsoft.ML.Trainers.FastTree.Internal
{
    /// <summary>
    /// This class enables basic buffer pooling.
    /// It supports different types of buffers and returns buffers of the requested size or larger.
    /// This class was implemented to reduce frequent allocation/deallocation of large buffers which caused fragmentation of the large object heap.
    /// </summary>
    public static class BufferPoolManager
    {
        /// <summary>
        /// The minimum size in bytes for a buffer to be stored in the pool
        /// This is the minimum size in bytes for an object to be stored in the large object heap
        /// </summary>
        private const int MinBufferSizeInBytes = 85000;

        /// <summary>
        /// A dictionary containing all buffer pool types
        /// </summary>
        private static ConcurrentDictionary<Type, SortedList<int, List<Array>>> _bufferPools = new ConcurrentDictionary<Type, SortedList<int, List<Array>>>();

        /// <summary>
        /// Gets a buffer from the pool with at least the same size as passed as input parameter
        /// </summary>
        /// <typeparam name="T">Pool type</typeparam>
        /// <param name="size">Minimum size required</param>
        /// <returns>The buffer requested</returns>
        public static T[] TakeBuffer<T>(int size)
        {
            T[] buffer = null;
            SortedList<int, List<Array>> availableBuffers = null;

            if (!_bufferPools.TryGetValue(typeof(T), out availableBuffers))
            {
                InitializeBufferPool<T>();
                _bufferPools.TryGetValue(typeof(T), out availableBuffers);
            }

            lock (availableBuffers)
            {
                // Try to find an available buffer in the pool with the smallest size required to satisfy the request but not too big
                // If the only available buffer size is more than 20% the requested size, then allocate a new buffer.
                List<Array> buffers = availableBuffers.FirstOrDefault(x => x.Value.Count > 0 && x.Key >= size && (x.Key - size) < (size * 0.2)).Value as List<Array>;

                if (buffers != null && buffers.Count > 0)
                {
                    buffer = (T[])buffers[0];
                    buffer.Initialize();
                    buffers.RemoveAt(0);
                }
                else
                {
                    buffer = new T[size];
                }
            }

            return buffer;
        }

        /// <summary>
        /// Returns a buffer back to the pool.
        /// It only keeps buffers bigger than MaxBufferSizeInBytes = 85K bytes
        /// </summary>
        /// <param name="buffer">The buffer array to add to the pool of buffers</param>
        public static void ReturnBuffer<T>(ref T[] buffer)
            where T : struct
        {
            Contracts.AssertValueOrNull(buffer);

            // Small arrays other than Double should not be allocated in LOH. Avoid storing them in the buffer pool
            if (buffer != null && buffer.Length * Marshal.SizeOf(typeof(T)) >= MinBufferSizeInBytes)
            {
                SortedList<int, List<Array>> availableBuffers = null;
                List<Array> buffers = null;

                if (!_bufferPools.TryGetValue(typeof(T), out availableBuffers))
                {
                    InitializeBufferPool(typeof(T));
                    bool tmp = _bufferPools.TryGetValue(typeof(T), out availableBuffers);
                    Contracts.Assert(tmp);
                }

                lock (availableBuffers)
                {
                    if (!availableBuffers.TryGetValue(buffer.Length, out buffers))
                    {
                        buffers = new List<Array>();
                        availableBuffers.Add(buffer.Length, buffers);
                    }
                    buffers.Add(buffer);
                }
            }
            buffer = null;
        }

        /// <summary>
        /// Releases all available buffers in a specific pool
        /// </summary>
        /// <param name="type">Buffer pool type</param>
        public static void ReleaseAllAvailableBuffers(Type type)
        {
            SortedList<int, List<Array>> availableBuffers = null;

            if (_bufferPools.TryGetValue(type, out availableBuffers))
            {
                lock (availableBuffers)
                {
                    availableBuffers.Clear();
                }
            }
        }

        /// <summary>
        /// Releases all available buffers in all pools
        /// </summary>
        public static void ReleaseAllAvailableBuffers()
        {
            foreach (var type in _bufferPools.Keys)
            {
                ReleaseAllAvailableBuffers(type);
            }
        }

        /// <summary>
        /// Initializes a new buffer pool of a specific type
        /// </summary>
        /// <typeparam name="T">Type of buffer to initialize</typeparam>
        private static void InitializeBufferPool<T>()
        {
            InitializeBufferPool(typeof(T));
        }

        /// <summary>
        /// Initializes a new buffer pool of a specific type
        /// </summary>
        /// <param name="type">Type of buffer to initialize</param>
        private static void InitializeBufferPool(Type type)
        {
            lock (_bufferPools)
            {
                if (!_bufferPools.ContainsKey(type))
                {
                    _bufferPools[type] = new SortedList<int, List<Array>>();
                }
            }
        }
    }
}
