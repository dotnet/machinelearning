// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;

namespace Microsoft.ML.Runtime.FastTree.Internal
{
    /// <summary>
    /// Implements a paging mechanism on indexed objects.
    /// </summary>
    public class MappedObjectPool<T> where T : class
    {
        private T[] _pool;
        private int[] _map;
        private int[] _inverseMap;
        private int[] _lastAccessTime;
        private int _time;

        /// <summary>
        /// Initializes a new instance of the <see cref="MappedObjectPool{T}"/> class.
        /// </summary>
        /// <param name="pool">A pool of objects on top of which the paging mechanism is built</param>
        /// <param name="maxIndex">The maximal index</param>
        public MappedObjectPool(T[] pool, int maxIndex)
        {
            _pool = pool;
            _map = Enumerable.Range(0, maxIndex).Select(x => -1).ToArray(maxIndex);
            _inverseMap = Enumerable.Range(0, _pool.Length).Select(x => -1).ToArray(_pool.Length);
            _lastAccessTime = new int[_pool.Length];
            _time = 0;
        }

        /// <summary>
        /// If the given index maps to a cached object, that object is retrieved and the return value is true.
        /// If the index is not cached, an object from the pool is retrieved (possibly paging-out the least-recently used) and the return value is false.
        /// </summary>
        /// <param name="index">The requested index</param>
        /// <param name="obj">The retrieved object</param>
        /// <returns>true if the index was found, false if a new object was assigned from the pool</returns>
        public bool Get(int index, out T obj)
        {
            // obj is cached
            if (_map[index] >= 0)
            {
                int position = _map[index];
                _lastAccessTime[position] = ++_time;
                obj = _pool[position];
                return true;
            }

            // page fault - steal someone else's obj
            else
            {
                int stealPosition = _lastAccessTime.ArgMin();

                _lastAccessTime[stealPosition] = ++_time;
                if (_inverseMap[stealPosition] >= 0)
                    _map[_inverseMap[stealPosition]] = -1;
                _map[index] = stealPosition;
                _inverseMap[stealPosition] = index;
                obj = _pool[stealPosition];
                return false;
            }
        }

        public void Steal(int fromIndex, int toIndex)
        {
            if (_map[fromIndex] < 0)
                return;

            int stealPosition = _map[toIndex] = _map[fromIndex];
            _lastAccessTime[stealPosition] = ++_time;
            _inverseMap[stealPosition] = toIndex;
            _map[fromIndex] = -1;
        }

        /// <summary>
        /// Resets the MappedObjectPool
        /// </summary>
        public void Reset()
        {
            Array.Clear(_lastAccessTime, 0, _lastAccessTime.Length);
            _time = 0;
            for (int i = 0; i < _map.Length; ++i)
                _map[i] = -1;
            for (int i = 0; i < _inverseMap.Length; ++i)
                _inverseMap[i] = -1;
        }
    }
}
