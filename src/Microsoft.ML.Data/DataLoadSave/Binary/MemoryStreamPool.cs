// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Threading;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data.IO
{
    internal sealed class MemoryStreamPool
    {
        private readonly ObjectPool<MemoryStream> _memPool;

        public MemoryStreamPool()
        {
            _memPool = new ObjectPool<MemoryStream>();
        }

        public void Return(ref MemoryStream mem)
        {
            // REVIEW: Maybe make special handling of get/return on the object pool easier to embed into the pool itself...
            mem.Position = 0;
            mem.SetLength(0);
            _memPool.Return(mem);
            mem = null;
        }

        public MemoryStream Get()
        {
            MemoryStream mem = _memPool.Get();
            Contracts.Assert(mem.Position == 0);
            Contracts.Assert(mem.Length == 0);
            return mem;
        }
    }

    internal sealed class MemoryStreamCollection
    {
        // The idea is that we split requestors by maximum size.
        private readonly MemoryStreamPool[] _pools;

        public MemoryStreamCollection()
        {
            _pools = new MemoryStreamPool[IndexFor(int.MaxValue) + 1];
        }

        /// <summary>
        /// Given a byte size, returns an appropriate index to <see cref="_pools"/>.
        /// This is a non-decreasing function w.r.t. <paramref name="maxSize"/>.
        /// </summary>
        private static int IndexFor(int maxSize)
        {
            return Math.Max(Utils.IbitHigh((uint)maxSize) - 15, 0);
        }

        public MemoryStreamPool Get(int maxSize)
        {
            Contracts.CheckParam(0 <= maxSize, nameof(maxSize), "Must be positive");
            int index = IndexFor(maxSize);
            Contracts.Assert(0 <= index && index < _pools.Length);
            if (_pools[index] == null)
                Interlocked.CompareExchange(ref _pools[index], new MemoryStreamPool(), null);
            return _pools[index];
        }
    }
}
