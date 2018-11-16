// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Concurrent;
using System.Threading;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    [BestFriend]
    internal sealed class ObjectPool<T> : ObjectPoolBase<T> where T : class, new()
    {
        protected override T Create()
        {
            return new T();
        }
    }

    [BestFriend]
    internal sealed class MadeObjectPool<T> : ObjectPoolBase<T>
    {
        private readonly Func<T> _maker;

        public MadeObjectPool(Func<T> maker)
        {
            _maker = maker;
        }

        protected override T Create()
        {
            return _maker();
        }
    }

    internal abstract class ObjectPoolBase<T>
    {
        private readonly ConcurrentBag<T> _pool;
        private int _numCreated;

        public int Count => _pool.Count;
        public int NumCreated { get { return _numCreated; } }

        private protected ObjectPoolBase()
        {
            _pool = new ConcurrentBag<T>();
        }

        public T Get()
        {
            T result;
            if (_pool.TryTake(out result))
                return result;
            Interlocked.Increment(ref _numCreated);
            return Create();
        }

        protected abstract T Create();

        public void Return(T item)
        {
            _pool.Add(item);
        }
    }
}
