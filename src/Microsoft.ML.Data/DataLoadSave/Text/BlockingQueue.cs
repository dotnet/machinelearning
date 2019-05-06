// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    // NOTE:
    // This is a temporary workaround for https://github.com/dotnet/corefx/issues/34602 or until TextLoader is rearchitected.
    // BlockingCollection is fairly efficient for blocking producer/consumer scenarios, but it was optimized for scenarios where
    // they're not created/destroyed quickly.  Its CompleteAdding mechanism is implemented internally in such a way that if a
    // taker is currently blocked when CompleteAdding is called, that taker thread will incur an OperationCanceledException that's
    // eaten internally.  If such an exception is only happening rarely, it's not a big deal, but the way TextLoader uses
    // BlockingCollections, they can end up being created and destroyed very frequently, tens of thousands of times during the
    // course of an algorithm like SDCA (when no caching is employed).  That in turn can result in tens of thousands of exceptions
    // getting thrown and caught. While in normal processing even that number of exceptions results in an overhead that's not
    // particularly impactful, things change when a debugger is attached, as that makes the overhead of exceptions several orders
    // of magnitude higher (e.g. 1000x). Until either TextLoader is rearchitected to not create so many BlockingCollections in these
    // situations, or until this implementation detail in BlockingCollection is changed, we use a replacement BlockingQueue implementation,
    // that's similar in nature to BlockingCollection (albeit without many of its bells and whistles) but that specifically doesn't
    // rely on cancellation for its CompleteAdding mechanism, and thus doesn't incur this impactful exception. Other than type name, this
    // type is designed to expose the exact surface area required by the existing usage of BlockingCollection<T>, no more, no less,
    // making it easy to swap in and out.

    /// <summary>Provides a thread-safe queue that supports blocking takes when empty and blocking adds when full.</summary>
    /// <typeparam name="T">Specifies the type of data contained.</typeparam>
    internal sealed class BlockingQueue<T> : IDisposable
    {
        /// <summary>The underlying queue storing all elements.</summary>
        private readonly ConcurrentQueue<T> _queue;
        /// <summary>A semaphore that can be waited on to know when an item is available for taking.</summary>
        private readonly CompletableSemaphore _itemsAvailable;
        /// <summary>A semaphore that can be waited on to know when space is available for adding.</summary>
        private readonly CompletableSemaphore _spaceAvailable;

        /// <summary>Initializes the blocking queue.</summary>
        /// <param name="boundedCapacity">The maximum number of items the queue may contain.</param>
        public BlockingQueue(int boundedCapacity)
        {
            Contracts.Assert(boundedCapacity > 0);

            _queue = new ConcurrentQueue<T>();
            _itemsAvailable = new CompletableSemaphore(0);
            _spaceAvailable = new CompletableSemaphore(boundedCapacity);
        }

        /// <summary>Cleans up all resources used by the blocking collection.</summary>
        public void Dispose()
        {
            // This method/IDisposable implementation is here for API compat with BlockingCollection<T>,
            // but there's nothing to actually dispose.
        }

        /// <summary>Adds an item to the blocking collection.</summary>
        /// <param name="item">The item to add.</param>
        /// <param name="millisecondsTimeout">The time to wait, in milliseconds, or -1 to wait indefinitely.</param>
        /// <returns>
        /// true if the item was successfully added; false if the timeout expired or if the collection were marked
        /// as complete for adding before the item could be added.
        /// </returns>
        public bool TryAdd(T item, int millisecondsTimeout = 0)
        {
            Contracts.Assert(!_itemsAvailable.Completed);

            // Wait for space to be available, then once it is, enqueue the item,
            // and notify anyone waiting that another item is available.
            if (_spaceAvailable.Wait(millisecondsTimeout))
            {
                _queue.Enqueue(item);
                _itemsAvailable.Release();
                return true;
            }

            return false;
        }

        /// <summary>Tries to take an item from the blocking collection.</summary>
        /// <param name="item">The item removed, or default if none could be taken.</param>
        /// <param name="millisecondsTimeout">The time to wait, in milliseconds, or -1 to wait indefinitely.</param>
        /// <returns>
        /// true if the item was successfully taken; false if the timeout expired or if the collection is empty
        /// and has been marked as complete for adding.
        /// </returns>
        public bool TryTake(out T item, int millisecondsTimeout = 0)
        {
            // Wait for an item to be available, and once one is, dequeue it,
            // and assuming we got one, notify anyone waiting that space is available.
            if (_itemsAvailable.Wait(millisecondsTimeout))
            {
                bool gotItem = _queue.TryDequeue(out item);
                Contracts.Assert(gotItem || _itemsAvailable.Completed);
                if (gotItem)
                {
                    _spaceAvailable.Release();
                    return true;
                }
            }

            item = default;
            return false;
        }

        /// <summary>
        /// Gets an enumerable for taking all items out of the collection until
        /// the collection has been marked as complete for adding and is empty.
        /// </summary>
        public IEnumerable<T> GetConsumingEnumerable()
        {
            // Block waiting for each additional item, yielding each as we take it,
            // and exiting only when the collection is and will forever be empty.
            while (TryTake(out T item, Timeout.Infinite))
            {
                yield return item;
            }
        }

        /// <summary>Mark the collection as complete for adding.</summary>
        /// <remarks>After this is called, no calls made on this queue will block.</remarks>
        public void CompleteAdding()
        {
            _itemsAvailable.Complete();
            _spaceAvailable.Complete();
        }

        /// <summary>
        /// A basic monitor-based semaphore that, in addition to standard Wait/Release semantics,
        /// also supports marking the semaphore as completed, in which case all waiters immediately
        /// fail if there's no count remaining.
        /// </summary>
        private sealed class CompletableSemaphore
        {
            /// <summary>The remaining count in the semaphore.</summary>
            private int _count;
            /// <summary>The number of threads currently waiting in Wait.</summary>
            private int _waiters;

            /// <summary>Initializes the semaphore with the specified initial count.</summary>
            /// <param name="initialCount">The initial count.</param>
            public CompletableSemaphore(int initialCount)
            {
                Contracts.Assert(initialCount >= 0);
                _count = initialCount;
            }

            /// <summary>Gets whether the semaphore has been marked as completed.</summary>
            /// <remarks>
            /// If completed, no calls to Wait will block; if no count remains, regardless of timeout, Waits will
            /// return immediately with a result of false.
            /// </remarks>
            public bool Completed { get; private set; }

            /// <summary>Releases the semaphore once.</summary>
            public void Release()
            {
                lock (this)
                {
                    // Increment the count, and if anyone is waiting, notify one of them.
                    _count++;
                    if (_waiters > 0)
                    {
                        Monitor.Pulse(this);
                    }
                }
            }

            /// <summary>Blocks the current thread until it can enter the semaphore once.</summary>
            /// <param name="millisecondsTimeout">The maximum amount of time to wait to enter the semaphore, or -1 to wait indefinitely.</param>
            /// <returns>true if the semaphore was entered; otherwise, false.</returns>
            public bool Wait(int millisecondsTimeout = Timeout.Infinite)
            {
                lock (this)
                {
                    while (true)
                    {
                        // If the count is greater than 0, take one, and we're done.
                        Contracts.Assert(_count >= 0);
                        if (_count > 0)
                        {
                            _count--;
                            return true;
                        }

                        // If the count is 0 but we've been marked as completed, fail.
                        if (Completed)
                        {
                            return false;
                        }

                        // Wait until either there's a count available or the timeout expires.
                        // In practice we should never have a case where the timeout occurs
                        // and we need to wait again, so we don't bother doing any manual
                        // tracking of the timeout.
                        _waiters++;
                        try
                        {
                            if (!Monitor.Wait(this, millisecondsTimeout))
                            {
                                return false;
                            }
                        }
                        finally
                        {
                            _waiters--;
                            Contracts.Assert(_waiters >= 0);
                        }
                    }
                }
            }

            /// <summary>Marks the semaphore as completed, such that no further operations will block.</summary>
            public void Complete()
            {
                lock (this)
                {
                    // Mark the semaphore as completed and wake up all waiters.
                    Completed = true;
                    if (_waiters > 0)
                    {
                        Monitor.PulseAll(this);
                    }
                }
            }
        }
    }
}
