// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Threading;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    /// <summary>
    /// The primary use case for this structure is to impose ordering among
    /// multiple producer threads, in case one is producing output that one
    /// wishes to have ordered.
    ///
    /// More specifically, the ordered waiter allows a thread to wait on a
    /// particular position. So if three threads wait on 0, 1, then 2 (in
    /// any order), the first thread to clear the wait will be 0, then 1 will
    /// be cleared once incremented, then 2 will be cleared once incremented.
    /// </summary>
    [BestFriend]
    internal sealed class OrderedWaiter
    {
        /// <summary>
        /// This is an event-line pair. The intended usage is, when the line
        /// is hit by the containing ordered waiter, the thread will be hit.
        /// </summary>
        private readonly struct WaitStats
        {
            public readonly long Line;
            public readonly ManualResetEventSlim Event;

            public WaitStats(long line)
            {
                Line = line;
                Event = new ManualResetEventSlim(false);
            }
        }

        // A minheap of waiters.
        private readonly Heap<WaitStats> _waiters;
        // Which level has yet cleared.
        private long _currCleared;
        // What exception has been signaled, or null if no exception has been signaled.
        private Exception _ex;

        /// <summary>
        /// Creates an ordered waiter.
        /// </summary>
        /// <param name="firstCleared">If true, then the first position (that is,
        /// position 0) will be considered already free to proceed. If not something
        /// will need to hit increment.</param>
        public OrderedWaiter(bool firstCleared = true)
        {
            _waiters = new Heap<WaitStats>((s1, s2) => s1.Line > s2.Line);
            _currCleared = firstCleared ? 0 : -1;
        }

        /// <summary>
        /// Creates an ordered waiter.
        /// </summary>
        /// <param name="startPos">If startPos is &gt;= 0 then waiter starts from that position.
        /// If not something will need to hit increment |startPos| times.</param>
        public OrderedWaiter(long startPos)
        {
            _waiters = new Heap<WaitStats>((s1, s2) => s1.Line > s2.Line);
            _currCleared = startPos;
        }

        /// <summary>
        /// Wait on a given position. This will block, until this object has
        /// <see cref="Increment"/> called up to the position indicated. This
        /// accepts cancellation tokens, but the default cancellation token also
        /// works.
        /// </summary>
        public void Wait(long position, CancellationToken token = default(CancellationToken))
        {
            if (_ex != null)
                throw Contracts.Except(_ex, "Event we were waiting on was subject to an exception");
            if (position <= _currCleared)
                return;
            WaitStats ev;
            lock (_waiters)
            {
                // No need to do anything in this strange case.
                if (_ex != null)
                    throw Contracts.Except(_ex, "Event we were waiting on was subject to an exception");
                if (position <= _currCleared)
                    return;
                ev = new WaitStats(position);
                _waiters.Add(ev);
            }
            ev.Event.Wait(token);
            if (_ex != null)
                throw Contracts.Except(_ex, "Event we were waiting on was subject to an exception");
        }

        /// <summary>
        /// Moves the waiter to the next position, and signals any waiters waiting at
        /// or before that position.
        /// </summary>
        public long Increment()
        {
            lock (_waiters)
            {
                // REVIEW: There's no code that will actually hit this condition
                // unless there's a bug somewhere, so should this be an assert?
                if (_currCleared < long.MaxValue)
                    _currCleared++;
                while (_waiters.Count > 0 && _waiters.Top.Line <= _currCleared)
                    _waiters.Pop().Event.Set();
                return _currCleared;
            }
        }

        /// <summary>
        /// Signals all waiters. No more calls to <see cref="Increment"/> should be
        /// attempted.
        /// </summary>
        public long IncrementAll()
        {
            lock (_waiters)
            {
                _currCleared = long.MaxValue;
                while (_waiters.Count > 0)
                    _waiters.Pop().Event.Set();
                return _currCleared;
            }
        }

        /// <summary>
        /// This will signal all the waiters, but cause them to throw an exception.
        /// </summary>
        /// <param name="ex">The exception that will be the inner exception, of an
        /// exception that will throw for all current and subsequent waiters.</param>
        public void SignalException(Exception ex)
        {
            Contracts.CheckValue(ex, nameof(ex));
            lock (_waiters)
            {
                // REVIEW: In the event that an exception was already set,
                // my preference is to continue to throw the original one, since
                // the actual current use cases of the waiter make this more natural.
                if (_ex == null)
                    _ex = ex;
                while (_waiters.Count > 0)
                    _waiters.Pop().Event.Set();
            }
        }
    }
}
