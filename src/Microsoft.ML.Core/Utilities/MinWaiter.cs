// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Threading;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    /// <summary>
    /// A synchronization primitive meant to address situations where you have a set of
    /// entities of known count, where you want to iteratively provide critical sections
    /// for each depending on which comes first, but you do not necessarily know what
    /// constitutes "first" until all such entities tell you where they stand in line.
    ///
    /// The anticipated usage is that whatever entity is using the <see cref="MinWaiter"/>
    /// to synchronize itself, will register itself using <see cref="Register"/>
    /// so as to unblock any "lower" waiters as soon as it knows what value it needs to
    /// wait on, perform whatever local work it can, and when it needs to, wait on the
    /// event it got when it registered itself. It may then repeat the cycle by
    /// registering itself for a new event (or, finally, retiring itself through
    /// <see cref="Retire"/>).
    /// </summary>
    public sealed class MinWaiter
    {
        /// <summary>
        /// This is an event-line pair. The intended usage is, when the line
        /// is the minimum at a point when all waiters have registered, the event
        /// will be signaled.
        /// </summary>
        private readonly struct WaitStats
        {
            public readonly long Line;
            public readonly ManualResetEventSlim Event;

            public WaitStats(long line)
            {
                Line = line;
                // REVIEW: Since there are a maximum number of waiters at any one
                // scheme, we should investigate re-using these events rather than composing
                // new ones on each wait.
                Event = new ManualResetEventSlim(false);
            }
        }

        // A minheap of waiters.
        private readonly Heap<WaitStats> _waiters;
        // The maximum number of weighters.
        private int _maxWaiters;

        /// <summary>
        /// Creates a minimum waiter.
        /// </summary>
        /// <param name="waiters">The initial number of waiters</param>
        public MinWaiter(int waiters)
        {
            Contracts.CheckParam(waiters > 0, nameof(waiters), "Must have at least one waiter");
            _maxWaiters = waiters;
            _waiters = new Heap<WaitStats>((s1, s2) => s1.Line > s2.Line);
        }

        /// <summary>
        /// Indicates to the waiter that we want to, at some future point, wait at a given
        /// position. This object will return a reset event that can be waited on, at the
        /// point when we actually want to wait. This method itself has the potential to
        /// signal other events, if by registering ourselves the waiter becomes aware of
        /// the maximum number of waiters, allowing that waiter to enter its critical state.
        ///
        /// If multiple events are associated with the minimum value, then only one will
        /// be signaled, and the rest will remain unsignaled. Which is chosen is undefined.
        /// </summary>
        public ManualResetEventSlim Register(long position)
        {
            WaitStats ev;
            lock (_waiters)
            {
                Contracts.Check(_maxWaiters > 0, "All waiters have been retired, Wait should not be called at this point");
                // We should never reach the state
                Contracts.Assert(_waiters.Count < _maxWaiters);
                ev = new WaitStats(position);
                // REVIEW: Optimize the case where this is the minimum?
                _waiters.Add(ev);
                SignalIfNeeded();
                Contracts.Assert(_waiters.Count < _maxWaiters);
            }
            // REVIEW: At first I instead returned an action, ev.Event.Wait.
            // It may be less efficient, but I don't know if returning an action here
            // is really that bad?
            return ev.Event;
        }

        /// <summary>
        /// Retires one of the waiters, and return the current maximum number of waiters.
        /// If it so happens that by retiring this waiter the number of waiters reaches the
        /// maximum, the appropriate waiter will be signaled as described in <see cref="Register"/>.
        /// </summary>
        public int Retire()
        {
            lock (_waiters)
            {
                Contracts.Check(_maxWaiters > 0, "Attempt to retire more waiters than were initially declared");
                Contracts.Assert(_waiters.Count < _maxWaiters);
                if (--_maxWaiters > 0)
                    SignalIfNeeded();
                Contracts.Assert(_maxWaiters == 0 || _waiters.Count < _maxWaiters);
                return _maxWaiters;
            }
        }

        private void SignalIfNeeded()
        {
            if (_waiters.Count == _maxWaiters)
            {
                var ev = _waiters.Pop();
                ev.Event.Set();
            }
        }
    }
}
