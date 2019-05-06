// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Internal.Utilities
{
    internal static partial class Utils
    {
        public static Task RunOnBackgroundThread(Action start) =>
            ImmediateBackgroundThreadPool.Queue(start);

        public static Task RunOnBackgroundThread(Action<object> start, object obj) =>
            ImmediateBackgroundThreadPool.Queue(start, obj);

        public static Thread RunOnForegroundThread(ParameterizedThreadStart start) =>
            new Thread(start) { IsBackground = false };

        /// <summary>
        /// Naive thread pool focused on reducing the latency to execution of chunky work items as much as possible.
        /// If a thread is ready to process a work item the moment a work item is queued, it's used, otherwise
        /// a new thread is created. This is meant as a stop-gap measure for workloads that would otherwise be
        /// creating a new thread for every work item.
        /// </summary>
        private static class ImmediateBackgroundThreadPool
        {
            /// <summary>How long should threads wait around for additional work items before retiring themselves.</summary>
            private const int IdleMilliseconds = 1_000;
            /// <summary>The queue of work items. Also used as a lock to protect all relevant state.</summary>
            private static readonly Queue<(Delegate, object, TaskCompletionSource<bool>)> _queue = new Queue<(Delegate, object, TaskCompletionSource<bool>)>();
            /// <summary>The number of threads currently waiting for work to arrive.</summary>
            private static int _availableThreads = 0;

            /// <summary>
            /// Queues an <see cref="Action"/> delegate to be executed immediately on another thread,
            /// and returns a <see cref="Task"/> that represents its eventual completion. The task will
            /// always end in the <see cref="TaskStatus.RanToCompletion"/> state; if the delegate throws
            /// an exception, it'll be allowed to propagate on the thread, crashing the process.
            /// </summary>
            public static Task Queue(Action threadStart) => Queue((Delegate)threadStart, null);

            /// <summary>
            /// Queues an <see cref="Action{Object}"/> delegate and associated state to be executed immediately on another thread,
            /// and returns a <see cref="Task"/> that represents its eventual completion. The task will
            /// always end in the <see cref="TaskStatus.RanToCompletion"/> state; if the delegate throws
            /// an exception, it'll be allowed to propagate on the thread, crashing the process.
            /// </summary>
            public static Task Queue(Action<object> threadStart, object state) => Queue((Delegate)threadStart, state);

            private static Task Queue(Delegate threadStart, object state)
            {
                // Create the TaskCompletionSource used to represent this work.
                // Call sites only care about completion, not about the distinction between
                // success and failure and do not expect exceptions to be propagated in this manner,
                // so only SetResult is used.
                var tcs = new TaskCompletionSource<bool>(TaskContinuationOptions.RunContinuationsAsynchronously);

                // Queue the work for a thread to pick up. If no thread is immediately available, it will create one.
                Enqueue((threadStart, state, tcs));

                // Return the task.
                return tcs.Task;

                void CreateThread()
                {
                    // Create a new background thread to run the work.
                    var t = new Thread(() =>
                    {
                        // Repeatedly get the next item and invoke it, setting its TCS when we're done.
                        // This will wait for up to the idle time before giving up and exiting.
                        while (TryDequeue(out (Delegate action, object state, TaskCompletionSource<bool> tcs) item))
                        {
                            try
                            {
                                if (item.action is Action<object> pts)
                                {
                                    pts(item.state);
                                }
                                else
                                {
                                    ((Action)item.action)();
                                }
                            }
                            finally
                            {
                                item.tcs.SetResult(true);
                            }
                        }
                    });
                    t.IsBackground = true;
                    t.Start();
                }

                void Enqueue((Delegate, object, TaskCompletionSource<bool>) item)
                {
                    // Enqueue the work. If there are currently fewer threads waiting
                    // for work than there are work items in the queue, create another
                    // thread. This is a heuristic, in that we might end up creating
                    // more threads than are truly needed, but this whole type is being
                    // used to replace a previous solution where every work item created
                    // its own thread, so this is an improvement regardless of any
                    // such inefficiencies.
                    lock (_queue)
                    {
                        _queue.Enqueue(item);

                        if (_queue.Count <= _availableThreads)
                        {
                            Monitor.Pulse(_queue);
                            return;
                        }
                    }

                    // No thread was currently available.  Create one.
                    CreateThread();
                }

                bool TryDequeue(out (Delegate action, object state, TaskCompletionSource<bool> tcs) item)
                {
                    // Dequeues the next item if one is available. Before checking,
                    // the available thread count is increased, so that enqueuers can
                    // see how many threads are currently waiting, with the count
                    // decreased after. Each time it waits, it'll wait for at most
                    // the idle timeout before giving up.
                    lock (_queue)
                    {
                        _availableThreads++;
                        try
                        {
                            while (_queue.Count == 0)
                            {
                                if (!Monitor.Wait(_queue, IdleMilliseconds))
                                {
                                    if (_queue.Count > 0)
                                    {
                                        break;
                                    }

                                    item = default;
                                    return false;
                                }
                            }
                        }
                        finally
                        {
                            _availableThreads--;
                        }

                        item = _queue.Dequeue();
                        return true;
                    }
                }
            }
        }
    }

    /// <summary>
    /// An object that serves as a source of a cancellation token, as well as having the ability
    /// for anything to push an exception into the message, to trigger the cancellation. The general
    /// intended usage is that, prior to creating a bunch of collaborating thread workers, this
    /// object is created and made accessible to them, somehow. Each thread worker will work as they
    /// would be engineered without this class, except they will wrap their contents in a try-catch
    /// block to push any exceptions (hopefully none) into this marshaller, using <see cref="Set"/>.
    /// Further, any potentially blocking operation of the thread workers must be changed to use
    /// <see cref="Token"/> as the cancellation token (this token is cancelled iff <see cref="Set"/>
    /// is ever called). The controlling thread, whatever that may be, once it is either sure
    /// <see cref="Set"/> has been called (possibly by receiving the cancellation) or is sure somehow
    /// that the workers have finished by its own means, will call <see cref="ThrowIfSet"/> to throw
    /// the set exception as an inner exception, in the wrapping thread.
    /// </summary>
    [BestFriend]
    internal sealed class ExceptionMarshaller : IDisposable
    {
        private readonly CancellationTokenSource _ctSource;
        private readonly object _lock;

        // The stored exception
        private string _component;
        private Exception _ex;

        /// <summary>
        /// A cancellation token, whose source will be cancelled if <see cref="Set"/> is ever called.
        /// Any thread blocking operation of a family of thread workers using this structure
        /// must use this cancellation token, or else there is a strong possibility for threads
        /// to stop responding if an exception is thrown at any point.
        /// </summary>
        public CancellationToken Token => _ctSource.Token;

        public ExceptionMarshaller()
        {
            _ctSource = new CancellationTokenSource();
            _lock = new object();
        }

        public void Dispose()
        {
            // We don't just put the ThrowIfSet here since we shouldn't throw in dispose.
            _ctSource.Dispose();
        }

        /// <summary>
        /// Store an exception and set the cancellation token. If this was already
        /// called, this newly passed in exception is ignored. (Important, since a
        /// common source of exceptions would be the exceptions relating to the cancellation.)
        /// </summary>
        /// <param name="component">The type of worker that threw the exception, used
        /// in the description of the wrapping exception</param>
        /// <param name="ex">The exception that will become the inner exception</param>
        public void Set(string component, Exception ex)
        {
            Contracts.AssertNonEmpty(component);
            Contracts.AssertValue(ex);

            lock (_lock)
            {
                if (_ex == null)
                {
                    _component = component;
                    _ex = ex;
                }
                _ctSource.Cancel();
            }
        }

        /// <summary>
        /// If an exception was ever set through <see cref="Set"/>, raise it as an appropriate
        /// inner exception. This should only be called just prior to dispose, when the workers
        /// have already finished. If there is no stored exception, this will do nothing. Note
        /// that this does not "expire" the exception, that is, if you were to call this again,
        /// it would throw the same exception.
        /// </summary>
        public void ThrowIfSet(IExceptionContext ectx)
        {
            if (_ex != null)
                throw ectx.Except(_ex, "Exception thrown in {0}", _component);
        }
    }
}
