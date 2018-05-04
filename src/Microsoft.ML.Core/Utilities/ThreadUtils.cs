// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    public static partial class Utils
    {
        public static Thread CreateBackgroundThread(ParameterizedThreadStart start)
        {
            return new Thread(start)
            {
                IsBackground = true
            };
        }

        public static Thread CreateBackgroundThread(ThreadStart start)
        {
            return new Thread(start)
            {
                IsBackground = true
            };
        }

        public static Thread CreateForegroundThread(ParameterizedThreadStart start)
        {
            return new Thread(start)
            {
                IsBackground = false
            };
        }

        public static Thread CreateForegroundThread(ThreadStart start)
        {
            return new Thread(start)
            {
                IsBackground = false
            };
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
    public sealed class ExceptionMarshaller : IDisposable
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

    /// <summary>
    /// Provides a task scheduler that ensures a maximum concurrency level while
    /// running on top of the ThreadPool.
    /// </summary>
    public sealed class LimitedConcurrencyLevelTaskScheduler : TaskScheduler
    {
        // Whether the current thread is processing work items.
        [ThreadStatic]
        private static bool _currentThreadIsProcessingItems;

        //The list of tasks to be executed.
        // protected by lock(_tasks).
        private readonly LinkedList<Task> _tasks;

        //The maximum concurrency level allowed by this scheduler.
        private readonly int _concurrencyLevel;

        // Currently queued or running delegates.
        // protected by lock(_tasks).
        private int _queuedOrRunningDelegatesCount;

        // Gets the maximum concurrency level supported by this scheduler.
        public override int MaximumConcurrencyLevel => _concurrencyLevel;

        /// <summary>
        /// Initializes an instance of the LimitedConcurrencyLevelTaskScheduler class with the
        /// specified concurrency level.
        /// </summary>
        public LimitedConcurrencyLevelTaskScheduler(int concurrencyLevel)
        {
            Contracts.Assert(concurrencyLevel >= 1);
            _tasks = new LinkedList<Task>();
            _concurrencyLevel = concurrencyLevel;
            _queuedOrRunningDelegatesCount = 0;
        }

        // Queues a task to the scheduler.
        protected override void QueueTask(Task task)
        {
            // Add the task to the list of tasks to be processed.  If there aren't enough
            // delegates currently queued or running to process tasks, schedule another.
            lock (_tasks)
            {
                _tasks.AddLast(task);
                if (_queuedOrRunningDelegatesCount < _concurrencyLevel)
                {
                    ++_queuedOrRunningDelegatesCount;
                    NotifyThreadPoolOfPendingWork();
                }
            }
        }

        // Attempts to execute the specified task on the current thread.
        // Returns whether the task could be executed on the current thread.
        protected override bool TryExecuteTaskInline(Task task, bool taskWasPreviouslyQueued)
        {
            // If this thread isn't already processing a task, we don't support inlining.
            if (!_currentThreadIsProcessingItems)
                return false;

            // If the task was previously queued, remove it from the queue.
            if (taskWasPreviouslyQueued)
                TryDequeue(task);

            // Try to run the task.
            return base.TryExecuteTask(task);
        }

        // Attempts to remove a previously scheduled task from the scheduler.
        // Returns whether the task could be found and removed.
        protected override bool TryDequeue(Task task)
        {
            lock (_tasks)
                return _tasks.Remove(task);
        }

        // Gets an enumerable of the tasks currently scheduled on this scheduler.
        // Returns an enumerable of the tasks currently scheduled.
        protected override IEnumerable<Task> GetScheduledTasks()
        {
            bool lockTaken = false;
            try
            {
                Monitor.TryEnter(_tasks, ref lockTaken);
                if (lockTaken)
                    return _tasks.ToArray();
                else
                    throw Contracts.ExceptNotSupp();
            }
            finally
            {
                if (lockTaken)
                    Monitor.Exit(_tasks);
            }
        }

        // Informs the ThreadPool that there's work to be executed for this scheduler.
        private void NotifyThreadPoolOfPendingWork()
        {
            WaitCallback action = (state =>
                  {// Note that the current thread is now processing work items.
                   // This is necessary to enable inlining of tasks into this thread.
                      _currentThreadIsProcessingItems = true;
                      try
                      {
                          // Process all available items in the queue.
                          while (true)
                          {
                              Task item;
                              lock (_tasks)
                              {
                                  // When there are no more items to be processed,
                                  // note that we're done processing, and get out.
                                  if (_tasks.Count == 0)
                                  {
                                      --_queuedOrRunningDelegatesCount;
                                      break;
                                  }

                                  // Get the next item from the queue.
                                  item = _tasks.First.Value;
                                  _tasks.RemoveFirst();
                              }

                              // Execute the task we pulled out of the queue.
                              base.TryExecuteTask(item);
                          }
                      }
                      // We're done processing items on the current thread.
                      finally
                      {
                          _currentThreadIsProcessingItems = false;
                      }
                  });
            // Core CLR doesn't have UnsafeQueueUserWorkItem .
            // In CLR world unsafe version is faster, but this is not the case for Core CLR.
            // more context can be found here: https://github.com/dotnet/coreclr/issues/1607
            ThreadPool.UnsafeQueueUserWorkItem(action, null);
        }
    }
}
