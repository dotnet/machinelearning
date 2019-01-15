// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace Microsoft.ML.Internal.Utilities
{
    internal static partial class Utils
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
