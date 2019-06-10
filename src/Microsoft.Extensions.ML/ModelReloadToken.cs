// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Threading;
using Microsoft.Extensions.Primitives;

namespace Microsoft.Extensions.ML
{
    /// <summary>
    /// Implements <see cref="IChangeToken"/>
    /// </summary>
    public class ModelReloadToken : IChangeToken
    {
        private readonly CancellationTokenSource _cts;

        public ModelReloadToken()
        {
            _cts = new CancellationTokenSource();
        }

        /// <summary>
        /// Indicates if this token will proactively raise callbacks.
        /// </summary>
        public bool ActiveChangeCallbacks => true;

        /// <summary>
        /// Gets a value that indicates if a change has occurred.
        /// </summary>
        public bool HasChanged => _cts.IsCancellationRequested;

        /// <summary>
        /// Registers for a callback that will be invoked when the entry has changed. <see cref="Microsoft.Extensions.Primitives.IChangeToken.HasChanged"/>
        /// MUST be set before the callback is invoked.
        /// </summary>
        /// <param name="callback">The callback to invoke.</param>
        /// <param name="state">State to be passed into the callback.</param>
        /// <returns>
        /// An System.IDisposable that is used to unregister the callback.
        /// </returns>
        public IDisposable RegisterChangeCallback(Action<object> callback, object state) => _cts.Token.Register(callback, state);

        /// <summary>
        /// Used to trigger the change token when a reload occurs.
        /// </summary>
        public void OnReload() => _cts.Cancel();
    }
}
