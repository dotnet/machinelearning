// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML
{
    /// <summary>
    /// Provides data for the <see cref="MLContext.Log"/> event.
    /// </summary>
    public class LoggingEventArgs : EventArgs
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="LoggingEventArgs"/> class.
        /// </summary>
        /// <param name="message">The message being logged.</param>
        public LoggingEventArgs(string message)
        {
            Message = message;
        }

        /// <summary>
        /// Gets the message being logged.
        /// </summary>
        public string Message { get; }
    }
}