// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;

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
        /// Initializes a new instane of <see cref="LoggingEventArgs"/> class that includes the kind and source of the message
        /// </summary>
        /// <param name="message"> The message being logged </param>
        /// <param name="kind"> The type of message <see cref="ChannelMessageKind"/> </param>
        /// <param name="source"> The source of the message </param>
        public LoggingEventArgs(string message, ChannelMessageKind kind, string source)
        {
            RawMessage = message;
            Kind = kind;
            Source = source;
            Message = $"[Source={Source}, Kind={Kind}] {RawMessage}";
        }

        /// <summary>
        /// Gets the source component of the event
        /// </summary>
        public string Source { get; }

        /// <summary>
        /// Gets the type of message
        /// </summary>
        public ChannelMessageKind Kind { get; }

        /// <summary>
        /// Gets the message being logged.
        /// </summary>
        public string Message { get; }

        /// <summary>
        /// Gets the original message that doesn't include the source and kind
        /// </summary>
        public string RawMessage { get; }
    }
}
