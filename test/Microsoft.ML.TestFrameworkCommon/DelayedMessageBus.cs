// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Reflection;
using Xunit.Abstractions;
using Xunit.Sdk;

namespace Microsoft.ML.TestFrameworkCommon
{
    /// <summary>
    /// Used to capture messages to potentially be forwarded later. Messages are forwarded by
    /// disposing of the message bus.
    /// </summary>
    public class DelayedMessageBus : IMessageBus
    {
        private readonly IMessageBus innerBus;
        public readonly List<IMessageSinkMessage> messages = new List<IMessageSinkMessage>();

        public DelayedMessageBus(IMessageBus innerBus)
        {
            this.innerBus = innerBus;
        }

        public bool QueueMessage(IMessageSinkMessage message)
        {
            lock (messages)
                messages.Add(message);

            // No way to ask the inner bus if they want to cancel without sending them the message, so
            // we just go ahead and continue always.
            return true;
        }

        public void Dispose()
        {
            foreach (var message in messages)
                innerBus.QueueMessage(message);
        }
    }
}
