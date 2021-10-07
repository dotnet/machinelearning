// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;

namespace Microsoft.ML.AutoML
{
    internal class AutoMLLogger
    {
        public const string ChannelName = "AutoML";

        private readonly IChannel _channel;

        public AutoMLLogger(MLContext context)
        {
            _channel = ((IChannelProvider)context).Start(ChannelName);
        }

        public void Trace(string message)
        {
            _channel.Trace(MessageSensitivity.None, message);
        }

        public void Error(string message)
        {
            _channel.Error(MessageSensitivity.None, message);
        }
    }
}
