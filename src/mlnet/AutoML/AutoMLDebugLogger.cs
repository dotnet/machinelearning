// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Auto;
using NLog;

namespace Microsoft.ML.CLI.AutoML
{
    internal class AutoMLDebugLogger: IDebugLogger
    {
        public static AutoMLDebugLogger Instance = new AutoMLDebugLogger();

        private static Logger logger = LogManager.GetCurrentClassLogger();

        public void Log(LogSeverity severity, string message)
        {
            logger.Log(LogLevel.Trace, message);
        }
    }
}
