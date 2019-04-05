// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Auto
{
    internal interface IDebugLogger
    {
        void Log(LogSeverity logLevel, string message);
    }

    internal enum LogSeverity
    {
        Error,
        Debug
    }
}
