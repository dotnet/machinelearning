using System;
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.CLI.ShellProgressBar
{
    public interface IProgressBar : IDisposable
    {
        ChildProgressBar Spawn(int maxTicks, string message, ProgressBarOptions options = null);

        void Tick(string message = null);

        int MaxTicks { get; set; }
        string Message { get; set; }

        double Percentage { get; }
        int CurrentTick { get; }

        ConsoleColor ForeGroundColor { get; }
    }
}
