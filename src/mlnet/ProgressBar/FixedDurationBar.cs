// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Threading;

namespace Microsoft.ML.CLI.ShellProgressBar
{
    public class FixedDurationBar : ProgressBar
    {
        public bool IsCompleted { get; private set; }

        private readonly ManualResetEvent _completedHandle;
        public WaitHandle CompletedHandle => _completedHandle;

        public FixedDurationBar(TimeSpan duration, string message, ProgressBarOptions options = null) : base((int)Math.Ceiling(duration.TotalSeconds), message, options)
        {
            _completedHandle = new ManualResetEvent(false);
            if (!Options.DisplayTimeInRealTime)
                throw new ArgumentException(
                    $"{nameof(ProgressBarOptions)}.{nameof(ProgressBarOptions.DisplayTimeInRealTime)} has to be true for {nameof(FixedDurationBar)}", nameof(options)
                );
        }

        private long _seenTicks;
        protected override void OnTimerTick()
        {
            Interlocked.Increment(ref _seenTicks);
            if (_seenTicks % 2 == 0) Tick();
            base.OnTimerTick();
        }

        protected override void OnDone()
        {
            IsCompleted = true;
            _completedHandle.Set();
        }
    }
}
