// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.CLI.ShellProgressBar
{
    public class ChildProgressBar : ProgressBarBase, IProgressBar
    {
        private readonly Action _scheduleDraw;
        private readonly Action<ProgressBarHeight> _growth;

        public DateTime StartDate { get; } = DateTime.Now;

        protected override void DisplayProgress() => _scheduleDraw?.Invoke();

        internal ChildProgressBar(int maxTicks, string message, Action scheduleDraw, ProgressBarOptions options = null, Action<ProgressBarHeight> growth = null)
            : base(maxTicks, message, options)
        {
            _scheduleDraw = scheduleDraw;
            _growth = growth;
            _growth?.Invoke(ProgressBarHeight.Increment);
        }

        private bool _calledDone;
        private readonly object _callOnce = new object();

        protected override void OnDone()
        {
            if (_calledDone) return;
            lock (_callOnce)
            {
                if (_calledDone) return;

                if (this.EndTime == null)
                    this.EndTime = DateTime.Now;

                if (this.Collapse)
                    _growth?.Invoke(ProgressBarHeight.Decrement);

                _calledDone = true;
            }
        }

        public void Dispose()
        {
            OnDone();
            foreach (var c in this.Children) c.Dispose();
        }
    }
}
