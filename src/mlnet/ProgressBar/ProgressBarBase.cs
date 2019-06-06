// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Concurrent;
using System.Text;
using System.Threading;

namespace Microsoft.ML.CLI.ShellProgressBar
{
    public abstract class ProgressBarBase
    {
        static ProgressBarBase()
        {
            Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
        }

        public readonly DateTime StartDate;
        private int _maxTicks;
        private int _currentTick;
        private string _message;

        protected ProgressBarBase(int maxTicks, string message, ProgressBarOptions options)
        {
            _maxTicks = Math.Max(0, maxTicks);
            _message = message;
            Options = options ?? ProgressBarOptions.Default;
            StartDate = DateTime.Now;
            Children = new ConcurrentBag<ChildProgressBar>();
        }

        internal ProgressBarOptions Options { get; }
        internal ConcurrentBag<ChildProgressBar> Children { get; }

        protected abstract void DisplayProgress();

        protected virtual void Grow(ProgressBarHeight direction)
        {
        }

        protected virtual void OnDone()
        {
        }

        public DateTime? EndTime { get; protected set; }

        public ConsoleColor ForeGroundColor =>
            EndTime.HasValue ? Options.ForegroundColorDone ?? Options.ForegroundColor : Options.ForegroundColor;

        public int CurrentTick => _currentTick;

        public int MaxTicks
        {
            get => _maxTicks;
            set
            {
                Interlocked.Exchange(ref _maxTicks, value);
                DisplayProgress();
            }
        }

        public string Message
        {
            get => _message;
            set
            {
                Interlocked.Exchange(ref _message, value);
                DisplayProgress();
            }
        }

        public double Percentage
        {
            get
            {
                var percentage = Math.Max(0, Math.Min(100, (100.0 / _maxTicks) * _currentTick));
                // Gracefully handle if the percentage is NaN due to division by 0
                if (double.IsNaN(percentage) || percentage < 0) percentage = 100;
                return percentage;
            }
        }

        public bool Collapse => EndTime.HasValue && Options.CollapseWhenFinished;

        public ChildProgressBar Spawn(int maxTicks, string message, ProgressBarOptions options = null)
        {
            var pbar = new ChildProgressBar(maxTicks, message, DisplayProgress, options, Grow);
            Children.Add(pbar);
            DisplayProgress();
            return pbar;
        }

        public void Tick(string message = null)
        {
            Interlocked.Increment(ref _currentTick);

            FinishTick(message);
        }

        public void Tick(int newTickCount, string message = null)
        {
            Interlocked.Exchange(ref _currentTick, newTickCount);

            FinishTick(message);
        }

        private void FinishTick(string message)
        {
            if (message != null)
                Interlocked.Exchange(ref _message, message);

            if (_currentTick >= _maxTicks)
            {
                EndTime = DateTime.Now;
                OnDone();
            }
            DisplayProgress();
        }
    }
}
