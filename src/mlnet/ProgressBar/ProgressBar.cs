// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

namespace Microsoft.ML.CLI.ShellProgressBar
{
    public class ProgressBar : ProgressBarBase, IProgressBar
    {
        private static readonly bool IsWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);

        private readonly ConsoleColor _originalColor;
        private readonly int _originalCursorTop;
        private readonly int _originalWindowTop;
        private int _isDisposed;

        private Timer _timer;
        private int _visibleDescendants = 0;
        private readonly AutoResetEvent _displayProgressEvent;
        private readonly Task _displayProgress;

        public ProgressBar(int maxTicks, string message, ConsoleColor color)
            : this(maxTicks, message, new ProgressBarOptions { ForegroundColor = color })
        {
        }

        public ProgressBar(int maxTicks, string message, ProgressBarOptions options = null)
            : base(maxTicks, message, options)
        {
            Console.WriteLine();
            Console.SetCursorPosition(Console.CursorLeft, Math.Max(0, Console.CursorTop - 1));
            _originalCursorTop = Console.CursorTop;
            _originalWindowTop = Console.WindowTop;
            _originalColor = Console.ForegroundColor;

            Console.CursorVisible = false;

            if (this.Options.EnableTaskBarProgress)
                TaskbarProgress.SetState(TaskbarProgress.TaskbarStates.Normal);

            if (this.Options.DisplayTimeInRealTime)
                _timer = new Timer((s) => OnTimerTick(), null, 500, 500);
            else //draw once
                _timer = new Timer((s) =>
                {
                    _timer.Dispose();
                    DisplayProgress();
                }, null, 0, 1000);

            _displayProgressEvent = new AutoResetEvent(false);
            _displayProgress = Task.Run(() =>
            {
                while (_isDisposed == 0)
                {
                    if (!_displayProgressEvent.WaitOne(TimeSpan.FromSeconds(10)))
                        continue;
                    try
                    {
                        UpdateProgress();
                    }
                    catch
                    {
                        // don't want to crash background thread
                    }
                }
            });
        }

        protected virtual void OnTimerTick()
        {
            DisplayProgress();
        }

        protected override void Grow(ProgressBarHeight direction)
        {
            switch (direction)
            {
                case ProgressBarHeight.Increment:
                    Interlocked.Increment(ref _visibleDescendants);
                    break;
                case ProgressBarHeight.Decrement:
                    Interlocked.Decrement(ref _visibleDescendants);
                    break;
            }
        }

        private struct Indentation
        {
            public Indentation(ConsoleColor color, bool lastChild)
            {
                this.ConsoleColor = color;
                this.LastChild = lastChild;
            }

            public string Glyph => !LastChild ? "├─" : "└─";

            public readonly ConsoleColor ConsoleColor;
            public readonly bool LastChild;
        }

        private static void ProgressBarBottomHalf(double percentage, DateTime startDate, DateTime? endDate, string message,
            Indentation[] indentation, bool progressBarOnBottom)
        {
            var depth = indentation.Length;
            var maxCharacterWidth = Console.WindowWidth - (depth * 2) + 2;
            var duration = ((endDate ?? DateTime.Now) - startDate);
            string durationString = null;
            if (duration.Days > 0)
                durationString = $"{duration.Days:00}:{duration.Hours:00}:{duration.Minutes:00}:{duration.Seconds:00}";
            else
                durationString = $"{duration.Hours:00}:{duration.Minutes:00}:{duration.Seconds:00}";

            var column1Width = Console.WindowWidth - durationString.Length - (depth * 2) + 2;
            var column2Width = durationString.Length;

            if (progressBarOnBottom)
                DrawTopHalfPrefix(indentation, depth);
            else
                DrawBottomHalfPrefix(indentation, depth);

            var format = $"{{0, -{column1Width}}}{{1,{column2Width}}}";

            var truncatedMessage = StringExtensions.Excerpt(message, column1Width);
            var formatted = string.Format(format, truncatedMessage, durationString);
            var m = formatted + new string(' ', Math.Max(0, maxCharacterWidth - formatted.Length));
            Console.Write(m);
        }

        private static void DrawBottomHalfPrefix(Indentation[] indentation, int depth)
        {
            for (var i = 1; i < depth; i++)
            {
                var ind = indentation[i];
                Console.ForegroundColor = indentation[i - 1].ConsoleColor;
                if (!ind.LastChild)
                    Console.Write(i == (depth - 1) ? ind.Glyph : "│ ");
                else
                    Console.Write(i == (depth - 1) ? ind.Glyph : "  ");
            }

            Console.ForegroundColor = indentation[depth - 1].ConsoleColor;
        }

        private static void ProgressBarTopHalf(
            double percentage,
            char progressCharacter,
            char? progressBackgroundCharacter,
            ConsoleColor? backgroundColor,
            Indentation[] indentation, bool progressBarOnTop)
        {
            var depth = indentation.Length;
            var width = Console.WindowWidth - (depth * 2) + 2;

            if (progressBarOnTop)
                DrawBottomHalfPrefix(indentation, depth);
            else
                DrawTopHalfPrefix(indentation, depth);

            var newWidth = (int)((width * percentage) / 100d);
            var progBar = new string(progressCharacter, newWidth);
            Console.Write(progBar);
            if (backgroundColor.HasValue)
            {
                Console.ForegroundColor = backgroundColor.Value;
                Console.Write(new string(progressBackgroundCharacter ?? progressCharacter, width - newWidth));
            }
            else Console.Write(new string(' ', width - newWidth));

            Console.ForegroundColor = indentation[depth - 1].ConsoleColor;
        }

        private static void DrawTopHalfPrefix(Indentation[] indentation, int depth)
        {
            for (var i = 1; i < depth; i++)
            {
                var ind = indentation[i];
                Console.ForegroundColor = indentation[i - 1].ConsoleColor;
                if (ind.LastChild && i != (depth - 1))
                    Console.Write("  ");
                else
                    Console.Write("│ ");
            }

            Console.ForegroundColor = indentation[depth - 1].ConsoleColor;
        }

        protected override void DisplayProgress()
        {
            _displayProgressEvent.Set();
        }

        private void UpdateProgress()
        {
            Console.CursorVisible = false;
            var indentation = new[] { new Indentation(this.ForeGroundColor, true) };
            var mainPercentage = this.Percentage;
            var cursorTop = _originalCursorTop;

            Console.ForegroundColor = this.ForeGroundColor;

            void TopHalf()
            {
                ProgressBarTopHalf(mainPercentage,
                    this.Options.ProgressCharacter,
                    this.Options.BackgroundCharacter,
                    this.Options.BackgroundColor,
                    indentation,
                    this.Options.ProgressBarOnBottom
                );
            }

            if (this.Options.ProgressBarOnBottom)
            {
                ProgressBarBottomHalf(mainPercentage, this._startDate, null, this.Message, indentation, this.Options.ProgressBarOnBottom);
                Console.SetCursorPosition(0, ++cursorTop);
                TopHalf();
            }
            else
            {
                TopHalf();
                Console.SetCursorPosition(0, ++cursorTop);
                ProgressBarBottomHalf(mainPercentage, this._startDate, null, this.Message, indentation, this.Options.ProgressBarOnBottom);
            }

            if (this.Options.EnableTaskBarProgress)
                TaskbarProgress.SetValue(mainPercentage, 100);

            DrawChildren(this.Children, indentation, ref cursorTop);

            ResetToBottom(ref cursorTop);

            Console.SetCursorPosition(0, _originalCursorTop);
            Console.ForegroundColor = _originalColor;

            if (!(mainPercentage >= 100)) return;
            _timer?.Dispose();
            _timer = null;
        }

        private static void ResetToBottom(ref int cursorTop)
        {
            var resetString = new string(' ', Console.WindowWidth);
            var windowHeight = Console.WindowHeight;
            if (cursorTop >= (windowHeight - 1)) return;
            do
            {
                Console.Write(resetString);
            } while (++cursorTop < (windowHeight - 1));
        }

        private static void DrawChildren(IEnumerable<ChildProgressBar> children, Indentation[] indentation, ref int cursorTop)
        {
            var view = children.Where(c => !c.Collapse).Select((c, i) => new { c, i }).ToList();
            if (!view.Any()) return;

            var windowHeight = Console.WindowHeight;
            var lastChild = view.Max(t => t.i);
            foreach (var tuple in view)
            {
                //Dont bother drawing children that would fall off the screen
                if (cursorTop >= (windowHeight - 2))
                    return;

                var child = tuple.c;
                var currentIndentation = new Indentation(child.ForeGroundColor, tuple.i == lastChild);
                var childIndentation = NewIndentation(indentation, currentIndentation);

                var percentage = child.Percentage;
                Console.ForegroundColor = child.ForeGroundColor;

                void TopHalf()
                {
                    ProgressBarTopHalf(percentage,
                        child.Options.ProgressCharacter,
                        child.Options.BackgroundCharacter,
                        child.Options.BackgroundColor,
                        childIndentation,
                        child.Options.ProgressBarOnBottom
                    );
                }

                Console.SetCursorPosition(0, ++cursorTop);

                if (child.Options.ProgressBarOnBottom)
                {
                    ProgressBarBottomHalf(percentage, child.StartDate, child.EndTime, child.Message, childIndentation, child.Options.ProgressBarOnBottom);
                    Console.SetCursorPosition(0, ++cursorTop);
                    TopHalf();
                }
                else
                {
                    TopHalf();
                    Console.SetCursorPosition(0, ++cursorTop);
                    ProgressBarBottomHalf(percentage, child.StartDate, child.EndTime, child.Message, childIndentation, child.Options.ProgressBarOnBottom);
                }

                DrawChildren(child.Children, childIndentation, ref cursorTop);
            }
        }

        private static Indentation[] NewIndentation(Indentation[] array, Indentation append)
        {
            var result = new Indentation[array.Length + 1];
            Array.Copy(array, result, array.Length);
            result[array.Length] = append;
            return result;
        }

        public void Dispose()
        {
            if (Interlocked.CompareExchange(ref _isDisposed, 1, 0) != 0)
                return;

            // make sure background task is stopped before we clean up
            _displayProgressEvent.Set();
            _displayProgress.Wait();

            // update one last time - needed because background task might have
            // been already in progress before Dispose was called and it might
            // have been running for a very long time due to poor performance
            // of System.Console
            UpdateProgress();

            if (this.EndTime == null) this.EndTime = DateTime.Now;
            var openDescendantsPadding = (_visibleDescendants * 2);

            if (this.Options.EnableTaskBarProgress)
                TaskbarProgress.SetState(TaskbarProgress.TaskbarStates.NoProgress);

            try
            {
                var moveDown = 0;
                var currentWindowTop = Console.WindowTop;
                if (currentWindowTop != _originalWindowTop)
                {
                    var x = Math.Max(0, Math.Min(2, currentWindowTop - _originalWindowTop));
                    moveDown = _originalCursorTop + x;
                }
                else moveDown = _originalCursorTop + 2;

                Console.CursorVisible = true;
                Console.SetCursorPosition(0, openDescendantsPadding + moveDown);
            }
            // This is bad and I should feel bad, but i rather eat pbar exceptions in productions then causing false negatives
            catch
            {
            }

            Console.WriteLine();
            _timer?.Dispose();
            _timer = null;
            foreach (var c in this.Children) c.Dispose();
            OnDone();
        }
    }
}
