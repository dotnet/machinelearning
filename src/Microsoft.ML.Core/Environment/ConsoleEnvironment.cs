// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma warning disable 420 // volatile with Interlocked.CompareExchange

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;

namespace Microsoft.ML.Runtime.Data
{
    using Stopwatch = System.Diagnostics.Stopwatch;

    public sealed class ConsoleEnvironment : HostEnvironmentBase<ConsoleEnvironment>
    {
        public const string ComponentHistoryKey = "ComponentHistory";

        private sealed class ConsoleWriter
        {
            private readonly object _lock;
            private readonly ConsoleEnvironment _parent;
            private readonly TextWriter _out;
            private readonly TextWriter _err;

            private readonly bool _colorOut;
            private readonly bool _colorErr;

            // Progress reporting. Print up to 50 dots, if there's no meaningful (checkpoint) events.
            // At the end of 50 dots, print current metrics.
            private const int _maxDots = 50;
            private int _dots;

            public ConsoleWriter(ConsoleEnvironment parent, TextWriter outWriter, TextWriter errWriter)
            {
                Contracts.AssertValue(parent);
                Contracts.AssertValue(outWriter);
                Contracts.AssertValue(errWriter);
                _lock = new object();
                _parent = parent;
                _out = outWriter;
                _err = errWriter;

                _colorOut = outWriter == Console.Out;
                _colorErr = outWriter == Console.Error;
            }

            public void PrintMessage(IMessageSource sender, ChannelMessage msg)
            {
                bool isError = false;

                var messageColor = default(ConsoleColor);

                switch (msg.Kind)
                {
                    case ChannelMessageKind.Trace:
                        if (!sender.Verbose)
                            return;
                        messageColor = ConsoleColor.Gray;
                        break;
                    case ChannelMessageKind.Info:
                        break;
                    case ChannelMessageKind.Warning:
                        messageColor = ConsoleColor.Yellow;
                        isError = true;
                        break;
                    default:
                        Contracts.Assert(msg.Kind == ChannelMessageKind.Error);
                        messageColor = ConsoleColor.Red;
                        isError = true;
                        break;
                }

                lock (_lock)
                {
                    EnsureNewLine(isError);
                    var wr = isError ? _err : _out;
                    bool toColor = isError ? _colorOut : _colorErr;

                    if (toColor && msg.Kind != ChannelMessageKind.Info)
                        Console.ForegroundColor = messageColor;
                    string prefix = WriteAndReturnLinePrefix(msg.Sensitivity, wr);
                    var commChannel = sender as PipeBase<ChannelMessage>;
                    if (commChannel?.Verbose == true)
                        WriteHeader(wr, commChannel);
                    if (msg.Kind == ChannelMessageKind.Warning)
                        wr.Write("Warning: ");
                    _parent.PrintMessageNormalized(wr, msg.Message, true, prefix);
                    if (toColor)
                        Console.ResetColor();
                }
            }

            private string LinePrefix(MessageSensitivity sensitivity)
            {
                if (_parent._sensitivityFlags == MessageSensitivity.All || ((_parent._sensitivityFlags & sensitivity) != MessageSensitivity.None))
                    return null;
                return "SystemLog:";
            }

            private string WriteAndReturnLinePrefix(MessageSensitivity sensitivity, TextWriter writer)
            {
                string prefix = LinePrefix(sensitivity);
                if (prefix != null)
                    writer.Write(prefix);
                return prefix;
            }

            private void WriteHeader(TextWriter wr, PipeBase<ChannelMessage> commChannel)
            {
                Contracts.Assert(commChannel.Verbose);
                // REVIEW: Change this to use IndentingTextWriter.
                wr.Write(new string(' ', commChannel.Depth * 2));
                WriteName(wr, commChannel);
            }

            private void WriteName(TextWriter wr, ChannelProviderBase provider)
            {
                var channel = provider as Channel;
                if (channel != null)
                    WriteName(wr, channel.Parent);
                wr.Write("{0}: ", provider.ShortName);
            }

            public void ChannelStarted(Channel channel)
            {
                if (!channel.Verbose)
                    return;

                lock (_lock)
                {
                    EnsureNewLine();
                    WriteAndReturnLinePrefix(MessageSensitivity.None, _out);
                    WriteHeader(_out, channel);
                    _out.WriteLine("Started.");
                }
            }

            public void ChannelDisposed(Channel channel)
            {
                if (!channel.Verbose)
                    return;

                lock (_lock)
                {
                    EnsureNewLine();
                    WriteAndReturnLinePrefix(MessageSensitivity.None, _out);
                    WriteHeader(_out, channel);
                    _out.WriteLine("Finished.");
                    EnsureNewLine();
                    WriteAndReturnLinePrefix(MessageSensitivity.None, _out);
                    WriteHeader(_out, channel);
                    _out.WriteLine("Elapsed {0:c}.", channel.Watch.Elapsed);
                }
            }

            /// <summary>
            /// Query all progress and:
            /// * If there's any checkpoint/start/stop event, print all of them.
            /// * If there's none, print a dot.
            /// * If there's <see cref="_maxDots"/> dots, print the current status for all running calculations.
            /// </summary>
            public void GetAndPrintAllProgress(ProgressReporting.ProgressTracker progressTracker)
            {
                Contracts.AssertValue(progressTracker);

                var entries = progressTracker.GetAllProgress();
                if (entries.Count == 0)
                {
                    // There's no calculation running. Don't even print a dot.
                    return;
                }

                var checkpoints = entries.Where(
                    x => x.Kind != ProgressReporting.ProgressEvent.EventKind.Progress || x.ProgressEntry.IsCheckpoint);

                lock (_lock)
                {
                    bool anyCheckpoint = false;
                    foreach (var ev in checkpoints)
                    {
                        anyCheckpoint = true;
                        EnsureNewLine();
                        // We assume that things like status counters, which contain only things
                        // like loss function values, counts of rows, counts of items, etc., are
                        // not sensitive.
                        WriteAndReturnLinePrefix(MessageSensitivity.None, _out);
                        switch (ev.Kind)
                        {
                            case ProgressReporting.ProgressEvent.EventKind.Start:
                                PrintOperationStart(_out, ev);
                                break;
                            case ProgressReporting.ProgressEvent.EventKind.Stop:
                                PrintOperationStop(_out, ev);
                                break;
                            case ProgressReporting.ProgressEvent.EventKind.Progress:
                                _out.Write("[{0}] ", ev.Index);
                                PrintProgressLine(_out, ev);
                                break;
                        }
                    }
                    if (anyCheckpoint)
                    {
                        // At least one checkpoint has been printed, so there's no need for dots.
                        return;
                    }

                    if (PrintDot())
                    {
                        // We need to print an extended status line. At this point, every event should be
                        // a non-checkpoint progress event.
                        bool needPrepend = entries.Count > 1;
                        foreach (var ev in entries)
                        {
                            Contracts.Assert(ev.Kind == ProgressReporting.ProgressEvent.EventKind.Progress);
                            Contracts.Assert(!ev.ProgressEntry.IsCheckpoint);
                            if (needPrepend)
                            {
                                EnsureNewLine();
                                WriteAndReturnLinePrefix(MessageSensitivity.None, _out);
                                _out.Write("[{0}] ", ev.Index);
                            }
                            else
                            {
                                // This is the only case we are printing something at the end of the line of dots.
                                // So, we need to reset the dots counter.
                                _dots = 0;
                            }
                            PrintProgressLine(_out, ev);
                        }
                    }
                }
            }

            private static void PrintOperationStart(TextWriter writer, ProgressReporting.ProgressEvent ev)
            {
                writer.WriteLine("[{0}] '{1}' started.", ev.Index, ev.Name);
            }

            private static void PrintOperationStop(TextWriter writer, ProgressReporting.ProgressEvent ev)
            {
                writer.WriteLine("[{0}] '{1}' finished in {2}.", ev.Index, ev.Name, ev.EventTime - ev.StartTime);
            }

            private void PrintProgressLine(TextWriter writer, ProgressReporting.ProgressEvent ev)
            {
                // Elapsed time.
                var elapsed = ev.EventTime - ev.StartTime;
                if (elapsed.TotalMinutes < 1)
                    writer.Write("(00:{0:00.00})", elapsed.TotalSeconds);
                else if (elapsed.TotalHours < 1)
                    writer.Write("({0:00}:{1:00.0})", elapsed.Minutes, elapsed.TotalSeconds - 60 * elapsed.Minutes);
                else
                    writer.Write("({0:00}:{1:00}:{2:00})", elapsed.Hours, elapsed.Minutes, elapsed.Seconds);

                // Progress units.
                bool first = true;
                for (int i = 0; i < ev.ProgressEntry.Header.UnitNames.Length; i++)
                {
                    if (ev.ProgressEntry.Progress[i] == null)
                        continue;
                    writer.Write(first ? "\t" : ", ");
                    first = false;
                    writer.Write("{0}", ev.ProgressEntry.Progress[i]);
                    if (ev.ProgressEntry.ProgressLim[i] != null)
                        writer.Write("/{0}", ev.ProgressEntry.ProgressLim[i].Value);
                    writer.Write(" {0}", ev.ProgressEntry.Header.UnitNames[i]);
                }

                // Metrics.
                for (int i = 0; i < ev.ProgressEntry.Header.MetricNames.Length; i++)
                {
                    if (ev.ProgressEntry.Metrics[i] == null)
                        continue;
                    // REVIEW: print metrics prettier.
                    writer.Write("\t{0}: {1}", ev.ProgressEntry.Header.MetricNames[i], ev.ProgressEntry.Metrics[i].Value);
                }

                writer.WriteLine();
            }

            /// <summary>
            /// If we printed any dots so far, finish the line. This call is expected to be protected by _lock.
            /// </summary>
            private void EnsureNewLine(bool isError = false)
            {
                if (_dots == 0)
                    return;

                // If _err and _out is the same writer, we need to print new line as well.
                // If _out and _err writes to Console.Out and Console.Error respectively,
                // in the general user scenario they ends up with writing to the same underlying stream,.
                // so write a new line to the stream anyways.
                if (isError && _err != _out && (_out != Console.Out || _err != Console.Error))
                    return;

                _out.WriteLine();
                _dots = 0;
            }

            /// <summary>
            /// Print a progress dot. Returns whether it is 'time' to print more info. This call is expected
            /// to be protected by _lock.
            /// </summary>
            private bool PrintDot()
            {
                _out.Write(".");
                _dots++;
                return (_dots == _maxDots);
            }
        }

        private sealed class Channel : ChannelBase
        {
            public readonly Stopwatch Watch;
            public Channel(ConsoleEnvironment root, ChannelProviderBase parent, string shortName,
                Action<IMessageSource, ChannelMessage> dispatch)
                : base(root, parent, shortName, dispatch)
            {
                Watch = Stopwatch.StartNew();
                Root._consoleWriter.ChannelStarted(this);
            }

            protected override void DisposeCore()
            {
                Watch.Stop();
                Root._consoleWriter.ChannelDisposed(this);
                base.DisposeCore();
            }
        }

        private volatile ConsoleWriter _consoleWriter;
        private readonly MessageSensitivity _sensitivityFlags;

        /// <summary>
        /// Create an ML.NET <see cref="IHostEnvironment"/> for local execution, with console feedback.
        /// </summary>
        /// <param name="seed">Random seed. Set to <c>null</c> for a non-deterministic environment.</param>
        /// <param name="verbose">Set to <c>true</c> for fully verbose logging.</param>
        /// <param name="sensitivity">Allowed message sensitivity.</param>
        /// <param name="conc">Concurrency level. Set to 1 to run single-threaded. Set to 0 to pick automatically.</param>
        /// <param name="outWriter">Text writer to print normal messages to.</param>
        /// <param name="errWriter">Text writer to print error messages to.</param>
        public ConsoleEnvironment(int? seed = null, bool verbose = false,
            MessageSensitivity sensitivity = MessageSensitivity.All, int conc = 0,
            TextWriter outWriter = null, TextWriter errWriter = null)
            : this(RandomUtils.Create(seed), verbose, sensitivity, conc, outWriter, errWriter)
        {
        }

        // REVIEW: do we really care about custom random? If we do, let's make this ctor public.
        /// <summary>
        /// Create an ML.NET environment for local execution, with console feedback.
        /// </summary>
        /// <param name="rand">An custom source of randomness to use in the environment.</param>
        /// <param name="verbose">Set to <c>true</c> for fully verbose logging.</param>
        /// <param name="sensitivity">Allowed message sensitivity.</param>
        /// <param name="conc">Concurrency level. Set to 1 to run single-threaded. Set to 0 to pick automatically.</param>
        /// <param name="outWriter">Text writer to print normal messages to.</param>
        /// <param name="errWriter">Text writer to print error messages to.</param>
        private ConsoleEnvironment(IRandom rand, bool verbose = false,
            MessageSensitivity sensitivity = MessageSensitivity.All, int conc = 0,
            TextWriter outWriter = null, TextWriter errWriter = null)
            : base(rand, verbose, conc, nameof(ConsoleEnvironment))
        {
            Contracts.CheckValueOrNull(outWriter);
            Contracts.CheckValueOrNull(errWriter);
            _consoleWriter = new ConsoleWriter(this, outWriter ?? Console.Out, errWriter ?? Console.Error);
            _sensitivityFlags = sensitivity;
            AddListener<ChannelMessage>(PrintMessage);
        }

        /// <summary>
        /// Pull running calculations for their progress and output all messages to the console.
        /// If no messages are available, print a dot.
        /// If a specified number of dots are printed, print an ad-hoc status of all running calculations.
        /// </summary>
        public void PrintProgress()
        {
            Root._consoleWriter.GetAndPrintAllProgress(ProgressTracker);
        }

        private void PrintMessage(IMessageSource src, ChannelMessage msg)
        {
            Root._consoleWriter.PrintMessage(src, msg);
        }

        protected override IFileHandle CreateTempFileCore(IHostEnvironment env, string suffix = null, string prefix = null)
        {
            // Prefix with "TLC_".
            return base.CreateTempFileCore(env, suffix, "TLC_" + prefix);
        }

        protected override IHost RegisterCore(HostEnvironmentBase<ConsoleEnvironment> source, string shortName, string parentFullName, IRandom rand, bool verbose, int? conc)
        {
            Contracts.AssertValue(rand);
            Contracts.AssertValueOrNull(parentFullName);
            Contracts.AssertNonEmpty(shortName);
            Contracts.Assert(source == this || source is Host);
            return new Host(source, shortName, parentFullName, rand, verbose, conc);
        }

        protected override IChannel CreateCommChannel(ChannelProviderBase parent, string name)
        {
            Contracts.AssertValue(parent);
            Contracts.Assert(parent is ConsoleEnvironment);
            Contracts.AssertNonEmpty(name);
            return new Channel(this, parent, name, GetDispatchDelegate<ChannelMessage>());
        }

        protected override IPipe<TMessage> CreatePipe<TMessage>(ChannelProviderBase parent, string name)
        {
            Contracts.AssertValue(parent);
            Contracts.Assert(parent is ConsoleEnvironment);
            Contracts.AssertNonEmpty(name);
            return new Pipe<TMessage>(parent, name, GetDispatchDelegate<TMessage>());
        }

        /// <summary>
        /// Redirects the channel output through the specified writers.
        /// </summary>
        /// <remarks>This method is not thread-safe.</remarks>
        internal IDisposable RedirectChannelOutput(TextWriter newOutWriter, TextWriter newErrWriter)
        {
            Contracts.CheckValue(newOutWriter, nameof(newOutWriter));
            Contracts.CheckValue(newErrWriter, nameof(newErrWriter));
            return new OutputRedirector(this, newOutWriter, newErrWriter);
        }

        internal void ResetProgressChannel()
        {
            ProgressTracker.Reset();
        }

        private sealed class OutputRedirector : IDisposable
        {
            private readonly ConsoleEnvironment _root;
            private ConsoleWriter _oldConsoleWriter;
            private readonly ConsoleWriter _newConsoleWriter;

            public OutputRedirector(ConsoleEnvironment env, TextWriter newOutWriter, TextWriter newErrWriter)
            {
                Contracts.AssertValue(env);
                Contracts.AssertValue(newOutWriter);
                Contracts.AssertValue(newErrWriter);
                _root = env.Root;
                _newConsoleWriter = new ConsoleWriter(_root, newOutWriter, newErrWriter);
                _oldConsoleWriter = Interlocked.Exchange(ref _root._consoleWriter, _newConsoleWriter);
                Contracts.AssertValue(_oldConsoleWriter);
            }

            public void Dispose()
            {
                if (_oldConsoleWriter == null)
                    return;

                Contracts.Assert(_root._consoleWriter == _newConsoleWriter);
                _root._consoleWriter = _oldConsoleWriter;
                _oldConsoleWriter = null;
            }
        }

        private sealed class Host : HostBase
        {
            public Host(HostEnvironmentBase<ConsoleEnvironment> source, string shortName, string parentFullName, IRandom rand, bool verbose, int? conc)
                : base(source, shortName, parentFullName, rand, verbose, conc)
            {
                IsCancelled = source.IsCancelled;
            }

            protected override IChannel CreateCommChannel(ChannelProviderBase parent, string name)
            {
                Contracts.AssertValue(parent);
                Contracts.Assert(parent is Host);
                Contracts.AssertNonEmpty(name);
                return new Channel(Root, parent, name, GetDispatchDelegate<ChannelMessage>());
            }

            protected override IPipe<TMessage> CreatePipe<TMessage>(ChannelProviderBase parent, string name)
            {
                Contracts.AssertValue(parent);
                Contracts.Assert(parent is Host);
                Contracts.AssertNonEmpty(name);
                return new Pipe<TMessage>(parent, name, GetDispatchDelegate<TMessage>());
            }

            protected override IHost RegisterCore(HostEnvironmentBase<ConsoleEnvironment> source, string shortName, string parentFullName, IRandom rand, bool verbose, int? conc)
            {
                return new Host(source, shortName, parentFullName, rand, verbose, conc);
            }
        }
    }
}
