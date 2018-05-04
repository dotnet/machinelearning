// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// The progress reporting classes used by <see cref="HostEnvironmentBase{THostEnvironmentBase}"/> descendants.
    /// </summary>
    public static class ProgressReporting
    {
        /// <summary>
        /// The progress channel for <see cref="TlcEnvironment"/>.
        /// This is coupled with a <see cref="ProgressTracker"/> that aggregates all events and returns them on demand.
        /// </summary>
        public sealed class ProgressChannel : IProgressChannel
        {
            private readonly IExceptionContext _ectx;

            private readonly string _name;

            /// <summary>
            /// The pair of (header, fill action) is updated atomically.
            /// </summary>
            private Tuple<ProgressHeader, Action<IProgressEntry>> _headerAndAction;

            /// <summary>
            /// Normally this should be readonly field, but we want to null it in Dispose to prevent memory leaking.
            /// </summary>
            private ProgressTracker _tracker;

            private readonly ConcurrentDictionary<int, SubChannel> _subChannels;
            private volatile int _maxSubId;
            private bool _isDisposed;

            public string Name { get { return _name; } }

            /// <summary>
            /// Initialize a <see cref="ProgressChannel"/> for the process identified by <paramref name="computationName"/>.
            /// </summary>
            /// <param name="ectx">The exception context.</param>
            /// <param name="tracker">The tracker to couple with.</param>
            /// <param name="computationName">The computation name.</param>
            public ProgressChannel(IExceptionContext ectx, ProgressTracker tracker, string computationName)
            {
                Contracts.CheckValueOrNull(ectx);
                _ectx = ectx;
                _ectx.CheckValue(tracker, nameof(tracker));
                _ectx.CheckNonEmpty(computationName, nameof(computationName));

                _name = computationName;
                _tracker = tracker;
                _subChannels = new ConcurrentDictionary<int, SubChannel>();
                _maxSubId = 0;

                _headerAndAction = Tuple.Create<ProgressHeader, Action<IProgressEntry>>(new ProgressHeader(null), null);
                Start();
            }

            public void SetHeader(ProgressHeader header, Action<IProgressEntry> fillAction)
            {
                _headerAndAction = Tuple.Create(header, fillAction);
            }

            public void Checkpoint(params double?[] values)
            {
                _ectx.AssertValueOrNull(values);
                _ectx.Check(!_isDisposed, "Can't report checkpoints after disposing");
                var entry = new ProgressEntry(true, _headerAndAction.Item1);

                int n = Utils.Size(values);
                int iSrc = 0;

                for (int iDst = 0; iDst < entry.Metrics.Length && iSrc < n;)
                    entry.Metrics[iDst++] = values[iSrc++];

                for (int iDst = 0; iDst < entry.Progress.Length && iSrc < n;)
                    entry.Progress[iDst++] = values[iSrc++];

                for (int iDst = 0; iDst < entry.ProgressLim.Length && iSrc < n;)
                {
                    var lim = values[iSrc++];
                    if (Double.IsNaN(lim.GetValueOrDefault()))
                        lim = null;
                    entry.ProgressLim[iDst++] = lim;
                }

                _ectx.Check(iSrc == n, "Too many values provided in Checkpoint");
                _tracker.Log(this, ProgressEvent.EventKind.Progress, entry);
            }

            private void Start()
            {
                _tracker.Log(this, ProgressEvent.EventKind.Start, null);
            }

            private void Stop()
            {
                _tracker.Log(this, ProgressEvent.EventKind.Stop, null);
            }

            public void Dispose()
            {
                if (_isDisposed)
                    return;
                _isDisposed = true;
                Stop();

                Contracts.Assert(_subChannels.Count == 0);
                // The 'get progress' action could potentially reference additional objects via closures.
                // This constitutes a memory leak potential, if the progress tracker object is retained for longer than the operation was running.
                _headerAndAction = null;
                _tracker = null;
            }

            /// <summary>
            /// Pull the current progress by invoking the fill delegate, if any.
            /// </summary>
            public ProgressEntry GetProgress()
            {
                // Make sure we get header and action from the same pair, even if outdated.
                var cache = _headerAndAction;
                var fillAction = cache.Item2;
                var entry = new ProgressEntry(false, cache.Item1);

                if (fillAction == null)
                    Contracts.Assert(entry.Header.MetricNames.Length == 0 && entry.Header.UnitNames.Length == 0);
                else
                    fillAction(entry);

                return BuildJointEntry(entry);
            }

            public IProgressChannel StartProgressChannel(string name)
            {
                return StartProgressChannel(1);
            }

            private IProgressChannel StartProgressChannel(int level)
            {
#pragma warning disable 420 // Interlocked with volatile.
                var newId = Interlocked.Increment(ref _maxSubId);
#pragma warning restore 420
                return new SubChannel(this, level, newId);
            }

            private void SubChannelStopped(int id)
            {
                SubChannel channel;
                _subChannels.TryRemove(id, out channel);
                // Duplicate removal is OK, so we don't inspect return value.
            }

            private void SubChannelStarted(int id, SubChannel channel)
            {
                var res = _subChannels.GetOrAdd(id, channel);
                Contracts.Assert(res == channel);
            }

            private ProgressEntry BuildJointEntry(ProgressEntry rootEntry)
            {
                if (_maxSubId == 0 || _subChannels.Count == 0)
                    return rootEntry;

                // REVIEW: consider caching the headers, in case the sub-reporters haven't changed.
                // This is not anticipated to be a perf-critical path though.
                var hProgress = new List<string>();
                var hMetrics = new List<string>();
                var progress = new List<double?>();
                var progressLim = new List<double?>();
                var metrics = new List<double?>();

                hProgress.AddRange(rootEntry.Header.UnitNames);
                hMetrics.AddRange(rootEntry.Header.MetricNames);
                progress.AddRange(rootEntry.Progress);
                progressLim.AddRange(rootEntry.ProgressLim);
                metrics.AddRange(rootEntry.Metrics);

                foreach (var subChannel in _subChannels.Values.ToArray().OrderBy(x => x.Level))
                {
                    var entry = subChannel.GetProgress();
                    hProgress.AddRange(entry.Header.UnitNames);
                    hMetrics.AddRange(entry.Header.MetricNames);
                    progress.AddRange(entry.Progress);
                    progressLim.AddRange(entry.ProgressLim);
                    metrics.AddRange(entry.Metrics);
                }

                var jointEntry = new ProgressEntry(false, new ProgressHeader(hMetrics.ToArray(), hProgress.ToArray()));
                progress.CopyTo(jointEntry.Progress);
                progressLim.CopyTo(jointEntry.ProgressLim);
                metrics.CopyTo(jointEntry.Metrics);
                return jointEntry;
            }

            /// <summary>
            /// This is a 'derived' or 'subordinate' progress channel.
            /// 
            /// The subordinates' Start/Stop events and checkpoints will not be propagated. 
            /// When the status is requested, all of the subordinate channels are also invoked,
            /// and the resulting metrics are then returned in the order of their 'subordinate level'.
            /// If there's more than one channel with the same level, the order is not defined.
            /// </summary>
            private sealed class SubChannel : IProgressChannel
            {
                private readonly ProgressChannel _root;
                private readonly int _id;
                // The 'depth' of subordinate.
                private readonly int _level;

                /// <summary>
                /// The pair of (header, fill action) is updated atomically.
                /// </summary>
                private Tuple<ProgressHeader, Action<IProgressEntry>> _headerAndAction;

                public int Level { get { return _level; } }

                /// <summary>
                /// Pull the current progress by invoking the fill delegate, if any.
                /// </summary>
                public ProgressEntry GetProgress()
                {
                    // Make sure we get header and action from the same pair, even if outdated.
                    var cache = _headerAndAction;
                    var fillAction = cache.Item2;
                    var entry = new ProgressEntry(false, cache.Item1);

                    if (fillAction == null)
                        Contracts.Assert(entry.Header.MetricNames.Length == 0 && entry.Header.UnitNames.Length == 0);
                    else
                        fillAction(entry);
                    return entry;
                }

                public SubChannel(ProgressChannel root, int id, int level)
                {
                    Contracts.AssertValue(root);
                    Contracts.Assert(level >= 0);
                    _root = root;
                    _id = id;
                    _level = level;
                    _headerAndAction = Tuple.Create<ProgressHeader, Action<IProgressEntry>>(new ProgressHeader(null), null);
                    Start();
                }

                public IProgressChannel StartProgressChannel(string name)
                {
                    return _root.StartProgressChannel(_level + 1);
                }

                public void Dispose()
                {
                    Stop();
                }

                public void SetHeader(ProgressHeader header, Action<IProgressEntry> fillAction)
                {
                    _headerAndAction = Tuple.Create(header, fillAction);
                }

                private void Start()
                {
                    _root.SubChannelStarted(_id, this);
                }

                private void Stop()
                {
                    _root.SubChannelStopped(_id);
                }

                public void Checkpoint(params Double?[] values)
                {
                    // We are ignoring all checkpoints from subordinates.
                    // REVIEW: maybe this could be changed in the future. Right now it seems that 
                    // this limitation is reasonable.
                }
            }
        }

        /// <summary>
        /// This class listens to the progress reporting channels, caches all checkpoints and
        /// start/stop events and, on demand, requests current progress on all active calculations.
        /// 
        /// The public methods of this class should only be called from one thread.
        /// </summary>
        public sealed class ProgressTracker
        {
            private readonly IExceptionContext _ectx;
            private readonly object _lock;

            /// <summary>
            /// Log of pending events.
            /// </summary>
            private readonly ConcurrentQueue<ProgressEvent> _pendingEvents;

            /// <summary>
            /// For each calculation, its properties.
            /// This list is protected by <see cref="_lock"/>, and it's updated every time a new calculation starts.
            /// The entries are cleaned up when the start and stop events are reported (that is, after the first 
            /// pull request after the calculation's 'Stop' event).
            /// </summary>
            private readonly List<CalculationInfo> _infos;

            /// <summary>
            /// This is a 'process index' that gets incremented whenever a new calculation is started.
            /// </summary>
            private int _index;

            /// <summary>
            /// The set of used process names.
            /// </summary>
            private readonly HashSet<string> _namesUsed;

            /// <summary>
            /// This class is an 'event log' for one calculation. 
            /// 
            /// Every time a calculation is 'started', it gets its own log, so if there are multiple 'start' calls,
            /// there will be multiple logs.
            /// </summary>
            private sealed class CalculationInfo
            {
                /// <summary>
                /// Auto-assigned index to serve as a unique ID.
                /// </summary>
                public readonly int Index;

                /// <summary>
                /// Name is auto-modified from the calculation name provided by the pipe.
                /// </summary>
                public readonly string Name;

                public readonly DateTime StartTime;

                public readonly ProgressChannel Channel;

                /// <summary>
                /// A log of pending checkpoint entries.
                /// </summary>
                public readonly ConcurrentQueue<KeyValuePair<DateTime, ProgressEntry>> PendingCheckpoints;

                /// <summary>
                /// Whether the calculation has finished.
                /// </summary>
                public bool IsFinished;

                public CalculationInfo(int index, string name, ProgressChannel channel)
                {
                    Contracts.Assert(index > 0);
                    Contracts.AssertNonEmpty(name);
                    Contracts.AssertValue(channel);

                    Index = index;
                    Name = name;
                    PendingCheckpoints = new ConcurrentQueue<KeyValuePair<DateTime, ProgressEntry>>();
                    StartTime = DateTime.Now;
                    Channel = channel;
                }
            }

            public ProgressTracker(IExceptionContext ectx)
            {
                Contracts.CheckValue(ectx, nameof(ectx));
                _ectx = ectx;
                _lock = new object();
                _pendingEvents = new ConcurrentQueue<ProgressEvent>();
                _infos = new List<CalculationInfo>();
                _namesUsed = new HashSet<string>();
            }

            public void Log(ProgressChannel source, ProgressEvent.EventKind kind, ProgressEntry entry)
            {
                _ectx.AssertValue(source);
                _ectx.AssertValueOrNull(entry);

                if (kind == ProgressEvent.EventKind.Start)
                {
                    _ectx.Assert(entry == null);
                    lock (_lock)
                    {
                        // Figure out an appropriate name.
                        int i = 1;
                        var name = source.Name;
                        string nameCandidate = name;
                        while (!_namesUsed.Add(nameCandidate))
                        {
                            i++;
                            nameCandidate = string.Format("{0} #{1}", name, i);
                        }
                        var newInfo = new CalculationInfo(++_index, nameCandidate, source);
                        _infos.Add(newInfo);
                        _pendingEvents.Enqueue(new ProgressEvent(newInfo.Index, newInfo.Name, newInfo.StartTime, ProgressEvent.EventKind.Start));
                        return;
                    }
                }

                // Not a start event, so we won't modify the _infos.
                CalculationInfo info;
                lock (_lock)
                {
                    info = _infos.FirstOrDefault(x => x.Channel == source);
                    if (info == null)
                        throw _ectx.Except("Event sent after the calculation lifetime expired.");
                }
                switch (kind)
                {
                case ProgressEvent.EventKind.Stop:
                    _ectx.Assert(entry == null);
                    info.IsFinished = true;
                    _pendingEvents.Enqueue(new ProgressEvent(info.Index, info.Name, info.StartTime, ProgressEvent.EventKind.Stop));
                    break;
                default:
                    _ectx.Assert(entry != null);
                    _ectx.Assert(kind == ProgressEvent.EventKind.Progress);
                    _ectx.Assert(!info.IsFinished);
                    _pendingEvents.Enqueue(new ProgressEvent(info.Index, info.Name, info.StartTime, entry));
                    break;
                }
            }

            /// <summary>
            /// Get progress reports from all current calculations. 
            /// For every calculation the following events will be returned:
            /// * A start event.
            /// * Each checkpoint.
            /// * If the calculation is finished, the stop event. 
            /// 
            /// Each of the above events will be returned exactly once.
            /// If, for one calculation, there's no events in the above categories, the tracker will
            /// request ('pull') the current progress and return this as an event.
            /// </summary>
            public List<ProgressEvent> GetAllProgress()
            {
                var list = new List<ProgressEvent>();
                var seen = new HashSet<int>();
                ProgressEvent cur;
                while (_pendingEvents.TryDequeue(out cur))
                {
                    seen.Add(cur.Index);
                    list.Add(cur);
                }

                // Get unseen calculations to pull progress from.
                CalculationInfo[] unseen;
                lock (_lock)
                {
                    unseen = _infos.Where(x => !seen.Contains(x.Index)).ToArray();
                    _infos.RemoveAll(x => x.IsFinished);
                }

                foreach (var info in unseen)
                {
                    // The calculation might finish while we're inside the GetAllProgress. We will report the finish
                    // event in the next status, but we make a half-hearted effort not to call the delegate on a finished
                    // calculation.
                    if (info.IsFinished)
                        continue;

                    var entry = info.Channel.GetProgress();
                    list.Add(new ProgressEvent(info.Index, info.Name, info.StartTime, entry));
                }

                return list;
            }
        }

        /// <summary>
        /// An array-backed implementation of <see cref="IProgressEntry"/>.
        /// </summary>
        public sealed class ProgressEntry : IProgressEntry
        {
            /// <summary>
            /// The header (names of metrics and units).
            /// The contents of the header should be treated as read-only. The calculation itself doesn't even
            /// need to access the header, since it will know it anyway.
            /// </summary>
            public readonly ProgressHeader Header;

            /// <summary>
            /// Whether the progress entry is a 'checkpoint' (that is, it's being pushed by the component).
            /// </summary>
            public readonly bool IsCheckpoint;

            /// <summary>
            /// The actual progress (amount of completed units), in the units that are contained in the header.
            /// Parallel to the header's <see cref="ProgressHeader.UnitNames"/>. Null value indicates 'not applicable now'.
            /// 
            /// The computation should not modify these arrays directly, and instead rely on <see cref="SetMetric"/>,
            /// <see cref="SetProgress(int,double)"/> and <see cref="SetProgress(int,double,double)"/>.
            /// </summary>
            public readonly Double?[] Progress;

            /// <summary>
            /// The lim values of each progress unit. 
            /// Parallel to the header's <see cref="ProgressHeader.UnitNames"/>. Null value indicates unbounded or unknown.
            /// </summary>
            public readonly Double?[] ProgressLim;

            /// <summary>
            /// The reported metrics. Parallel to the header's <see cref="ProgressHeader.MetricNames"/>.
            /// Null value indicates unknown.
            /// </summary>
            public readonly Double?[] Metrics;

            /// <summary>
            /// Set the progress value for the index <paramref name="index"/> to <paramref name="value"/>,
            /// and the limit value for the progress becomes 'unknown'.
            /// </summary>
            public void SetProgress(int index, Double value)
            {
                Contracts.Check(0 <= index && index < Progress.Length);
                Progress[index] = value;
                ProgressLim[index] = null;
            }

            /// <summary>
            /// Set the progress value for the index <paramref name="index"/> to <paramref name="value"/>,
            /// and the limit value to <paramref name="lim"/>.
            /// </summary>
            public void SetProgress(int index, Double value, Double lim)
            {
                Contracts.Check(0 <= index && index < Progress.Length);
                Contracts.Assert(0 <= index && index < Progress.Length);
                Progress[index] = value;
                ProgressLim[index] = Double.IsNaN(lim) ? (Double?)null : lim;
            }

            /// <summary>
            /// Sets the metric with index <paramref name="index"/> to <paramref name="value"/>.
            /// </summary>
            public void SetMetric(int index, Double value)
            {
                Contracts.Check(0 <= index && index < Metrics.Length);
                Metrics[index] = value;
            }

            /// <summary>
            /// Creates the progress entry corresponding to a given header.
            /// </summary>
            public ProgressEntry(bool isCheckpoint, ProgressHeader header)
            {
                Contracts.CheckValue(header, nameof(header));
                Header = header;
                IsCheckpoint = isCheckpoint;
                Progress = new Double?[header.UnitNames.Length];
                ProgressLim = new Double?[header.UnitNames.Length];
                Metrics = new Double?[header.MetricNames.Length];
            }
        }

        /// <summary>
        /// An event about calculation progress. It could be either start/stop of the calculation, or a progress entry.
        /// </summary>
        public sealed class ProgressEvent
        {
            // REVIEW: Separate kind for checkpoint?
            public enum EventKind
            {
                Start,
                Progress,
                Stop
            }

            public readonly int Index;
            public readonly string Name;
            // REVIEW: Maybe switch to the stopwatch-based wall clock?
            public readonly DateTime StartTime;
            public readonly DateTime EventTime;
            public readonly EventKind Kind;
            public readonly ProgressEntry ProgressEntry;

            public ProgressEvent(int index, string name, DateTime startTime, ProgressEntry entry)
            {
                Contracts.CheckParam(index >= 0, nameof(index));
                Contracts.CheckNonEmpty(name, nameof(name));
                Contracts.CheckValue(entry, nameof(entry));

                Index = index;
                Name = name;
                StartTime = startTime;
                EventTime = DateTime.Now;
                Kind = EventKind.Progress;
                ProgressEntry = entry;
            }

            public ProgressEvent(int index, string name, DateTime startTime, EventKind kind)
            {
                Contracts.CheckParam(index >= 0, nameof(index));
                Contracts.CheckNonEmpty(name, nameof(name));

                Index = index;
                Name = name;
                StartTime = startTime;
                EventTime = DateTime.Now;
                Kind = kind;
                ProgressEntry = null;
            }
        }
    }
}
