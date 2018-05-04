// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Text;
using System.Threading;

namespace Microsoft.ML.Runtime.FastTree.Internal
{
    using Stopwatch = System.Diagnostics.Stopwatch;

    public enum TimerEvent
    {
        TotalInitialization,
        TotalTrain,
        TotalSave,

        Iteration,
        Test,

        InitializeLoadDatasets,
        InitializeLabels,
        InitializeFirstInput,
        InitializeTraining,
        InitializeTests,

        NewPhasePreparation,

        ObjectiveFunctionGetDerivatives,
        TreeLearnerGetTree,
        TreeLearnerSyncTree,
        TreeLearnerAdjustTreeOutputs,
        GradientBoostingAddOutputsToScores,
        UpdateScores,

        FindBestSplit,
        FindBestSplitOfRoot,
        FindBestSplitOfSiblings,
        FindBestSplitInit,
        DocumentPartitioningConstruction,
        DocumentPartitioningSplit,
        CalculateLeafSplitCandidates,
        AnyThreadTask,
        AllThreadTask,
        ConcatBins,
        SparseConstruction,
        LoadFeature,
        ReadBytes,
        ConstructFromByteArray,
        Sumup,
        SumupDense10,
        SumupCppDense,
        SumupSparse,
        SumupRepeat,
        SumupSegment,
        AdHocTesting,

        ThresholdFinding,
        HistogramSyncup,
        HistogramSyncUpStaging1,
        HistogramSyncUpStaging2,
        HistogramSyncUpStaging3,

        FeatureVectorAggregation,
        FirstRoundAggregation,
        FirstRoundAggregationPerFeature,
        FirstRoundBroadcastSplitInfo,
        SecondRoundAggregation,
        SecondRoundAggregationPerFeature,
        SecondRoundBroadcastSplitInfo,

        MessageSerialize,
        MessageDeserialize,
        NetBroadcast,
        NetSending,
        NetReceiving,
        DataSend,
        DataReceive,
        DataWaitForDeserialization,
        OnReceivedData,

        // For parallel fasttree in TLC++
        NetworkSend,
        NetworkReceive,
        AllGather,
        ReduceScatter,
        AllReduce,

        GlobalFeatureBinSync,
        GlobalHistogramMerge,
        GlobalBestSplitSync,
        GlobalMeanForLeafOutput,
        GlobalVoting,
        HistogramCaching

    }

    public enum CountEvent
    {
        NetworkSend,
        NetworkReceive,
        AllGather,
        ReduceScatter,
        AllReduce,

        GlobalFeatureBinSync,
        GlobalHistogramMerge,
        GlobalBestSplitSync,
        GlobalMeanForLeafOutput,
        GlobalVoting
    }

    /// <summary>
    /// Static class for timing events.
    /// </summary>
    public static class Timer
    {
        private static TimerState _state;

        private sealed class TimerState
        {
            public readonly Stopwatch Watch;
            public readonly long[] TickTotals;
            public readonly int[] NumCalls;
            public readonly int MaxEventNameLen;
            public readonly long[] CountTotals;

            public TimerState()
            {
                TickTotals = new long[Enum.GetValues(typeof(TimerEvent)).Length];
                CountTotals = new long[Enum.GetValues(typeof(CountEvent)).Length];
                Watch = new Stopwatch();
                Watch.Start();
                NumCalls = new int[TickTotals.Length];
                foreach (string name in Enum.GetNames(typeof(TimerEvent)))
                {
                    if (name.Length > MaxEventNameLen)
                        MaxEventNameLen = name.Length;
                }
                foreach (string name in Enum.GetNames(typeof(CountEvent)))
                {
                    if (name.Length > MaxEventNameLen)
                        MaxEventNameLen = name.Length;
                }
            }

            /// <summary>
            /// Gets a string summary of the total times.
            /// </summary>
            public override string ToString()
            {
                var sb = new StringBuilder();

                long total = Watch.ElapsedTicks;

                string padded = "Name".PadRight(MaxEventNameLen);

                sb.AppendFormat("{0} {1,10}{2,10}{3,8}{4,11}\n", padded, "Time", "%", "#Calls", "Time/Call");
                foreach (TimerEvent n in Enum.GetValues(typeof(TimerEvent)))
                {
                    double time = (double)TickTotals[(int)n] / Stopwatch.Frequency;
                    int numCalls = NumCalls[(int)n];
                    double perc = 100.0 * (double)TickTotals[(int)n] / total;

                    double timePerCall = (numCalls > 0) ? time / numCalls : 0;

                    padded = n.ToString().PadRight(MaxEventNameLen);

                    sb.AppendFormat("{0} {1,10:0.000}{2,9:00.00}%{3,8}{4,11:0.000}\n", padded, time, perc, numCalls, timePerCall);
                }
                sb.AppendFormat("Count Statistics:\n");
                padded = "Name".PadRight(MaxEventNameLen);
                sb.AppendFormat("{0} {1,10}\n", padded, "Accumulate");
                foreach (CountEvent n in Enum.GetValues(typeof(CountEvent)))
                {
                    double count = _state.CountTotals[(int)n];

                    padded = n.ToString().PadRight(MaxEventNameLen);

                    sb.AppendFormat("{0} {1,10}\n", padded, count);
                }
                return sb.ToString();
            }
        }

        /// <summary>
        /// Returns the total number of CPU ticks spent in the specified timer so far.
        /// </summary>
        internal static long GetTicks(TimerEvent e)
        {
            return _state.TickTotals == null ? 0 : _state.TickTotals[(int)e];
        }

        public static long GetCounts(CountEvent e)
        {
            return _state.CountTotals[(int)e];
        }

        private static void EnsureValid()
        {
            if (_state == null)
                Interlocked.CompareExchange(ref _state, new TimerState(), null);
        }

        /// <summary>
        /// Creates a timed event which, when disposed, adds to the total time of that event type.
        /// </summary>
        /// <param name="e">The type of event</param>
        /// <returns>A timed event</returns>
        public static TimedEvent Time(TimerEvent e)
        {
            EnsureValid();
            return new TimedEvent(_state.Watch.ElapsedTicks, e);
        }

        public static void Count(long counts, CountEvent e)
        {
            Interlocked.Add(ref _state.CountTotals[(int)e], counts);
        }

        /// <summary>
        /// An object which, when disposed, adds to the total time of that event type.
        /// </summary>
        public sealed class TimedEvent : IDisposable
        {
            private readonly long _ticksBegin;
            private readonly TimerEvent _event;

            public TimedEvent(long ticks, TimerEvent evt)
            {
                _ticksBegin = ticks;
                _event = evt;
            }

            #region IDisposable Members

            void IDisposable.Dispose()
            {
                Interlocked.Add(ref _state.TickTotals[(int)_event], _state.Watch.ElapsedTicks - _ticksBegin);
                Interlocked.Increment(ref _state.NumCalls[(int)_event]);
            }

            #endregion
        }

        /// <summary>
        /// Gets a string summary of the total times.
        /// </summary>
        /// <returns></returns>
        public static string GetString()
        {
            EnsureValid();
            return _state.ToString();
        }
    }
}
