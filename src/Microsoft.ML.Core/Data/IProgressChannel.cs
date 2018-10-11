// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Runtime
{
    /// <summary>
    /// This is a factory interface for <see cref="IProgressChannel"/>.
    /// Both <see cref="IHostEnvironment"/> and <see cref="IProgressChannel"/> implement this interface,
    /// to allow for nested progress reporters.
    ///
    /// REVIEW: make <see cref="IChannelProvider"/> implement this, instead of the environment?
    /// </summary>
    public interface IProgressChannelProvider
    {
        /// <summary>
        /// Create a progress channel for a computation named <paramref name="name"/>.
        /// </summary>
        IProgressChannel StartProgressChannel(string name);
    }

    /// <summary>
    /// A common interface for progress reporting.
    /// It is expected that the progress channel interface is used from only one thread.
    ///
    /// Supported workflow:
    /// 1) Create the channel via <see cref="IProgressChannelProvider.StartProgressChannel"/>.
    /// 2) Call <see cref="SetHeader"/> as many times as desired (including 0).
    ///       Each call to <see cref="SetHeader"/> supersedes the previous one.
    /// 3) Report checkpoints (0 or more) by calling <see cref="Checkpoint"/>.
    /// 4) Repeat steps 2-3 as often as necessary.
    /// 5) Dispose the channel.
    /// </summary>
    public interface IProgressChannel : IProgressChannelProvider, IDisposable
    {
        /// <summary>
        /// Set up the reporting structure:
        /// - Set the 'header' of the progress reports, defining which progress units and metrics are going to be reported.
        /// - Provide a thread-safe delegate to be invoked whenever anyone needs to know the progress.
        ///
        /// It is acceptable to call <see cref="SetHeader"/> multiple times (or none), regardless of whether the calculation is running
        /// or not. Because of synchronization, the computation should not deny calls to the 'old' <paramref name="fillAction"/>
        /// delegates even after a new one is provided.
        /// </summary>
        /// <param name="header">The header object.</param>
        /// <param name="fillAction">The delegate to provide actual progress. The <see cref="IProgressEntry"/> parameter of
        /// the delegate will correspond to the provided <paramref name="header"/>.</param>
        void SetHeader(ProgressHeader header, Action<IProgressEntry> fillAction);

        /// <summary>
        /// Submit a 'checkpoint' entry. These entries are guaranteed to be delivered to the progress listener,
        /// if it is interested. Typically, this would contain some intermediate metrics, that are only calculated
        /// at certain moments ('checkpoints') of the computation.
        ///
        /// For example, SDCA may report a checkpoint every time it computes the loss, or LBFGS may report a checkpoint
        /// every iteration.
        ///
        /// The only parameter, <paramref name="values"/>, is interpreted in the following fashion:
        /// * First MetricNames.Length items, if present, are metrics.
        /// * Subsequent ProgressNames.Length items, if present, are progress units.
        /// * Subsequent ProgressNames.Length items, if present, are progress limits.
        /// * If any more values remain, an exception is thrown.
        /// </summary>
        /// <param name="values">The metrics, progress units and progress limits.</param>
        void Checkpoint(params Double?[] values);
    }

    /// <summary>
    /// This is the 'header' of the progress report.
    /// </summary>
    public sealed class ProgressHeader
    {
        /// <summary>
        /// These are the names of the progress 'units', from the least granular to the most granular.
        /// For example, neural network might have {'epoch', 'example'} and FastTree might have {'tree', 'split', 'feature'}.
        /// Will never be null, but can be empty.
        /// </summary>
        public readonly string[] UnitNames;

        /// <summary>
        /// These are the names of the reported metrics. For example, this could be the 'loss', 'weight updates/sec' etc.
        /// Will never be null, but can be empty.
        /// </summary>
        public readonly string[] MetricNames;

        /// <summary>
        /// Initialize the header. This will take ownership of the arrays.
        /// Both arrays can be null, even simultaneously. This 'empty' header indicated that the calculation doesn't report
        /// any units of progress, but the tracker can still track start, stop and elapsed time. Of course, if there's any
        /// progress or metrics to report, it is always better to report them.
        /// </summary>
        /// <param name="metricNames">The metrics that the calculation reports. These are completely independent, and there
        /// is no contract on whether the metric values should increase or not. As naming convention, <paramref name="metricNames"/>
        /// can have multiple words with spaces, and should be title-cased.</param>
        /// <param name="unitNames">The names of the progress units, listed from least granular to most granular.
        /// The idea is that the progress should be lexicographically increasing (like [0,0], [0,10], [1,0], [1,15], [2,5] etc.).
        /// As naming convention, <paramref name="unitNames"/> should be lower-cased and typically plural
        /// (e.g. iterations, clusters, examples). </param>
        public ProgressHeader(string[] metricNames, string[] unitNames)
        {
            Contracts.CheckValueOrNull(unitNames);
            Contracts.CheckValueOrNull(metricNames);

            UnitNames = unitNames ?? new string[0];
            MetricNames = metricNames ?? new string[0];
        }

        /// <summary>
        /// A constructor for no metrics, just progress units. As naming convention, <paramref name="unitNames"/> should be lower-cased
        /// and typically plural (e.g. iterations, clusters, examples).
        /// </summary>
        public ProgressHeader(params string[] unitNames)
            : this(null, unitNames)
        {
        }
    }

    /// <summary>
    /// A metric/progress holder item.
    /// </summary>
    public interface IProgressEntry
    {
        /// <summary>
        /// Set the progress value for the index <paramref name="index"/> to <paramref name="value"/>,
        /// and the limit value for the progress becomes 'unknown'.
        /// </summary>
        void SetProgress(int index, Double value);

        /// <summary>
        /// Set the progress value for the index <paramref name="index"/> to <paramref name="value"/>,
        /// and the limit value to <paramref name="lim"/>. If <paramref name="lim"/> is a NAN, it is set to null instead.
        /// </summary>
        void SetProgress(int index, Double value, Double lim);

        /// <summary>
        /// Sets the metric with index <paramref name="index"/> to <paramref name="value"/>.
        /// </summary>
        void SetMetric(int index, Double value);

    }
}