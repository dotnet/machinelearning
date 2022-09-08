// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// instance for monitor, which is used by <see cref="AutoMLExperiment"/> to report training progress.
    /// </summary>
    public interface IMonitor
    {
        void ReportCompletedTrial(TrialResult result);

        void ReportBestTrial(TrialResult result);

        void ReportFailTrial(TrialSettings settings, Exception exception = null);

        void ReportRunningTrial(TrialSettings setting);
    }

    /// <summary>
    /// monitor that redirect output results to context.Log channel.
    /// </summary>
    internal class MLContextMonitor : IMonitor
    {
        private readonly IChannel _logger;
        private readonly List<TrialResult> _completedTrials;
        private readonly SweepablePipeline _pipeline;
        public MLContextMonitor(IChannel logger, SweepablePipeline pipeline)
        {
            _logger = logger;
            _completedTrials = new List<TrialResult>();
            _pipeline = pipeline;
        }

        public virtual void ReportBestTrial(TrialResult result)
        {
            _logger.Info($"Update Best Trial - Id: {result.TrialSettings.TrialId} - Metric: {result.Metric} - Pipeline: {_pipeline.ToString(result.TrialSettings.Parameter)}");
        }

        public virtual void ReportCompletedTrial(TrialResult result)
        {
            _logger.Info($"Update Completed Trial - Id: {result.TrialSettings.TrialId} - Metric: {result.Metric} - Pipeline: {_pipeline.ToString(result.TrialSettings.Parameter)} - Duration: {result.DurationInMilliseconds} - Peak CPU: {result.PeakCpu?.ToString("p")} - Peak Memory in MB: {result.PeakMemoryInMegaByte?.ToString("F")}");
            _completedTrials.Add(result);
        }

        public virtual void ReportFailTrial(TrialSettings settings, Exception exception = null)
        {
            _logger.Info($"Update Failed Trial - Id: {settings.TrialId} - Pipeline: {_pipeline.ToString(settings.Parameter)}");
        }

        public virtual void ReportRunningTrial(TrialSettings setting)
        {
            _logger.Info($"Update Running Trial - Id: {setting.TrialId} - Pipeline: {_pipeline.ToString(setting.Parameter)}");
        }
    }

    internal class TrialResultMonitor<TMetrics> : MLContextMonitor
        where TMetrics : class
    {
        public TrialResultMonitor(IChannel channel, SweepablePipeline pipeline)
            : base(channel, pipeline)
        {
            this.RunDetails = new List<TrialResult<TMetrics>>();
        }

        public event EventHandler<TrialResult<TMetrics>> OnTrialCompleted;

        public List<TrialResult<TMetrics>> RunDetails { get; }

        public TrialResult<TMetrics> BestRun { get; private set; }

        public override void ReportBestTrial(TrialResult result)
        {
            base.ReportBestTrial(result);
            if (result is TrialResult<TMetrics> binaryClassificationResult)
            {
                BestRun = binaryClassificationResult;
            }
            else
            {
                throw new ArgumentException($"result must be of type {typeof(TrialResult<TMetrics>)}");
            }
        }

        public override void ReportCompletedTrial(TrialResult result)
        {
            base.ReportCompletedTrial(result);
            if (result is TrialResult<TMetrics> metricResult)
            {
                RunDetails.Add(metricResult);
                OnTrialCompleted?.Invoke(this, metricResult);
            }
            else
            {
                throw new ArgumentException($"result must be of type {typeof(TrialResult<TMetrics>)}");
            }
        }

        public override void ReportFailTrial(TrialSettings settings, Exception exp)
        {
            base.ReportFailTrial(settings, exp);

            var result = new TrialResult<TMetrics>
            {
                TrialSettings = settings,
                Exception = exp,
            };

            RunDetails.Add(result);
        }
    }
}
