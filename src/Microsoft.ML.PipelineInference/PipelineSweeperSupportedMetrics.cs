// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.PipelineInference;
using Microsoft.ML.Runtime.EntryPoints.JsonUtils;
using Newtonsoft.Json.Linq;

namespace Microsoft.ML.Runtime.PipelineInference
{
    /// <summary>
    /// PipelineSweeper will support metrics as they are added here.
    /// </summary>
    public sealed class PipelineSweeperSupportedMetrics
    {
        public enum Metrics
        {
            Auc,
            AccuracyMicro,
            AccuracyMacro,
            L1,
            L2,
            F1,
            AuPrc,
            TopKAccuracy,
            Rms,
            LossFn,
            RSquared,
            LogLoss,
            LogLossReduction,
            Ndcg,
            Dcg,
            PositivePrecision,
            PositiveRecall,
            NegativePrecision,
            NegativeRecall,
            DrAtK,
            DrAtPFpr,
            DrAtNumPos,
            NumAnomalies,
            ThreshAtK,
            ThreshAtP,
            ThreshAtNumPos,
            Nmi,
            AvgMinScore,
            Dbi
        };

        /// <summary>
        /// Map Enum Metrics to a SupportedMetric
        /// </summary>
        private static readonly Dictionary<string, SupportedMetric> _map = new Dictionary<string, SupportedMetric>
        {
            { Metrics.Auc.ToString(), new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.Auc, true)},
            { Metrics.AccuracyMicro.ToString(), new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.AccuracyMicro, true)},
            { Metrics.AccuracyMacro.ToString(),  new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.AccuracyMacro, true)},
            { Metrics.L1.ToString(), new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.L1, false)},
            { Metrics.L2.ToString(), new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.L2, false)},
            { Metrics.F1.ToString(), new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.F1, true)},
            { Metrics.AuPrc.ToString(), new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.AuPrc, true)},
            { Metrics.TopKAccuracy.ToString(), new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.TopKAccuracy, true)},
            { Metrics.Rms.ToString(), new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.Rms, false)},
            { Metrics.LossFn.ToString(), new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.LossFn, false)},
            { Metrics.RSquared.ToString(), new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.RSquared, false)},
            { Metrics.LogLoss.ToString(), new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.LogLoss, false)},
            { Metrics.LogLossReduction.ToString(), new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.LogLossReduction, true)},
            { Metrics.Ndcg.ToString(), new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.Ndcg, true)},
            { Metrics.Dcg.ToString(), new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.Dcg, true)},
            { Metrics.PositivePrecision.ToString(), new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.PositivePrecision, true)},
            { Metrics.PositiveRecall.ToString(), new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.PositiveRecall, true)},
            { Metrics.NegativePrecision.ToString(), new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.NegativePrecision, true)},
            { Metrics.NegativeRecall.ToString(), new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.NegativeRecall, true)},
            { Metrics.DrAtK.ToString(), new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.DrAtK, true)},
            { Metrics.DrAtPFpr.ToString(), new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.DrAtPFpr, true)},
            { Metrics.DrAtNumPos.ToString(), new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.DrAtNumPos, true)},
            { Metrics.NumAnomalies.ToString(), new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.NumAnomalies, true)},
            { Metrics.ThreshAtK.ToString(), new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.ThreshAtK, false)},
            { Metrics.ThreshAtP.ToString(), new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.ThreshAtP, false)},
            { Metrics.ThreshAtNumPos.ToString(), new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.ThreshAtNumPos, false)},
            { Metrics.Nmi.ToString(), new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.Nmi, false)},
            { Metrics.AvgMinScore.ToString(), new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.AvgMinScore, false)},
            { Metrics.Dbi.ToString(), new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.Dbi, false)}
        };

        public static SupportedMetric GetSupportedMetric(IHostEnvironment env, string metricName)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckNonEmpty(metricName, nameof(metricName));

            if (_map.ContainsKey(metricName))
            {
                return _map[metricName];
            }

            throw new NotSupportedException($"Metric '{metricName}' not supported.");
        }
    }

    public sealed class SupportedMetric
    {
        public string Name { get; }
        public bool IsMaximizing { get; }

        public SupportedMetric(string name, bool isMaximizing)
        {
            Name = name;
            IsMaximizing = isMaximizing;
        }

        public override string ToString() => Name;
    }
}
