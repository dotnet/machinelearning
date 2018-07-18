// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.EntryPoints.JsonUtils;
using System;

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

        public static SupportedMetric GetSupportedMetric(Metrics metric)
        {
            SupportedMetric supportedMetric = null;
            switch(metric)
            {
                case Metrics.Auc:
                    supportedMetric = new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.Auc, true);
                    break;
                case Metrics.AccuracyMicro:
                    supportedMetric = new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.AccuracyMicro, true);
                    break;
                case Metrics.AccuracyMacro:
                    supportedMetric = new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.AccuracyMacro, true);
                    break;
                case Metrics.L1:
                    supportedMetric = new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.L1, false);
                    break;
                case Metrics.L2:
                    supportedMetric = new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.L2, false);
                    break;
                case Metrics.F1:
                    supportedMetric = new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.F1, true);
                    break;
                case Metrics.AuPrc:
                    supportedMetric = new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.AuPrc, true);
                    break;
                case Metrics.TopKAccuracy:
                    supportedMetric = new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.TopKAccuracy, true);
                    break;
                case Metrics.Rms:
                    supportedMetric = new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.Rms, false);
                    break;
                case Metrics.LossFn:
                    supportedMetric = new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.LossFn, false);
                    break;
                case Metrics.RSquared:
                    supportedMetric = new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.RSquared, false);
                    break;
                case Metrics.LogLoss:
                    supportedMetric = new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.LogLoss, false);
                    break;
                case Metrics.LogLossReduction:
                    supportedMetric = new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.LogLossReduction, true);
                    break;
                case Metrics.Ndcg:
                    supportedMetric = new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.Ndcg, true);
                    break;
                case Metrics.Dcg:
                    supportedMetric = new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.Dcg, true);
                    break;
                case Metrics.PositivePrecision:
                    supportedMetric = new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.PositivePrecision, true);
                    break;
                case Metrics.PositiveRecall:
                    supportedMetric = new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.PositiveRecall, true);
                    break;
                case Metrics.NegativePrecision:
                    supportedMetric = new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.NegativePrecision, true);
                    break;
                case Metrics.NegativeRecall:
                    supportedMetric = new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.NegativeRecall, true);
                    break;
                case Metrics.DrAtK:
                    supportedMetric = new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.DrAtK, true);
                    break;
                case Metrics.DrAtPFpr:
                    supportedMetric = new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.DrAtPFpr, true);
                    break;
                case Metrics.DrAtNumPos:
                    supportedMetric = new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.DrAtNumPos, true);
                    break;
                case Metrics.NumAnomalies:
                    supportedMetric = new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.NumAnomalies, true);
                    break;
                case Metrics.ThreshAtK:
                    supportedMetric = new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.ThreshAtK, false);
                    break;
                case Metrics.ThreshAtP:
                    supportedMetric = new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.ThreshAtP, false);
                    break;
                case Metrics.ThreshAtNumPos:
                    supportedMetric = new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.ThreshAtNumPos, false);
                    break;
                case Metrics.Nmi:
                    supportedMetric = new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.Nmi, true);
                    break;
                case Metrics.AvgMinScore:
                    supportedMetric = new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.AvgMinScore, false);
                    break;
                case Metrics.Dbi:
                    supportedMetric = new SupportedMetric(FieldNames.PipelineSweeperSupportedMetrics.Dbi, false);
                    break;
                default:
                    throw new NotSupportedException($"Metric '{metric}' not supported.");
            }
            return supportedMetric;
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
