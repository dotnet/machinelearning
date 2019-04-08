using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    public sealed class ConfusionMatrix
    {
        public double[] PrecisionSums { get; }
        public double[] RecallSums { get; }
        public ReadOnlyMemory<char>[] Labels => PredictedLabelNames.ToArray(); //sefilipi: return as string[]

        public double[][] ConfusionTableCounts { get; }

        internal readonly List<ReadOnlyMemory<char>> PredictedLabelNames;
        internal readonly bool Sampled;
        internal readonly bool Binary;

        private readonly IHost _host;

        internal ConfusionMatrix(IHost host, double[] precisionSums, double[] recallSums, double[][] confusionTableCounts,
            List<ReadOnlyMemory<char>> labelNames, bool sampled, bool binary)
        {
            _host = host;

            PrecisionSums = precisionSums;
            RecallSums = recallSums;
            ConfusionTableCounts = confusionTableCounts;
            PredictedLabelNames = labelNames;
            Sampled = sampled;
            Binary = binary;
        }

        public override string ToString() => MetricWriter.GetConfusionTableAsString(this, false);

        public double GetCountForClassPair(string predictedLabel, string actualLabel)
        {
            int predictedLabelIndex = PredictedLabelNames.IndexOf(predictedLabel.AsMemory());
            int actualLabelIndex = PredictedLabelNames.IndexOf(actualLabel.AsMemory());

            _host.CheckParam(predictedLabelIndex > -1, nameof(predictedLabel), "Unknown given PredictedLabel.");
            _host.CheckParam(actualLabelIndex > -1, nameof(actualLabel), "Unknown given ActualLabel.");

            _host.Assert(predictedLabelIndex < ConfusionTableCounts.Length && actualLabelIndex < ConfusionTableCounts.Length);

            return ConfusionTableCounts[actualLabelIndex][predictedLabelIndex];
        }
    }
}
