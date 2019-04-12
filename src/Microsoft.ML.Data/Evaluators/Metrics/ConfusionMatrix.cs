using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Represents the <a href="https://en.wikipedia.org/wiki/Confusion_matrix">confusion matrix</a> of the classification results.
    /// </summary>
    public sealed class ConfusionMatrix
    {
        /// <summary>
        /// The calculated value of <a href="https://en.wikipedia.org/wiki/Precision_and_recall#Precision">precision</a> for each class.
        /// </summary>
        public ImmutableArray<double> PerClassPrecision { get; }

        /// <summary>
        /// The calculated value of <a href="https://en.wikipedia.org/wiki/Precision_and_recall#Recall">recall</a> for each class.
        /// </summary>
        public ImmutableArray<double> PerClassRecall { get; }

        /// <summary>
        /// The confsion matrix counts for the combinations actual class/predicted class.
        /// The actual classes are in the rows of the table, and the predicted classes in the columns.
        /// </summary>
        public ImmutableArray<ImmutableArray<double>> ConfusionTableCounts { get; }

        /// <summary>
        /// The indicators of the predicted classes.
        /// It might be the classes names, or just indices of the predicted classes, if the name mapping is missing.
        /// </summary>
        public IReadOnlyList<ReadOnlyMemory<char>> PredictedClassesIndicators;

        internal readonly bool Sampled;
        internal readonly bool Binary;

        private readonly IHost _host;

        internal ConfusionMatrix(IHost host, double[] precision, double[] recall, double[][] confusionTableCounts,
            List<ReadOnlyMemory<char>> labelNames, bool sampled, bool binary, DataViewSchema.Annotations classIndicators)
        {
            _host = host;

            PerClassPrecision = precision.ToImmutableArray();
            PerClassRecall = recall.ToImmutableArray();
            Sampled = sampled;
            Binary = binary;
            PredictedClassesIndicators = labelNames.AsReadOnly();

            var classNumber = confusionTableCounts.Length;
            List<ImmutableArray<double>> counts = new List<ImmutableArray<double>>(classNumber);

            for (int i = 0; i < classNumber; i++)
                counts.Add(ImmutableArray.Create(confusionTableCounts[i]));

            ConfusionTableCounts = counts.ToImmutableArray();

        }

        /// <summary>
        /// Returns a human readable representation of the confusion table.
        /// </summary>
        /// <returns></returns>
        public string GetFormattedConfusionTable() => MetricWriter.GetConfusionTableAsString(this, false);

        /// <summary>
        /// Gets the confusion table count for the pair <paramref name="predictedClassIndicatorIndex"/>/<paramref name="actualClassIndicatorIndex"/>.
        /// </summary>
        /// <param name="predictedClassIndicatorIndex">The index of the predicted label indicator, in the <see cref="PredictedClassesIndicators"/>.</param>
        /// <param name="actualClassIndicatorIndex">The index of the actual label indicator, in the <see cref="PredictedClassesIndicators"/>.</param>
        /// <returns></returns>
        public double GetCountForClassPair(uint predictedClassIndicatorIndex, uint actualClassIndicatorIndex)
        {
            _host.Assert(predictedClassIndicatorIndex < ConfusionTableCounts.Length && actualClassIndicatorIndex < ConfusionTableCounts.Length);
            return ConfusionTableCounts[(int)actualClassIndicatorIndex][(int)predictedClassIndicatorIndex];
        }
    }
}
