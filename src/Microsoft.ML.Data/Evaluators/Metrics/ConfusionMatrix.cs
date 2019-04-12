// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

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
        public int NumberOfPredictedClasses { get; }

        /// <summary>
        /// The <see cref="DataViewSchema.Annotations"/> associated with the Confusion Matrix counts.
        /// It contains information about the predicted classes, if that information is available.
        /// It can be the classes names, or their indices.
        /// </summary>
        public DataViewSchema.Annotations ClassIndicators { get; }

        /// <summary>
        /// The indicators of the predicted classes.
        /// It might be the classes names, or just indices of the predicted classes, if the name mapping is missing.
        /// </summary>
        internal IReadOnlyList<ReadOnlyMemory<char>> PredictedClassesIndicators;

        internal readonly bool Sampled;
        internal readonly bool Binary;

        private readonly IHost _host;

        /// <summary>
        /// The confusion matrix as a structured type, built from the counts of the confusion table idv.
        /// </summary>
        /// <param name="host">The IHost instance. </param>
        /// <param name="precision">The values of precision per class.</param>
        /// <param name="recall">The vales of recall per class.</param>
        /// <param name="confusionTableCounts">The counts of the confusion table. The actual classes values are in the rows of the 2D array,
        /// and the counts of the predicted classes are in the columns.</param>
        /// <param name="labelNames">The predicted classes names, or the indexes of the classes, if the names are missing.</param>
        /// <param name="sampled">Whether the classes are sampled.</param>
        /// <param name="binary">Whether the confusion table is the result of a binary classification. </param>
        /// <param name="classIndicators">The Annotations of the Count column, in the confusionTable idv.</param>
        internal ConfusionMatrix(IHost host, double[] precision, double[] recall, double[][] confusionTableCounts,
            List<ReadOnlyMemory<char>> labelNames, bool sampled, bool binary, DataViewSchema.Annotations classIndicators)
        {
            _host = host;

            _host.AssertNonEmpty(precision);
            _host.AssertNonEmpty(recall);
            _host.AssertNonEmpty(confusionTableCounts);
            _host.AssertNonEmpty(labelNames);
            _host.AssertNonEmpty(precision);

            _host.Assert(precision.Length == confusionTableCounts.Length);
            _host.Assert(recall.Length == confusionTableCounts.Length);
            _host.Assert(labelNames.Count == confusionTableCounts.Length);

            PerClassPrecision = precision.ToImmutableArray();
            PerClassRecall = recall.ToImmutableArray();
            Sampled = sampled;
            Binary = binary;
            PredictedClassesIndicators = labelNames.AsReadOnly();

            NumberOfPredictedClasses = confusionTableCounts.Length;
            List<ImmutableArray<double>> counts = new List<ImmutableArray<double>>(NumberOfPredictedClasses);

            for (int i = 0; i < NumberOfPredictedClasses; i++)
                counts.Add(ImmutableArray.Create(confusionTableCounts[i]));

            ConfusionTableCounts = counts.ToImmutableArray();
            ClassIndicators = classIndicators;
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
        public double GetCountForClassPair(int predictedClassIndicatorIndex, int actualClassIndicatorIndex)
        {
            _host.CheckParam(predictedClassIndicatorIndex > -1 && predictedClassIndicatorIndex < ConfusionTableCounts.Length,
                nameof(predictedClassIndicatorIndex), "Invalid index. Should be non-negative, less than the number of classes.");
            _host.CheckParam(actualClassIndicatorIndex > -1 && actualClassIndicatorIndex < ConfusionTableCounts.Length,
                nameof(actualClassIndicatorIndex), "Invalid index. Should be non-negative, less than the number of classes.");

            return ConfusionTableCounts[actualClassIndicatorIndex][predictedClassIndicatorIndex];
        }
    }
}
