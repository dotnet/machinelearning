// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Evaluation results for multi-class classification trainers.
    /// </summary>
    public sealed class MulticlassClassificationMetrics
    {
        /// <summary>
        /// Gets the average log-loss of the classifier. Log-loss measures the performance of a classifier
        /// with respect to how much the predicted probabilities diverge from the true class label. Lower
        /// log-loss indicates a better model. A perfect model, which predicts a probability of 1 for the
        /// true class, will have a log-loss of 0.
        /// </summary>
        /// <remarks>
        /// <format type="text/markdown"><![CDATA[
        /// The log-loss metric is computed as follows:
        /// $LogLoss = - \frac{1}{m} \sum_{i = 1}^m log(p_i)$,
        /// where $m$ is the number of instances in the test set and
        /// $p_i$ is the probability returned by the classifier
        /// of the instance belonging to the true class.
        /// ]]>
        /// </format>
        /// </remarks>
        public double LogLoss { get; }

        /// <summary>
        /// Gets the log-loss reduction (also known as relative log-loss, or reduction in information gain - RIG)
        /// of the classifier. It gives a measure of how much a model improves on a model that gives random predictions.
        /// Log-loss reduction closer to 1 indicates a better model.
        /// </summary>
        /// <remarks>
        /// <format type="text/markdown"><![CDATA[
        /// The log-loss reduction is scaled relative to a classifier that predicts the prior for every example:
        /// $LogLossReduction = \frac{LogLoss(prior) - LogLoss(classifier)}{LogLoss(prior)}$
        /// This metric can be interpreted as the advantage of the classifier over a random prediction.
        /// For example, if the RIG equals 0.2, it can be interpreted as "the probability of a correct prediction is
        /// 20% better than random guessing".
        /// ]]>
        /// </format>
        /// </remarks>
        public double LogLossReduction { get; private set; }

        /// <summary>
        /// Gets the macro-average accuracy of the model.
        /// </summary>
        /// <remarks>
        /// The macro-average is the average accuracy at the class level. The accuracy for each class is computed
        /// and the macro-accuracy is the average of these accuracies.
        ///
        /// The macro-average metric gives the same weight to each class, no matter how many instances from
        /// that class the dataset contains.
        /// </remarks>
        public double MacroAccuracy { get; }

        /// <summary>
        /// Gets the micro-average accuracy of the model.
        /// </summary>
        /// <remarks>
        /// The micro-average is the fraction of instances predicted correctly across all classes. Micro-average can
        /// be a more useful metric than macro-average if class imbalance is suspected (i.e. one class has many more
        /// instances than the rest).
        /// </remarks>
        public double MicroAccuracy { get; }

        /// <summary>
        /// Convenience method for "TopKAccuracyForAllK[TopKPredictionCount - 1]". If <see cref="TopKPredictionCount"/> is positive,
        /// this is the relative number of examples where
        /// the true label is one of the top K predicted labels by the predictor.
        /// </summary>
        public double TopKAccuracy => TopKAccuracyForAllK?.LastOrDefault() ?? 0;

        /// <summary>
        /// If positive, this indicates the K in <see cref="TopKAccuracy"/> and <see cref="TopKAccuracyForAllK"/>.
        /// </summary>
        public int TopKPredictionCount { get; }

        /// <summary>
        /// Returns the top K accuracy for all K from 1 to the value of TopKPredictionCount.
        /// </summary>
        public IReadOnlyList<double> TopKAccuracyForAllK { get; }

        /// <summary>
        /// Gets the log-loss of the classifier for each class. Log-loss measures the performance of a classifier
        /// with respect to how much the predicted probabilities diverge from the true class label. Lower
        /// log-loss indicates a better model. A perfect model, which predicts a probability of 1 for the
        /// true class, will have a log-loss of 0.
        /// </summary>
        /// <remarks>
        /// The log-loss metric is computed as $-\frac{1}{m} \sum_{i=1}^m \log(p_i)$,
        /// where $m$ is the number of instances in the test set.
        /// $p_i$ is the probability returned by the classifier if the instance belongs to the class,
        /// and 1 minus the probability returned by the classifier if the instance does not belong to the class.
        /// </remarks>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[LogLoss](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/MulticlassClassification/LogLossPerClass.cs)]
        /// ]]></format>
        /// </example>
        public IReadOnlyList<double> PerClassLogLoss { get; }

        /// <summary>
        /// The <a href="https://en.wikipedia.org/wiki/Confusion_matrix">confusion matrix</a> giving the counts of the
        /// predicted classes versus the actual classes.
        /// </summary>
        public ConfusionMatrix ConfusionMatrix { get; }

        internal MulticlassClassificationMetrics(IHost host, DataViewRow overallResult, int topKPredictionCount, IDataView confusionMatrix)
        {
            double FetchDouble(string name) => RowCursorUtils.Fetch<double>(host, overallResult, name);
            MicroAccuracy = FetchDouble(MulticlassClassificationEvaluator.AccuracyMicro);
            MacroAccuracy = FetchDouble(MulticlassClassificationEvaluator.AccuracyMacro);
            LogLoss = FetchDouble(MulticlassClassificationEvaluator.LogLoss);
            LogLossReduction = FetchDouble(MulticlassClassificationEvaluator.LogLossReduction);
            TopKPredictionCount = topKPredictionCount;

            if (topKPredictionCount > 0)
                TopKAccuracyForAllK = RowCursorUtils.Fetch<VBuffer<double>>(host, overallResult, MulticlassClassificationEvaluator.AllTopKAccuracy).DenseValues().ToImmutableArray();

            var perClassLogLoss = RowCursorUtils.Fetch<VBuffer<double>>(host, overallResult, MulticlassClassificationEvaluator.PerClassLogLoss);
            PerClassLogLoss = perClassLogLoss.DenseValues().ToImmutableArray();
            ConfusionMatrix = MetricWriter.GetConfusionMatrix(host, confusionMatrix, binary: false, perClassLogLoss.Length);
        }

        internal MulticlassClassificationMetrics(double accuracyMicro, double accuracyMacro, double logLoss, double logLossReduction,
            int topKPredictionCount, double[] topKAccuracies, double[] perClassLogLoss)
        {
            MicroAccuracy = accuracyMicro;
            MacroAccuracy = accuracyMacro;
            LogLoss = logLoss;
            LogLossReduction = logLossReduction;
            TopKPredictionCount = topKPredictionCount;
            TopKAccuracyForAllK = topKAccuracies;
            PerClassLogLoss = perClassLogLoss.ToImmutableArray();
        }

        internal MulticlassClassificationMetrics(double accuracyMicro, double accuracyMacro, double logLoss, double logLossReduction,
            int topKPredictionCount, double[] topKAccuracies, double[] perClassLogLoss, ConfusionMatrix confusionMatrix)
            : this(accuracyMicro, accuracyMacro, logLoss, logLossReduction, topKPredictionCount, topKAccuracies, perClassLogLoss)
        {
            ConfusionMatrix = confusionMatrix;
        }
    }
}
