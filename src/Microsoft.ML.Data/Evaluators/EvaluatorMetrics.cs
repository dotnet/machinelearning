using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Data.Evaluators
{
    public class EvaluatorMetrics
    {
        /// <summary>
        /// Gets the area under the ROC curve.
        /// </summary>
        /// <remarks>
        /// The area under the ROC curve is equal to the probability that the classifier ranks
        /// a randomly chosen positive instance higher than a randomly chosen negative one
        /// (assuming 'positive' ranks higher than 'negative').
        /// </remarks>
        public double Auc { get; }

        /// <summary>
        /// Gets the accuracy of a classifier which is the proportion of correct predictions in the test set.
        /// </summary>
        public double Accuracy { get; }

        /// <summary>
        /// Gets the positive precision of a classifier which is the proportion of correctly predicted
        /// positive instances among all the positive predictions (i.e., the number of positive instances
        /// predicted as positive, divided by the total number of instances predicted as positive).
        /// </summary>
        public double PositivePrecision { get; }

        /// <summary>
        /// Gets the positive recall of a classifier which is the proportion of correctly predicted
        /// positive instances among all the positive instances (i.e., the number of positive instances
        /// predicted as positive, divided by the total number of positive instances).
        /// </summary>
        public double PositiveRecall { get; private set; }

        /// <summary>
        /// Gets the negative precision of a classifier which is the proportion of correctly predicted
        /// negative instances among all the negative predictions (i.e., the number of negative instances
        /// predicted as negative, divided by the total number of instances predicted as negative).
        /// </summary>
        public double NegativePrecision { get; }

        /// <summary>
        /// Gets the negative recall of a classifier which is the proportion of correctly predicted
        /// negative instances among all the negative instances (i.e., the number of negative instances
        /// predicted as negative, divided by the total number of negative instances).
        /// </summary>
        public double NegativeRecall { get; }

        /// <summary>
        /// Gets the F1 score of the classifier.
        /// </summary>
        /// <remarks>
        /// F1 score is the harmonic mean of precision and recall: 2 * precision * recall / (precision + recall).
        /// </remarks>
        public double F1Score { get; }

        /// <summary>
        /// Gets the area under the precision/recall curve of the classifier.
        /// </summary>
        /// <remarks>
        /// The area under the precision/recall curve is a single number summary of the information in the
        /// precision/recall curve. It is increasingly used in the machine learning community, particularly
        /// for imbalanced datasets where one class is observed more frequently than the other. On these
        /// datasets, AUPRC can highlight performance differences that are lost with AUC.
        /// </remarks>
        public double Auprc { get; }

        protected private static T Fetch<T>(IExceptionContext ectx, IRow row, string name)
        {
            if (!row.Schema.TryGetColumnIndex(name, out int col))
                throw ectx.Except($"Could not find column '{name}'");
            T val = default;
            row.GetGetter<T>(col)(ref val);
            return val;
        }

        internal EvaluatorMetrics(IExceptionContext ectx, IRow overallResult)
        {
            double Fetch(string name) => Fetch<double>(ectx, overallResult, name);
            Auc = Fetch(BinaryClassifierEvaluator.Auc);
            Accuracy = Fetch(BinaryClassifierEvaluator.Accuracy);
            PositivePrecision = Fetch(BinaryClassifierEvaluator.PosPrecName);
            PositiveRecall = Fetch(BinaryClassifierEvaluator.PosRecallName);
            NegativePrecision = Fetch(BinaryClassifierEvaluator.NegPrecName);
            NegativeRecall = Fetch(BinaryClassifierEvaluator.NegRecallName);
            F1Score = Fetch(BinaryClassifierEvaluator.F1);
            Auprc = Fetch(BinaryClassifierEvaluator.AuPrc);
        }
    }
}
