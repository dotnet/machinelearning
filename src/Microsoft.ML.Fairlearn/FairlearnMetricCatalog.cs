// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.Data.Analysis;
using Microsoft.ML.Data;

namespace Microsoft.ML.Fairlearn
{
    public class FairlearnMetricCatalog
    {
        private readonly MLContext _context;
        public FairlearnMetricCatalog(MLContext context)
        {
            _context = context;
        }

        #region binary classification
        public BinaryGroupMetric BinaryClassificationMetrics(IDataView eval, string labelColumn, string predictedColumn, string sensitiveFeatureColumn)
        {
            return new BinaryGroupMetric(eval, labelColumn, predictedColumn, sensitiveFeatureColumn);
        }
        #endregion
    }

    public class BinaryGroupMetric : IGroupMetric
    {
        private readonly IDataView _eval;
        private readonly string _labelColumn;
        private readonly string _predictedColumn;
        private readonly string _sensitiveFeatureColumn;

        public BinaryGroupMetric(IDataView eval, string labelColumn, string predictedColumn, string sensitiveFeatureColumn)
        {
            _eval = eval;
            _labelColumn = labelColumn;
            _predictedColumn = predictedColumn;
            _sensitiveFeatureColumn = sensitiveFeatureColumn;
        }
        private readonly MLContext _context;

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        /// <exception cref="NotImplementedException"></exception>
        public DataFrame ByGroup()
        {
            // 1. group row according to sensitive feature column
            // 2. split dataset to different groups, data_g1, data_g2.....
            // 3. calculate binary metrics for different groups
            // 4. create datafrome from result of step 3
            // 5. return it.


            DataFrame result = new DataFrame();
            return result;
        }

        public Dictionary<string, double> DifferenceBetweenGroups()
        {
            throw new NotImplementedException();
        }

        public Dictionary<string, double> Overall()
        {
            CalibratedBinaryClassificationMetrics metrics = _context.BinaryClassification.Evaluate(_eval, _labelColumn);

            // create the dictionary to hold the results
            Dictionary<string, double> metricsDict = new Dictionary<string, double>();
            metricsDict.Add("AUC", metrics.AreaUnderRocCurve);
            metricsDict.Add("Accuracy", metrics.Accuracy);
            metricsDict.Add("PosPrec", metrics.PositivePrecision);
            metricsDict.Add("PosRecall", metrics.PositiveRecall);
            metricsDict.Add("NegPrec", metrics.NegativePrecision);
            metricsDict.Add("NegRecall", metrics.NegativeRecall);
            metricsDict.Add("F1Score", metrics.F1Score);
            metricsDict.Add("AreaUnderPrecisionRecallCurve", metrics.AreaUnderPrecisionRecallCurve);

            metricsDict.Add("LogLoss", metrics.LogLoss);
            metricsDict.Add("LogLossReduction", metrics.LogLossReduction);
            metricsDict.Add("Entropy", metrics.Entropy);
            return metricsDict;
        }
    }
}
