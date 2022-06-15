// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
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
            var sensitiveCol = _eval.Schema[_sensitiveFeatureColumn];
            // get all the columns of the schema
            DataViewSchema columns = _eval.Schema;

            var evalDf = _eval.ToDataFrame();
            var groups = evalDf.Rows.GroupBy(r => r[sensitiveCol.Index]);
            var groupMetric = new Dictionary<object, CalibratedBinaryClassificationMetrics>();
            foreach (var kv in groups)
            {
                var data = new DataFrame();
                data.Append(kv);
                CalibratedBinaryClassificationMetrics metrics = _context.BinaryClassification.Evaluate(data, _labelColumn);
                groupMetric[kv.Key] = metrics;
            }

            DataFrame result = new DataFrame();
            result[_sensitiveFeatureColumn] = DataFrameColumn.Create(_sensitiveFeatureColumn, groupMetric.Keys.Select(x => x.ToString()));
            result["AUC"] = DataFrameColumn.Create("AUC", groupMetric.Keys.Select(k => groupMetric[k].Accuracy)); //coloumn name?

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
            // following metrics are from the extensions
            metricsDict.Add("LogLoss", metrics.LogLoss);
            metricsDict.Add("LogLossReduction", metrics.LogLossReduction);
            metricsDict.Add("Entropy", metrics.Entropy);
            return metricsDict;
        }
    }
}
