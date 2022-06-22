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
        public BinaryGroupMetric BinaryClassification(IDataView eval, string labelColumn, string predictedColumn, string sensitiveFeatureColumn)
        {
            return new BinaryGroupMetric(eval, labelColumn, predictedColumn, sensitiveFeatureColumn);
        }
        #endregion

        #region regression
        public RegressionMetric Regression(IDataView eval, string labelColumn, string scoreColumn, string sensitiveFeatureColumn)
        {
            return new RegressionMetric(eval, labelColumn, scoreColumn, sensitiveFeatureColumn);
        }
        #endregion
    }

    public class BinaryGroupMetric : IGroupMetric
    {
        private static readonly string[] _looseBooleanFalseValue = new[] { "0", "false", "f" };

        private readonly IDataView _eval;
        private readonly string _labelColumn;
        private readonly string _predictedColumn;
        private readonly string _sensitiveFeatureColumn;
        private readonly MLContext _context = new MLContext();

        public BinaryGroupMetric(IDataView eval, string labelColumn, string predictedColumn, string sensitiveFeatureColumn)
        {
            _eval = eval;
            _labelColumn = labelColumn;
            _predictedColumn = predictedColumn;
            _sensitiveFeatureColumn = sensitiveFeatureColumn;
        }

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

            // TODO: is converting IDataview to DataFrame the best practice?
            // .ToDataFram pulls the data into memory.

            //Brainstorm:  1. save it to a text file, temp file. figure unique columns. do a filter on those columns
            // 2. filtering (maybe not the best approach) dataview
            // 3. custom mapping 
            var evalDf = _eval.ToDataFrame();
            var groups = evalDf.Rows.GroupBy(r => r[sensitiveCol.Index]);
            var groupMetric = new Dictionary<object, CalibratedBinaryClassificationMetrics>();
            foreach (var kv in groups)
            {
                var data = new DataFrame(_eval.Schema.AsEnumerable().Select<DataViewSchema.Column, DataFrameColumn>(column =>
                {
                    if (column.Type is TextDataViewType)
                    {
                        var columns = new StringDataFrameColumn(column.Name);
                        return columns;
                    }
                    else if (column.Type.RawType == typeof(bool))
                    {
                        var primitiveColumn = new BooleanDataFrameColumn(column.Name);

                        return primitiveColumn;
                    }
                    else if (column.Type.RawType == typeof(int))
                    {
                        var primitiveColumn = new Int32DataFrameColumn(column.Name);

                        return primitiveColumn;
                    }
                    else if (column.Type.RawType == typeof(float))
                    {
                        var primitiveColumn = new SingleDataFrameColumn(column.Name);

                        return primitiveColumn;
                    }
                    else if (column.Type.RawType == typeof(DateTime))
                    {
                        // BLOCKED by DataFrame bug https://github.com/dotnet/machinelearning/issues/6213
                        // Evaluate as a string for now 
                        var columns = new StringDataFrameColumn(column.Name, 0);
                        return columns;
                    }
                    else
                    {
                        throw new NotImplementedException();
                    }
                }).Where(x => x != null));
                // create the column
                data.Append(kv, inPlace: true);
                CalibratedBinaryClassificationMetrics metrics = _context.BinaryClassification.Evaluate(data, _labelColumn); // how does this work?
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
    public class RegressionMetric : IGroupMetric
    {
        private readonly IDataView _eval;
        private readonly string _labelColumn;
        private readonly string _scoreColumn;
        private readonly string _sensitiveFeatureColumn;
        private readonly MLContext _context = new MLContext();

        public RegressionMetric(IDataView eval, string labelColumn, string scoreColumn, string sensitiveFeatureColumn)
        {
            _eval = eval;
            _labelColumn = labelColumn;
            _scoreColumn = scoreColumn;
            _sensitiveFeatureColumn = sensitiveFeatureColumn;
        }

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

            // TODO: is converting IDataview to DataFrame the best practice?
            // .ToDataFram pulls the data into memory.

            //Brainstorm:  1. save it to a text file, temp file. figure unique columns. do a filter on those columns
            // 2. filtering (maybe not the best approach) dataview
            // 3. custom mapping 
            var evalDf = _eval.ToDataFrame();
            var groups = evalDf.Rows.GroupBy(r => r[sensitiveCol.Index]);
            var groupMetric = new Dictionary<object, RegressionMetrics>();
            foreach (var kv in groups)
            {
                var data = new DataFrame(_eval.Schema.AsEnumerable().Select<DataViewSchema.Column, DataFrameColumn>(column =>
                {
                    if (column.Type is TextDataViewType)
                    {
                        var columns = new StringDataFrameColumn(column.Name);
                        return columns;
                    }
                    else if (column.Type.RawType == typeof(bool))
                    {
                        var primitiveColumn = new BooleanDataFrameColumn(column.Name);

                        return primitiveColumn;
                    }
                    else if (column.Type.RawType == typeof(int))
                    {
                        var primitiveColumn = new Int32DataFrameColumn(column.Name);

                        return primitiveColumn;
                    }
                    else if (column.Type.RawType == typeof(float))
                    {
                        var primitiveColumn = new SingleDataFrameColumn(column.Name);

                        return primitiveColumn;
                    }
                    else if (column.Type.RawType == typeof(DateTime))
                    {
                        // BLOCKED by DataFrame bug https://github.com/dotnet/machinelearning/issues/6213
                        // Evaluate as a string for now 
                        var columns = new StringDataFrameColumn(column.Name, 0);
                        return columns;
                    }
                    else
                    {
                        throw new NotImplementedException();
                    }
                }).Where(x => x != null));
                // create the column
                data.Append(kv, inPlace: true);
                RegressionMetrics metrics = _context.Regression.Evaluate(data, _labelColumn, _scoreColumn);
                groupMetric[kv.Key] = metrics;
            }

            DataFrame result = new DataFrame();
            result[_sensitiveFeatureColumn] = DataFrameColumn.Create(_sensitiveFeatureColumn, groupMetric.Keys.Select(x => x.ToString()));
            result["RSquared"] = DataFrameColumn.Create("RSquared", groupMetric.Keys.Select(k => groupMetric[k].RSquared));
            result["RMS"] = DataFrameColumn.Create("RMS", groupMetric.Keys.Select(k => groupMetric[k].RootMeanSquaredError));

            return result;
        }



        public Dictionary<string, double> DifferenceBetweenGroups()
        {
            throw new NotImplementedException();
        }

        public Dictionary<string, double> Overall()
        {
            RegressionMetrics metrics = _context.Regression.Evaluate(_eval, _labelColumn);

            // create the dictionary to hold the results
            Dictionary<string, double> metricsDict = new Dictionary<string, double>();
            metricsDict.Add("RSquared", metrics.RSquared);
            metricsDict.Add("RMS", metrics.RootMeanSquaredError);
            return metricsDict;
        }

    }
}
