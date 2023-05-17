// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Data.Analysis;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

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
        public BinaryGroupMetric BinaryClassification(IDataView eval, string labelColumn, string predictedColumn, string sensitiveFeatureColumn, string scoreColumn = "Score")
        {
            return new BinaryGroupMetric(_context, eval, labelColumn, predictedColumn, sensitiveFeatureColumn, scoreColumn);
        }
        #endregion

        #region regression
        public RegressionGroupMetric Regression(IDataView eval, string labelColumn, string scoreColumn, string sensitiveFeatureColumn)
        {
            return new RegressionGroupMetric(eval, labelColumn, scoreColumn, sensitiveFeatureColumn);
        }
        #endregion
    }

    public class BinaryGroupMetric : IGroupMetric
    {
        private readonly IDataView _eval;
        private readonly string _labelColumn;
        private readonly string _predictedColumn;
        private readonly string _scoreColumn;
        private readonly string _sensitiveFeatureColumn;
        private readonly MLContext _context;

        public BinaryGroupMetric(MLContext context, IDataView eval, string labelColumn, string predictedColumn, string sensitiveFeatureColumn, string scoreColumn)
        {
            _context = context;
            _eval = eval;
            _labelColumn = labelColumn;
            _predictedColumn = predictedColumn;
            _sensitiveFeatureColumn = sensitiveFeatureColumn;
            _scoreColumn = scoreColumn;
        }

        public IEnumerable<string> GroupIds
        {
            get
            {
                var sensitiveCol = _eval.Schema[_sensitiveFeatureColumn];
                if (sensitiveCol.Type == TextDataViewType.Instance)
                {
                    return _eval.GetColumn<string>(sensitiveCol.Name);
                }
                else
                {
                    var convertToString = _context.Transforms.Conversion.ConvertType(sensitiveCol.Name, sensitiveCol.Name, DataKind.String);
                    var data = convertToString.Fit(_eval).Transform(_eval);

                    return data.GetColumn<string>(sensitiveCol.Name);
                }
            }
        }

        public DataFrame ByGroup()
        {
            var truths = _eval.GetColumn<bool>(_labelColumn).ToArray();
            var predicted = _eval.GetColumn<bool>(_predictedColumn).ToArray();
            var scores = _eval.GetColumn<float>(_scoreColumn).ToArray();
            Contracts.Assert(truths.Count() == predicted.Count());
            Contracts.Assert(truths.Count() == scores.Count());
            Contracts.Assert(GroupIds.Count() == truths.Count());

            var res = GroupIds.Select((id, i) =>
            {
                return (id, new ModelInput
                {
                    Label = truths[i],
                    PredictedLabel = predicted[i],
                    Score = scores[i],
                });
            }).GroupBy(kv => kv.id)
            .ToDictionary(group => group.Key, group => _context.Data.LoadFromEnumerable(group.Select(g => g.Item2)));

            var groupMetric = res.Select(kv => (kv.Key, _context.BinaryClassification.EvaluateNonCalibrated(kv.Value)))
                                .ToDictionary(kv => kv.Key, kv => kv.Item2);

            DataFrame result = new DataFrame();
            result[_sensitiveFeatureColumn] = DataFrameColumn.Create(_sensitiveFeatureColumn, groupMetric.Keys.Select(x => x.ToString()));
            result["AUC"] = DataFrameColumn.Create("AUC", groupMetric.Keys.Select(k => groupMetric[k].AreaUnderRocCurve)); //coloumn name?
            result["Accuracy"] = DataFrameColumn.Create("Accuracy", groupMetric.Keys.Select(k => groupMetric[k].Accuracy));
            result["PosPrec"] = DataFrameColumn.Create("PosPrec", groupMetric.Keys.Select(k => groupMetric[k].PositivePrecision));
            result["PosRecall"] = DataFrameColumn.Create("PosRecall", groupMetric.Keys.Select(k => groupMetric[k].PositiveRecall));
            result["NegPrec"] = DataFrameColumn.Create("NegPrec", groupMetric.Keys.Select(k => groupMetric[k].NegativePrecision));
            result["NegRecall"] = DataFrameColumn.Create("NegRecall", groupMetric.Keys.Select(k => groupMetric[k].NegativeRecall));
            result["F1Score"] = DataFrameColumn.Create("F1Score", groupMetric.Keys.Select(k => groupMetric[k].F1Score));
            result["AreaUnderPrecisionRecallCurve"] = DataFrameColumn.Create("AreaUnderPrecisionRecallCurve", groupMetric.Keys.Select(k => groupMetric[k].AreaUnderPrecisionRecallCurve));

            return result;
        }

        public Dictionary<string, double> Overall()
        {
            CalibratedBinaryClassificationMetrics metrics = _context.BinaryClassification.Evaluate(_eval, _labelColumn);

            // create the dictionary to hold the results
            Dictionary<string, double> metricsDict = new Dictionary<string, double>
            {
                { "AUC", metrics.AreaUnderRocCurve },
                { "Accuracy", metrics.Accuracy },
                { "PosPrec", metrics.PositivePrecision },
                { "PosRecall", metrics.PositiveRecall },
                { "NegPrec", metrics.NegativePrecision },
                { "NegRecall", metrics.NegativeRecall },
                { "F1Score", metrics.F1Score },
                { "AreaUnderPrecisionRecallCurve", metrics.AreaUnderPrecisionRecallCurve },
                // following metrics are from the extensions
                { "LogLoss", metrics.LogLoss },
                { "LogLossReduction", metrics.LogLossReduction },
                { "Entropy", metrics.Entropy }
            };

            return metricsDict;
        }

        private class ModelInput
        {
            public bool Label { get; set; }

            public bool PredictedLabel { get; set; }

            public float Score { get; set; }
        }
    }

    public class RegressionGroupMetric : IGroupMetric
    {
        private readonly IDataView _eval;
        private readonly string _labelColumn;
        private readonly string _scoreColumn;
        private readonly string _sensitiveFeatureColumn;
        private readonly MLContext _context = new MLContext();

        public RegressionGroupMetric(IDataView eval, string labelColumn, string scoreColumn, string sensitiveFeatureColumn)
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
            result["MSE"] = DataFrameColumn.Create("MSE", groupMetric.Keys.Select(k => groupMetric[k].MeanSquaredError));
            result["MAE"] = DataFrameColumn.Create("MAE", groupMetric.Keys.Select(k => groupMetric[k].MeanAbsoluteError));
            return result;
        }

        public Dictionary<string, double> DifferenceBetweenGroups()
        {
            Dictionary<string, double> diffDict = new Dictionary<string, double>();
            DataFrame groupMetrics = ByGroup();
            diffDict.Add("RSquared", Math.Abs((double)groupMetrics["RSquared"].Max() - (double)groupMetrics["RSquared"].Min()));
            diffDict.Add("RMS", Math.Abs((double)groupMetrics["RMS"].Max() - (double)groupMetrics["RMS"].Min()));
            diffDict.Add("MSE", Math.Abs((double)groupMetrics["MSE"].Max() - (double)groupMetrics["MSE"].Min()));
            diffDict.Add("MAE", Math.Abs((double)groupMetrics["MAE"].Max() - (double)groupMetrics["MAE"].Min()));

            return diffDict;
        }

        public Dictionary<string, double> Overall()
        {
            RegressionMetrics metrics = _context.Regression.Evaluate(_eval, _labelColumn);

            // create the dictionary to hold the results
            Dictionary<string, double> metricsDict = new Dictionary<string, double>
            {
                { "RSquared", metrics.RSquared },
                { "RMS", metrics.RootMeanSquaredError },
                { "MSE", metrics.MeanSquaredError },
                { "MAE", metrics.MeanAbsoluteError }
            };

            return metricsDict;
        }
    }
}
