// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML
{
    internal interface IMetricSettings
    {
        bool IsMaximize { get; }
    }

    internal class BinaryMetricSettings : IMetricSettings
    {
        public BinaryClassificationMetric Metric { get; set; }

        public string PredictedColumn { get; set; }

        public string TruthColumn { get; set; }

        public bool IsMaximize => this.Metric switch
        {
            BinaryClassificationMetric.Accuracy => true,
            BinaryClassificationMetric.AreaUnderPrecisionRecallCurve => true,
            BinaryClassificationMetric.AreaUnderRocCurve => true,
            BinaryClassificationMetric.PositivePrecision => true,
            BinaryClassificationMetric.NegativePrecision => true,
            BinaryClassificationMetric.NegativeRecall => true,
            BinaryClassificationMetric.PositiveRecall => true,
            BinaryClassificationMetric.F1Score => throw new NotImplementedException(),
            _ => throw new NotImplementedException(),
        };
    }

    internal class MultiClassMetricSettings : IMetricSettings
    {
        public MulticlassClassificationMetric Metric { get; set; }

        public string PredictedColumn { get; set; }

        public string TruthColumn { get; set; }

        public bool IsMaximize => this.Metric switch
        {
            MulticlassClassificationMetric.MacroAccuracy => true,
            MulticlassClassificationMetric.MicroAccuracy => true,
            MulticlassClassificationMetric.LogLoss => false,
            MulticlassClassificationMetric.LogLossReduction => false,
            MulticlassClassificationMetric.TopKAccuracy => true,
            _ => throw new NotImplementedException(),
        };
    }

    internal class RegressionMetricSettings : IMetricSettings
    {
        public RegressionMetric Metric { get; set; }

        public string PredictedColumn { get; set; }

        public string TruthColumn { get; set; }

        public bool IsMaximize => this.Metric switch
        {
            RegressionMetric.RSquared => true,
            RegressionMetric.RootMeanSquaredError => false,
            RegressionMetric.MeanSquaredError => false,
            RegressionMetric.MeanAbsoluteError => false,
            _ => throw new NotImplementedException(),
        };
    }
}
