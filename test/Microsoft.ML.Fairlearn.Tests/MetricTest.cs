// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.Data.Analysis;
using Xunit;

namespace Microsoft.ML.Fairlearn.Tests
{
    public class MetricTest
    {
        MLContext mlContext;
        IDataView data;
        public MetricTest()
        {
            mlContext = new MLContext();
            data = mlContext.Data.LoadFromEnumerable(houseData);
        }

        public class HouseData
        {
            public float Size { get; set; }
            public float Price { get; set; }
            public float Score { get; set; }
            public string Gender { get; set; }
        }

        HouseData[] houseData = {
                new HouseData() { Size = 1.1F, Price = 0.2F, Gender = "Male", Score = 1.2F},
                new HouseData() { Size = 1.9F, Price = 1.3F, Gender = "Male", Score = 2.3F },
                new HouseData() { Size = 2.8F, Price = 3.0F, Gender = "Female", Score = 25.0F },
                new HouseData() { Size = 3.4F, Price = 3.7F, Gender = "Female", Score = 7.7F } };

        [Fact]
        public void RegressionMetricTest()
        {
            RegressionGroupMetric regressionMetric = mlContext.Fairlearn().Metric.Regression(eval: data, labelColumn: "Price", scoreColumn: "Score", sensitiveFeatureColumn: "Gender");
            var metricByGroup = regressionMetric.ByGroup();
            Assert.Equal(-2.30578, Convert.ToSingle(metricByGroup["RSquared"][0]), 3);
            Assert.Equal(-2039.81453, Convert.ToSingle(metricByGroup["RSquared"][1]), 3);
            Assert.Equal(1.00000, Convert.ToSingle(metricByGroup["RMS"][0]), 3);
            Assert.Equal(15.811388, Convert.ToSingle(metricByGroup["RMS"][1]), 3);
            metricByGroup.Description();
            Dictionary<string, double> metricOverall = regressionMetric.Overall();
            Assert.Equal(125.5, metricOverall["MSE"], 1);
            Assert.Equal(11.202678, metricOverall["RMS"], 4);
            Dictionary<string, double> diff = regressionMetric.DifferenceBetweenGroups();
            Assert.Equal(14.81138, diff["RMS"], 4);
            Assert.Equal(2037.5, diff["RSquared"], 1);

        }

        [Fact]
        public void BinaryClassificationMetricTest()
        {
            //create dummy dataset
            bool[] vs = { true, true, true, true, true, true, true, false, false, false };
            PrimitiveDataFrameColumn<bool> label = new PrimitiveDataFrameColumn<bool>("label", vs);
            string[] str = { "a", "b", "a", "a", "b", "a", "b", "b", "a", "b" };
            StringDataFrameColumn groupId = new StringDataFrameColumn("group_id", str);
            bool[] fl = { true, true, true, true, false, false, false, false, false, false };
            PrimitiveDataFrameColumn<bool> pred = new PrimitiveDataFrameColumn<bool>("PredictedLabel", fl);
            float[] fl2 = { 1.0F, 1.0F, 1.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F };
            PrimitiveDataFrameColumn<float> score = new PrimitiveDataFrameColumn<float>("Score", fl2);
            float[] fl3 = { 1.0F, 1.0F, 1.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F };
            PrimitiveDataFrameColumn<float> prob = new PrimitiveDataFrameColumn<float>("Probability", fl3);
            DataFrame df = new DataFrame(label, groupId, pred, score, prob);

            BinaryGroupMetric metrics = mlContext.Fairlearn().Metric.BinaryClassification(eval: df, labelColumn: "label", predictedColumn: "PredictedLabel", sensitiveFeatureColumn: "group_id");
            var metricByGroup = metrics.ByGroup();
            Assert.Equal(0.8, Convert.ToSingle(metricByGroup["Accuracy"][0]), 1);
            Assert.Equal(0.6, Convert.ToSingle(metricByGroup["Accuracy"][1]), 1);
            var metricOverall = metrics.Overall();
            Assert.Equal(0.7, Convert.ToSingle(metricOverall["Accuracy"]), 1);
        }
    }
}
