// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.


using System;
using Xunit;


namespace Microsoft.ML.Fairlearn.Tests
{
    public class UnitTest1
    {
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
        public void Test1()
        {
            MLContext mlContext = new MLContext();
            IDataView data = mlContext.Data.LoadFromEnumerable(houseData);
            RegressionMetric regressionMetric = mlContext.Fairlearn().Metric.Regression(eval: data, labelColumn: "Price", scoreColumn: "Score", sensitiveFeatureColumn: "Gender");
            var metricByGroup = regressionMetric.ByGroup();
            Assert.Equal(-2.30578, Convert.ToSingle(metricByGroup["RSquared"][0]), 3);
            Assert.Equal(-2039.81453, Convert.ToSingle(metricByGroup["RSquared"][1]), 3);
            Assert.Equal(1.00000, Convert.ToSingle(metricByGroup["RMS"][0]), 3);
            Assert.Equal(15.811388, Convert.ToSingle(metricByGroup["RMS"][1]), 3);
        }
    }
}
