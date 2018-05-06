// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Xunit;

namespace Microsoft.ML.Scenarios
{
    public partial class Top5Scenarios
    {
        [Fact(Skip = "Missing data set. See https://github.com/dotnet/machinelearning/issues/3")]
        public void TrainAndPredictHousePriceModelTest()
        {
            string dataPath = GetDataPath("kc_house_data.csv");

            var pipeline = new LearningPipeline();

            pipeline.Add(new TextLoader<HousePriceData>(dataPath, useHeader: true, separator: ","));

            pipeline.Add(new ColumnConcatenator(outputColumn: "NumericalFeatures",
                "SqftLiving", "SqftLot", "SqftAbove", "SqftBasement", "Lat", "Long", "SqftLiving15", "SqftLot15"));

            pipeline.Add(new ColumnConcatenator(outputColumn: "CategoryFeatures",
                 "Bedrooms", "Bathrooms", "Floors", "Waterfront", "View", "Condition", "Grade", "YearBuilt", "YearRenovated", "Zipcode"));

            pipeline.Add(new CategoricalOneHotVectorizer("CategoryFeatures"));
            pipeline.Add(new ColumnConcatenator(outputColumn: "Features",
                "NumericalFeatures", "CategoryFeatures"));
            pipeline.Add(new StochasticDualCoordinateAscentRegressor());

            PredictionModel<HousePriceData, HousePricePrediction> model = pipeline.Train<HousePriceData, HousePricePrediction>();

            HousePricePrediction prediction = model.Predict(new HousePriceData()
            {
                Bedrooms = 3,
                Bathrooms = 2,
                SqftLiving = 1710,
                SqftLot = 4697,
                Floors = 1.5f,
                Waterfront = 0,
                View = 0,
                Condition = 5,
                Grade = 6,
                SqftAbove = 1710,
                SqftBasement = 0,
                YearBuilt = 1941,
                YearRenovated = 0,
                Zipcode = 98002,
                Lat = 47.3048f,
                Long = -122.218f,
                SqftLiving15 = 1030,
                SqftLot15 = 4705
            });

            Assert.InRange(prediction.Price, 260_000, 330_000);

            string testDataPath = GetDataPath("kc_house_test.csv");
            var testData = new TextLoader<HousePriceData>(testDataPath, useHeader: true, separator: ",");

            var evaluator = new RegressionEvaluator();
            RegressionMetrics metrics = evaluator.Evaluate(model, testData);
            Assert.InRange(metrics.L1, 85_000, 89_000);
            Assert.InRange(metrics.L2, 17_000_000_000, 19_000_000_000);
            Assert.InRange(metrics.Rms, 130_500, 135_000);
            Assert.InRange(metrics.LossFn, 17_000_000_000, 19_000_000_000);
            Assert.Equal(.8, metrics.RSquared, 1);
        }
    }
}

