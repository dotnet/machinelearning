// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Models;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.TestFramework;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Scenarios
{
    public partial class ScenariosTests : BaseTestClass
    {
        /*
         A real-estate firm Contoso wants to add a house price prediction to their ASP.NET/Xamarin application.
         The application will let users submit information about their house, and see a price they could expect if they put the house for sale.
         Because real estate transaction data is public, Contoso has historical data they intend to use to train Machine Learning prediction engine. 
        */
        [Fact(Skip = "Missing data set. See https://github.com/dotnet/machinelearning/issues/3")]
        public async void PredictHousePriceModelTest()
        {
            string modelFilePath = GetOutputPath("PredictHousePriceModelTest.zip");
            ModelHelper.WriteKcHousePriceModel(GetDataPath("kc_house_data.csv"), modelFilePath);

            PredictionModel<HousePriceData, HousePricePrediction> model = await PredictionModel.ReadAsync<HousePriceData, HousePricePrediction>(modelFilePath);

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
        }

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

        public class HousePriceData
        {
            [Column(ordinal: "0")]
            public string Id;

            [Column(ordinal: "1")]
            public string Date;

            [Column(ordinal: "2", name: "Label")]
            public float Price;

            [Column(ordinal: "3")]
            public float Bedrooms;

            [Column(ordinal: "4")]
            public float Bathrooms;

            [Column(ordinal: "5")]
            public float SqftLiving;

            [Column(ordinal: "6")]
            public float SqftLot;

            [Column(ordinal: "7")]
            public float Floors;

            [Column(ordinal: "8")]
            public float Waterfront;

            [Column(ordinal: "9")]
            public float View;

            [Column(ordinal: "10")]
            public float Condition;

            [Column(ordinal: "11")]
            public float Grade;

            [Column(ordinal: "12")]
            public float SqftAbove;

            [Column(ordinal: "13")]
            public float SqftBasement;

            [Column(ordinal: "14")]
            public float YearBuilt;

            [Column(ordinal: "15")]
            public float YearRenovated;

            [Column(ordinal: "16")]
            public float Zipcode;

            [Column(ordinal: "17")]
            public float Lat;

            [Column(ordinal: "18")]
            public float Long;

            [Column(ordinal: "19")]
            public float SqftLiving15;

            [Column(ordinal: "20")]
            public float SqftLot15;
        }

        public class HousePricePrediction
        {
            [ColumnName("Score")]
            public float Price;
        }

        public ScenariosTests(ITestOutputHelper output) : base(output)
        {
        }
    }
}

