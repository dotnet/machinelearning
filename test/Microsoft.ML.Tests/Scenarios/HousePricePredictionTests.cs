// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.TestFramework;
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
#pragma warning disable 612
        [Fact(Skip = "Missing data set. See https://github.com/dotnet/machinelearning/issues/3")]
        public async void PredictHousePriceModelTest()
        {
            string modelFilePath = GetOutputPath("PredictHousePriceModelTest.zip");
            ModelHelper.WriteKcHousePriceModel(GetDataPath("kc_house_data.csv"), modelFilePath);

            var model = await Legacy.PredictionModel.ReadAsync<HousePriceData, HousePricePrediction>(modelFilePath);

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
#pragma warning restore 612

        public class HousePriceData
        {
            [LoadColumn(0)]
            public string Id;

            [LoadColumn(1)]
            public string Date;

            [LoadColumn(2), ColumnName("Label")]
            public float Price;

            [LoadColumn(3)]
            public float Bedrooms;

            [LoadColumn(4)]
            public float Bathrooms;

            [LoadColumn(5)]
            public float SqftLiving;

            [LoadColumn(6)]
            public float SqftLot;

            [LoadColumn(7)]
            public float Floors;

            [LoadColumn(8)]
            public float Waterfront;

            [LoadColumn(9)]
            public float View;

            [LoadColumn(10)]
            public float Condition;

            [LoadColumn(11)]
            public float Grade;

            [LoadColumn(12)]
            public float SqftAbove;

            [LoadColumn(13)]
            public float SqftBasement;

            [LoadColumn(14)]
            public float YearBuilt;

            [LoadColumn(15)]
            public float YearRenovated;

            [LoadColumn(16)]
            public float Zipcode;

            [LoadColumn(17)]
            public float Lat;

            [LoadColumn(18)]
            public float Long;

            [LoadColumn(19)]
            public float SqftLiving15;

            [LoadColumn(20)]
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

