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
            [LoadColumn(range: "0")]
            public string Id;

            [LoadColumn(range: "1")]
            public string Date;

            [LoadColumn(range: "2", name: "Label")]
            public float Price;

            [LoadColumn(range: "3")]
            public float Bedrooms;

            [LoadColumn(range: "4")]
            public float Bathrooms;

            [LoadColumn(range: "5")]
            public float SqftLiving;

            [LoadColumn(range: "6")]
            public float SqftLot;

            [LoadColumn(range: "7")]
            public float Floors;

            [LoadColumn(range: "8")]
            public float Waterfront;

            [LoadColumn(range: "9")]
            public float View;

            [LoadColumn(range: "10")]
            public float Condition;

            [LoadColumn(range: "11")]
            public float Grade;

            [LoadColumn(range: "12")]
            public float SqftAbove;

            [LoadColumn(range: "13")]
            public float SqftBasement;

            [LoadColumn(range: "14")]
            public float YearBuilt;

            [LoadColumn(range: "15")]
            public float YearRenovated;

            [LoadColumn(range: "16")]
            public float Zipcode;

            [LoadColumn(range: "17")]
            public float Lat;

            [LoadColumn(range: "18")]
            public float Long;

            [LoadColumn(range: "19")]
            public float SqftLiving15;

            [LoadColumn(range: "20")]
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

