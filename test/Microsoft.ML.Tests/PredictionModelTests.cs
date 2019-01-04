// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.EntryPoints.Tests
{
    public class PredictionModelTests : BaseTestClass
    {
        public class HousePriceData
        {
            public float Bedrooms;
            public float Bathrooms;
            public float SqftLiving;
            public float SqftLot;
            public float Floors;
            public float Waterfront;
            public float View;
            public float Condition;
            public float Grade;
            public float SqftAbove;
            public float SqftBasement;
            public float YearBuilt;
            public float YearRenovated;
            public float Zipcode;
            public float Lat;
            public float Long;
            public float SqftLiving15;
            public float SqftLot15;
        }

        public class HousePricePrediction
        {
            [ColumnName("Score")]
            public float Price;
        }

        [Fact(Skip = "Missing data set. See https://github.com/dotnet/machinelearning/issues/3")]
        public void ReadStrongTypeModelFromStream()
        {
            var mlContext = new MLContext(seed: 1);
            var data = ModelHelper.GetKcHouseDataView(mlContext, GetDataPath("kc_house_data.csv"));
            var pipeline = ModelHelper.GetKcHousePipeline(mlContext);
            var model = pipeline.Fit(data);

            var engine = model.CreatePredictionEngine<HousePriceData, HousePricePrediction>(mlContext);

            HousePricePrediction prediction = engine.Predict(new HousePriceData()
            {
                Bedrooms = 3,
                Bathrooms = 1.75f,
                SqftLiving = 2450,
                SqftLot = 2691,
                Floors = 2,
                Waterfront = 0,
                View = 0,
                Condition = 3,
                Grade = 8,
                SqftAbove = 1750,
                SqftBasement = 700,
                YearBuilt = 1915,
                YearRenovated = 0,
                Zipcode = 98119,
                Lat = 47.6386f,
                Long = -122.36f,
                SqftLiving15 = 1760,
                SqftLot15 = 3573
            });

            Assert.InRange(prediction.Price, 790_000, 850_000);

            var dataView = model.Transform(data);
            dataView.Schema.TryGetColumnIndex("Score", out int scoreColumn);
            using (var cursor = dataView.GetRowCursor((int col) => col == scoreColumn))
            {
                var scoreGetter = cursor.GetGetter<float>(scoreColumn);
                float score = 0;
                cursor.MoveNext();
                scoreGetter(ref score);
                Assert.InRange(score, 100_000, 200_000);
            }
        }

        public PredictionModelTests(ITestOutputHelper output)
            : base(output)
        {
        }
    }
}
