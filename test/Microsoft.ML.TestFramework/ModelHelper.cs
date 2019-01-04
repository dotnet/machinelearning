// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;

namespace Microsoft.ML.TestFramework
{
    public static class ModelHelper
    {
        public static IDataView GetKcHouseDataView(MLContext mlContext, string dataPath)
        {
            return mlContext.Data.ReadFromTextFile(dataPath, 
                columns: new[]
                {
                    new TextLoader.Column("Id", DataKind.TX, 0),
                    new TextLoader.Column("Date", DataKind.TX, 1),
                    new TextLoader.Column("Label", DataKind.R4, 2),
                    new TextLoader.Column("BedRooms", DataKind.R4, 3),
                    new TextLoader.Column("BathRooms", DataKind.R4, 4),
                    new TextLoader.Column("SqftLiving", DataKind.R4, 5),
                    new TextLoader.Column("SqftLot", DataKind.R4, 6),
                    new TextLoader.Column("Floors", DataKind.R4, 7),
                    new TextLoader.Column("WaterFront", DataKind.R4, 8),
                    new TextLoader.Column("View", DataKind.R4, 9),
                    new TextLoader.Column("Condition", DataKind.R4, 10),
                    new TextLoader.Column("Grade", DataKind.R4, 11),
                    new TextLoader.Column("SqftAbove", DataKind.R4, 12),
                    new TextLoader.Column("SqftBasement", DataKind.R4, 13),
                    new TextLoader.Column("YearBuilt", DataKind.R4, 14),
                    new TextLoader.Column("YearRenovated", DataKind.R4, 15),
                    new TextLoader.Column("Zipcode", DataKind.R4, 16),
                    new TextLoader.Column("Lat", DataKind.R4, 17),
                    new TextLoader.Column("Long", DataKind.R4, 18),
                    new TextLoader.Column("SqftLiving15", DataKind.R4, 19),
                    new TextLoader.Column("SqftLot15", DataKind.R4, 20)
                }, 
                hasHeader: true,
                separatorChar: ','
            );
        }

        public static IEstimator<ITransformer> GetKcHousePipeline(MLContext mlContext)
        {
            // Define pipeline.
            return mlContext.Transforms.Concatenate("NumericalFeatures", "SqftLiving", "SqftLot", "SqftAbove", "SqftBasement", "Lat", "Long", "SqftLiving15", "SqftLot15")
                .Append(mlContext.Transforms.Concatenate("CategoryFeatures", "Bedrooms", "Bathrooms", "Floors", "Waterfront", "View", "Condition", "Grade", "YearBuilt", "YearRenovated", "Zipcode"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("CategoryFeatures"))
                .Append(mlContext.Transforms.Concatenate("Features", "NumericalFeatures", "CategoryFeatures"))
                .Append(mlContext.Regression.Trainers.StochasticDualCoordinateAscent(advancedSettings: s => { s.NumThreads = 1; }));
        }
    }
}
