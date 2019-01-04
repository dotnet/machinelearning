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
                    new Data.TextLoader.Column("Id", Data.DataKind.TX, 0),
                    new Data.TextLoader.Column("Date", Data.DataKind.TX, 1),
                    new Data.TextLoader.Column("Label", Data.DataKind.R4, 2),
                    new Data.TextLoader.Column("BedRooms", Data.DataKind.R4, 3),
                    new Data.TextLoader.Column("BathRooms", Data.DataKind.R4, 4),
                    new Data.TextLoader.Column("SqftLiving", Data.DataKind.R4, 5),
                    new Data.TextLoader.Column("SqftLot", Data.DataKind.R4, 6),
                    new Data.TextLoader.Column("Floors", Data.DataKind.R4, 7),
                    new Data.TextLoader.Column("WaterFront", Data.DataKind.R4, 8),
                    new Data.TextLoader.Column("View", Data.DataKind.R4, 9),
                    new Data.TextLoader.Column("Condition", Data.DataKind.R4, 10),
                    new Data.TextLoader.Column("Grade", Data.DataKind.R4, 11),
                    new Data.TextLoader.Column("SqftAbove", Data.DataKind.R4, 12),
                    new Data.TextLoader.Column("SqftBasement", Data.DataKind.R4, 13),
                    new Data.TextLoader.Column("YearBuilt", Data.DataKind.R4, 14),
                    new Data.TextLoader.Column("YearRenovated", Data.DataKind.R4, 15),
                    new Data.TextLoader.Column("Zipcode", Data.DataKind.R4, 16),
                    new Data.TextLoader.Column("Lat", Data.DataKind.R4, 17),
                    new Data.TextLoader.Column("Long", Data.DataKind.R4, 18),
                    new Data.TextLoader.Column("SqftLiving15", Data.DataKind.R4, 19),
                    new Data.TextLoader.Column("SqftLot15", Data.DataKind.R4, 20)
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
