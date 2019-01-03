// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Legacy.Data;

namespace Microsoft.ML.TestFramework
{
#pragma warning disable 612, 618
    public static class ModelHelper
    {
        private static MLContext mlContext = new MLContext(seed: 1);
        private static ITransformer s_housePriceModel;

        public static void WriteKcHousePriceModel(string dataPath, string outputModelPath)
        {
            if (File.Exists(outputModelPath))
            {
                File.Delete(outputModelPath);
            }

            using (var saveStream = File.OpenWrite(outputModelPath))
            {
                WriteKcHousePriceModel(dataPath, saveStream);
            }
        }

        public static void WriteKcHousePriceModel(string dataPath, Stream stream)
        {
            if (s_housePriceModel == null)
            {
                s_housePriceModel = CreateKcHousePricePredictorModel(dataPath);
            }
            mlContext.Model.Save(s_housePriceModel, stream);
        }

        public static IDataView GetKcHouseDataView(string dataPath)
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

        private static ITransformer CreateKcHousePricePredictorModel(string dataPath)
        {
            Experiment experiment = mlContext.CreateExperiment();

            var data = GetKcHouseDataView(dataPath);
            var pipeline = mlContext.Transforms.Concatenate("NumericalFeatures", "SqftLiving", "SqftLot", "SqftAbove", "SqftBasement", "Lat", "Long", "SqftLiving15", "SqftLot15")
                .Append(mlContext.Transforms.Concatenate("CategoryFeatures", "Bedrooms", "Bathrooms", "Floors", "Waterfront", "View", "Condition", "Grade", "YearBuilt", "YearRenovated", "Zipcode"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("CategoryFeatures"))
                .Append(mlContext.Transforms.Concatenate("Features", "NumericalFeatures", "CategoryFeatures"))
                .Append(mlContext.Regression.Trainers.StochasticDualCoordinateAscent(advancedSettings: s => { s.NumThreads = 1; }));

            //var numericalConcatenate = new Legacy.Transforms.ColumnConcatenator();
            //numericalConcatenate.Data = GetKcHouseDataView(dataPath);
            //numericalConcatenate.AddColumn("NumericalFeatures", "SqftLiving", "SqftLot", "SqftAbove", "SqftBasement", "Lat", "Long", "SqftLiving15", "SqftLot15");
            //Legacy.Transforms.ColumnConcatenator.Output numericalConcatenated = experiment.Add(numericalConcatenate);

            //var categoryConcatenate = new Legacy.Transforms.ColumnConcatenator();
            //categoryConcatenate.Data = numericalConcatenated.OutputData;
            //categoryConcatenate.AddColumn("CategoryFeatures", "Bedrooms", "Bathrooms", "Floors", "Waterfront", "View", "Condition", "Grade", "YearBuilt", "YearRenovated", "Zipcode");
            //Legacy.Transforms.ColumnConcatenator.Output categoryConcatenated = experiment.Add(categoryConcatenate);

            //var categorize = new Legacy.Transforms.CategoricalOneHotVectorizer();
            //categorize.AddColumn("CategoryFeatures");
            //categorize.Data = categoryConcatenated.OutputData;
            //Legacy.Transforms.CategoricalOneHotVectorizer.Output categorized = experiment.Add(categorize);

            //var featuresConcatenate = new Legacy.Transforms.ColumnConcatenator();
            //featuresConcatenate.Data = categorized.OutputData;
            //featuresConcatenate.AddColumn("Features", "NumericalFeatures", "CategoryFeatures");
            //Legacy.Transforms.ColumnConcatenator.Output featuresConcatenated = experiment.Add(featuresConcatenate);

            //var learner = new Legacy.Trainers.StochasticDualCoordinateAscentRegressor();
            //learner.TrainingData = featuresConcatenated.OutputData;
            //learner.NumThreads = 1;
            //Legacy.Trainers.StochasticDualCoordinateAscentRegressor.Output learnerOutput = experiment.Add(learner);

            //var combineModels = new Legacy.Transforms.ManyHeterogeneousModelCombiner();
            //combineModels.TransformModels = new ArrayVar<TransformModel>(numericalConcatenated.Model, categoryConcatenated.Model, categorized.Model, featuresConcatenated.Model);
            //combineModels.PredictorModel = learnerOutput.PredictorModel;
            //Legacy.Transforms.ManyHeterogeneousModelCombiner.Output combinedModels = experiment.Add(combineModels);

            //var scorer = new Legacy.Transforms.Scorer
            //{
            //    PredictorModel = combinedModels.PredictorModel
            //};

            //var scorerOutput = experiment.Add(scorer);
            //experiment.Compile();
            //experiment.SetInput(importData.InputFile, new SimpleFileHandle(mlContext, dataPath, false, false));
            //experiment.Run();

            //return experiment.GetOutput(scorerOutput.ScoringTransform);
            return pipeline.Fit(data);
        }
    }
#pragma warning restore 612, 618
}
