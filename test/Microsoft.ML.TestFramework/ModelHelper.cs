﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Legacy.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using System.IO;

namespace Microsoft.ML.TestFramework
{
#pragma warning disable 612, 618
    public static class ModelHelper
    {
        private static IHostEnvironment s_environment = new MLContext(seed: 1);
        private static ITransformModel s_housePriceModel;

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
            s_housePriceModel.Save(s_environment, stream);
        }

        public static IDataView GetKcHouseDataView(string dataPath)
        {
            var dataSchema = "col=Id:TX:0 col=Date:TX:1 col=Label:R4:2 col=Bedrooms:R4:3 " +
                "col=Bathrooms:R4:4 col=SqftLiving:R4:5 col=SqftLot:R4:6 col=Floors:R4:7 " +
                "col=Waterfront:R4:8 col=View:R4:9 col=Condition:R4:10 col=Grade:R4:11 " +
                "col=SqftAbove:R4:12 col=SqftBasement:R4:13 col=YearBuilt:R4:14 " +
                "col=YearRenovated:R4:15 col=Zipcode:R4:16 col=Lat:R4:17 col=Long:R4:18 " +
                "col=SqftLiving15:R4:19 col=SqftLot15:R4:20 header+ sep=,";

            var txtArgs = new Runtime.Data.TextLoader.Arguments();
            bool parsed = CmdParser.ParseArguments(s_environment, dataSchema, txtArgs);
            s_environment.Assert(parsed);
            return Runtime.Data.TextLoader.ReadFile(s_environment, txtArgs, new MultiFileSource(dataPath));
        }

        private static ITransformModel CreateKcHousePricePredictorModel(string dataPath)
        {
            Experiment experiment = s_environment.CreateExperiment();
            var importData = new Legacy.Data.TextLoader(dataPath)
            {
                Arguments = new TextLoaderArguments
                {
                    Separator = new[] { ',' },
                    HasHeader = true,
                    Column = new[]
                    {
                        new TextLoaderColumn()
                        {
                            Name = "Id",
                            Source = new [] { new TextLoaderRange(0) },
                            Type =  Legacy.Data.DataKind.Text
                        },

                        new TextLoaderColumn()
                        {
                            Name = "Date",
                            Source = new [] { new TextLoaderRange(1) },
                            Type =  Legacy.Data.DataKind.Text
                        },

                        new TextLoaderColumn()
                        {
                            Name = "Label",
                            Source = new [] { new TextLoaderRange(2) },
                            Type =  Legacy.Data.DataKind.Num
                        },

                        new TextLoaderColumn()
                        {
                            Name = "Bedrooms",
                            Source = new [] { new TextLoaderRange(3) },
                            Type =  Legacy.Data.DataKind.Num
                        },

                        new TextLoaderColumn()
                        {
                            Name = "Bathrooms",
                            Source = new [] { new TextLoaderRange(4) },
                            Type =  Legacy.Data.DataKind.Num
                        },

                        new TextLoaderColumn()
                        {
                            Name = "SqftLiving",
                            Source = new [] { new TextLoaderRange(5) },
                            Type =  Legacy.Data.DataKind.Num
                        },

                        new TextLoaderColumn()
                        {
                            Name = "SqftLot",
                            Source = new [] { new TextLoaderRange(6) },
                            Type =  Legacy.Data.DataKind.Num
                        },

                        new TextLoaderColumn()
                        {
                            Name = "Floors",
                            Source = new [] { new TextLoaderRange(7) },
                            Type =  Legacy.Data.DataKind.Num
                        },

                        new TextLoaderColumn()
                        {
                            Name = "Waterfront",
                            Source = new [] { new TextLoaderRange(8) },
                            Type =  Legacy.Data.DataKind.Num
                        },

                        new TextLoaderColumn()
                        {
                            Name = "View",
                            Source = new [] { new TextLoaderRange(9) },
                            Type =  Legacy.Data.DataKind.Num
                        },

                        new TextLoaderColumn()
                        {
                            Name = "Condition",
                            Source = new [] { new TextLoaderRange(10) },
                            Type =  Legacy.Data.DataKind.Num
                        },

                        new TextLoaderColumn()
                        {
                            Name = "Grade",
                            Source = new [] { new TextLoaderRange(11) },
                            Type =  Legacy.Data.DataKind.Num
                        },

                        new TextLoaderColumn()
                        {
                            Name = "SqftAbove",
                            Source = new [] { new TextLoaderRange(12) },
                            Type =  Legacy.Data.DataKind.Num
                        },

                        new TextLoaderColumn()
                        {
                            Name = "SqftBasement",
                            Source = new [] { new TextLoaderRange(13) },
                            Type =  Legacy.Data.DataKind.Num
                        },

                        new TextLoaderColumn()
                        {
                            Name = "YearBuilt",
                            Source = new [] { new TextLoaderRange(14) },
                            Type =  Legacy.Data.DataKind.Num
                        },

                        new TextLoaderColumn()
                        {
                            Name = "YearRenovated",
                            Source = new [] { new TextLoaderRange(15) },
                            Type =  Legacy.Data.DataKind.Num
                        },

                        new TextLoaderColumn()
                        {
                            Name = "Zipcode",
                            Source = new [] { new TextLoaderRange(16) },
                            Type =  Legacy.Data.DataKind.Num
                        },

                        new TextLoaderColumn()
                        {
                            Name = "Lat",
                            Source = new [] { new TextLoaderRange(17) },
                            Type =  Legacy.Data.DataKind.Num
                        },

                        new TextLoaderColumn()
                        {
                            Name = "Long",
                            Source = new [] { new TextLoaderRange(18) },
                            Type =  Legacy.Data.DataKind.Num
                        },

                        new TextLoaderColumn()
                        {
                            Name = "SqftLiving15",
                            Source = new [] { new TextLoaderRange(19) },
                            Type =  Legacy.Data.DataKind.Num
                        },

                        new TextLoaderColumn()
                        {
                            Name = "SqftLot15",
                            Source = new [] { new TextLoaderRange(20) },
                            Type =  Legacy.Data.DataKind.Num
                        },
                    }
                }

                //new Data.CustomTextLoader();
                // importData.CustomSchema = dataSchema;
                //
            };

            Legacy.Data.TextLoader.Output imported = experiment.Add(importData);
            var numericalConcatenate = new Legacy.Transforms.ColumnConcatenator();
            numericalConcatenate.Data = imported.Data;
            numericalConcatenate.AddColumn("NumericalFeatures", "SqftLiving", "SqftLot", "SqftAbove", "SqftBasement", "Lat", "Long", "SqftLiving15", "SqftLot15");
            Legacy.Transforms.ColumnConcatenator.Output numericalConcatenated = experiment.Add(numericalConcatenate);

            var categoryConcatenate = new Legacy.Transforms.ColumnConcatenator();
            categoryConcatenate.Data = numericalConcatenated.OutputData;
            categoryConcatenate.AddColumn("CategoryFeatures", "Bedrooms", "Bathrooms", "Floors", "Waterfront", "View", "Condition", "Grade", "YearBuilt", "YearRenovated", "Zipcode");
            Legacy.Transforms.ColumnConcatenator.Output categoryConcatenated = experiment.Add(categoryConcatenate);

            var categorize = new Legacy.Transforms.CategoricalOneHotVectorizer();
            categorize.AddColumn("CategoryFeatures");
            categorize.Data = categoryConcatenated.OutputData;
            Legacy.Transforms.CategoricalOneHotVectorizer.Output categorized = experiment.Add(categorize);

            var featuresConcatenate = new Legacy.Transforms.ColumnConcatenator();
            featuresConcatenate.Data = categorized.OutputData;
            featuresConcatenate.AddColumn("Features", "NumericalFeatures", "CategoryFeatures");
            Legacy.Transforms.ColumnConcatenator.Output featuresConcatenated = experiment.Add(featuresConcatenate);

            var learner = new Legacy.Trainers.StochasticDualCoordinateAscentRegressor();
            learner.TrainingData = featuresConcatenated.OutputData;
            learner.NumThreads = 1;
            Legacy.Trainers.StochasticDualCoordinateAscentRegressor.Output learnerOutput = experiment.Add(learner);

            var combineModels = new Legacy.Transforms.ManyHeterogeneousModelCombiner();
            combineModels.TransformModels = new ArrayVar<ITransformModel>(numericalConcatenated.Model, categoryConcatenated.Model, categorized.Model, featuresConcatenated.Model);
            combineModels.PredictorModel = learnerOutput.PredictorModel;
            Legacy.Transforms.ManyHeterogeneousModelCombiner.Output combinedModels = experiment.Add(combineModels);

            var scorer = new Legacy.Transforms.Scorer
            {
                PredictorModel = combinedModels.PredictorModel
            };

            var scorerOutput = experiment.Add(scorer);
            experiment.Compile();
            experiment.SetInput(importData.InputFile, new SimpleFileHandle(s_environment, dataPath, false, false));
            experiment.Run();

            return experiment.GetOutput(scorerOutput.ScoringTransform);
        }
    }
#pragma warning restore 612, 618
}
