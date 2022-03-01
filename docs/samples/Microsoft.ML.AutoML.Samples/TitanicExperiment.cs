using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;
using Microsoft.Data.Analysis;
using Microsoft.ML.SearchSpace;
using Microsoft.ML.SearchSpace.Tuner;
using Microsoft.ML.Transforms.Text;
using static Microsoft.ML.Transforms.Text.NgramExtractingEstimator;
using static Microsoft.ML.Transforms.Text.TextNormalizingEstimator;

namespace Microsoft.ML.AutoML.Samples
{
    /// <summary>
    /// titanic experiment using sweepable api and customize search space
    /// </summary>
    internal static class TitanicExperiment
    {
        private static string TrainDataPath = "<Path to your train dataset goes here>";

        public static void Run()
        {
            var context = new MLContext();
            var ss = new SearchSpace<FastTreeOption>();
            var textFeaturizeSS = new SearchSpace<FeaturizeTextOption>();
            var pipeline = context.Transforms.Categorical.OneHotEncoding(new[] { new InputOutputColumnPair(@"Sex", @"Sex"), new InputOutputColumnPair(@"Embarked", @"Embarked") })
                           .Append(context.Transforms.ReplaceMissingValues(new[] { new InputOutputColumnPair(@"Pclass", @"Pclass"), new InputOutputColumnPair(@"Age", @"Age"), new InputOutputColumnPair(@"SibSp", @"SibSp"), new InputOutputColumnPair(@"Parch", @"Parch"), new InputOutputColumnPair(@"Fare", @"Fare") }))
                           .Append(context.Transforms.Concatenate(@"TextFeature", @"Name", "Ticket", "Cabin"))
                           .Append(context.Auto().CreateSweepableEstimator(
                               (mlContext, option) =>
                               {
                                   var textOption = new TextFeaturizingEstimator.Options
                                   {
                                       CaseMode = option.CaseMode,
                                       KeepDiacritics = option.KeepDiacritics,
                                       KeepNumbers = option.KeepNumbers,
                                       KeepPunctuations = option.KeepPunctuations,
                                       CharFeatureExtractor = new WordBagEstimator.Options()
                                       {
                                           NgramLength = option.WordBagEstimatorOption.NgramLength,
                                           UseAllLengths = option.WordBagEstimatorOption.UseAllLengths,
                                           Weighting = option.WordBagEstimatorOption.WeightingCriteria,
                                       },
                                   };

                                   return context.Transforms.Text.FeaturizeText("TextFeature", textOption);
                               },
                               textFeaturizeSS))
                           .Append(context.Transforms.Concatenate(@"Features", new[] { @"Sex", @"Embarked", @"Pclass", @"Age", @"SibSp", @"Parch", @"Fare", "TextFeature" }))
                           .Append(context.Transforms.Conversion.ConvertType("Survived", "Survived", Data.DataKind.Boolean))
                           .Append(context.Auto().CreateSweepableEstimator(
                               (mlContext, option) =>
                               {
                                   return mlContext.BinaryClassification.Trainers.FastForest(labelColumnName: "Survived", featureColumnName: "Features", numberOfLeaves: option.NumberOfLeaves, numberOfTrees: option.NumberOfTrees);
                               },
                               ss))
                           .Append(context.BinaryClassification.Calibrators.Naive(labelColumnName: @"Survived", scoreColumnName: @"Score"));

            var tuner = new GridSearchTuner(pipeline.SearchSpace);
            var df = DataFrame.LoadCsv(TrainDataPath);
            var trainTestSplit = context.Data.TrainTestSplit(df, 0.1);
            var bestAccuracy = 0.0;
            var i = 0;

            foreach (var param in tuner.Propose())
            {
                Console.WriteLine($"trial {i++}");

                var trainingPipeline = pipeline.BuildTrainingPipeline(context, param);
                var model = trainingPipeline.Fit(trainTestSplit.TrainSet);
                var eval = model.Transform(trainTestSplit.TestSet);
                var accuracy = context.BinaryClassification.Evaluate(eval, "Survived").Accuracy;
                if (accuracy > bestAccuracy)
                {
                    Console.WriteLine("Found best accuracy");
                    Console.WriteLine("Current best parameter");
                    Console.WriteLine(JsonSerializer.Serialize(param));
                    bestAccuracy = accuracy;
                }

                Console.WriteLine($"Trial {i}: Current Best Accuracy {bestAccuracy}, Current Accuracy {accuracy}");
            }

        }

        private class FastTreeOption
        {
            [Range(2, 1024)]
            public int NumberOfTrees { get; set; }

            [Range(2, 1024)]
            public int NumberOfLeaves { get; set; }
        }

        private class FeaturizeTextOption
        {
            [Choice(0, 1, 2)]
            public CaseMode CaseMode { get; set; }

            [BooleanChoice]
            public bool KeepDiacritics { get; set; }

            [BooleanChoice]
            public bool KeepNumbers { get; set; }

            [BooleanChoice]
            public bool KeepPunctuations { get; set; }

            [Option]
            public WordBagEstimatorOption WordBagEstimatorOption { get; set; }
        }

        private class WordBagEstimatorOption
        {
            [Range(1, 10)]
            public int NgramLength { get; set; }

            [BooleanChoice]
            public bool UseAllLengths { get; set; }

            [Choice(0, 1, 2)]
            public WeightingCriteria WeightingCriteria { get; set; }
        }
    }
}
