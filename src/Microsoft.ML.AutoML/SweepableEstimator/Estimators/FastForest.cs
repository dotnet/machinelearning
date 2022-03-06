// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Trainers.FastTree;

namespace Microsoft.ML.AutoML.CodeGen
{
    internal partial class FastForestOva
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, FastForestOption param)
        {
            var option = new FastForestBinaryTrainer.Options()
            {
                NumberOfTrees = param.NumberOfTrees,
                LabelColumnName = param.LabelColumnName,
                FeatureColumnName = param.FeatureColumnName,
                ExampleWeightColumnName = param.ExampleWeightColumnName,
                FeatureFraction = param.FeatureFraction,
                NumberOfThreads = AutoMlUtils.GetNumberOfThreadFromEnvrionment(),
            };

            return context.MulticlassClassification.Trainers.OneVersusAll(context.BinaryClassification.Trainers.FastForest(option), labelColumnName: param.LabelColumnName);
        }
    }

    internal partial class FastForestRegression
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, FastForestOption param)
        {
            var option = new FastForestRegressionTrainer.Options()
            {
                NumberOfTrees = param.NumberOfTrees,
                FeatureFraction = param.FeatureFraction,
                LabelColumnName = param.LabelColumnName,
                FeatureColumnName = param.FeatureColumnName,
                ExampleWeightColumnName = param.ExampleWeightColumnName,
                NumberOfThreads = AutoMlUtils.GetNumberOfThreadFromEnvrionment(),
            };

            return context.Regression.Trainers.FastForest(option);
        }
    }

    internal partial class FastForestBinary
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, FastForestOption param)
        {
            var option = new FastForestBinaryTrainer.Options()
            {
                NumberOfTrees = param.NumberOfTrees,
                NumberOfLeaves = param.NumberOfLeaves,
                FeatureFraction = param.FeatureFraction,
                LabelColumnName = param.LabelColumnName,
                FeatureColumnName = param.FeatureColumnName,
                ExampleWeightColumnName = param.ExampleWeightColumnName,
                NumberOfThreads = AutoMlUtils.GetNumberOfThreadFromEnvrionment(),
            };

            return context.BinaryClassification.Trainers.FastForest(option);
        }
    }
}
