// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Trainers.LightGbm;

namespace Microsoft.ML.AutoML.CodeGen
{
    internal partial class LightGbmMulti
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, LgbmOption param)
        {
            var option = new LightGbmMulticlassTrainer.Options()
            {
                NumberOfLeaves = param.NumberOfLeaves,
                NumberOfIterations = param.NumberOfTrees,
                MinimumExampleCountPerLeaf = param.MinimumExampleCountPerLeaf,
                LearningRate = param.LearningRate,
                NumberOfThreads = AutoMlUtils.GetNumberOfThreadFromEnvrionment(),
                LabelColumnName = param.LabelColumnName,
                FeatureColumnName = param.FeatureColumnName,
                ExampleWeightColumnName = param.ExampleWeightColumnName,
                Booster = new GradientBooster.Options()
                {
                    SubsampleFraction = param.SubsampleFraction,
                    FeatureFraction = param.FeatureFraction,
                    L1Regularization = param.L1Regularization,
                    L2Regularization = param.L2Regularization,
                },
                MaximumBinCountPerFeature = param.MaximumBinCountPerFeature,
            };

            return context.MulticlassClassification.Trainers.LightGbm(option);
        }
    }

    internal partial class LightGbmBinary
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, LgbmOption param)
        {
            var option = new LightGbmBinaryTrainer.Options()
            {
                NumberOfLeaves = param.NumberOfLeaves,
                NumberOfIterations = param.NumberOfTrees,
                MinimumExampleCountPerLeaf = param.MinimumExampleCountPerLeaf,
                LearningRate = param.LearningRate,
                NumberOfThreads = AutoMlUtils.GetNumberOfThreadFromEnvrionment(),
                LabelColumnName = param.LabelColumnName,
                FeatureColumnName = param.FeatureColumnName,
                ExampleWeightColumnName = param.ExampleWeightColumnName,
                Booster = new GradientBooster.Options()
                {
                    SubsampleFraction = param.SubsampleFraction,
                    FeatureFraction = param.FeatureFraction,
                    L1Regularization = param.L1Regularization,
                    L2Regularization = param.L2Regularization,
                },
                MaximumBinCountPerFeature = param.MaximumBinCountPerFeature,
            };

            return context.BinaryClassification.Trainers.LightGbm(option);
        }
    }

    internal partial class LightGbmRegression
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, LgbmOption param)
        {
            var option = new LightGbmRegressionTrainer.Options()
            {
                NumberOfLeaves = param.NumberOfLeaves,
                NumberOfIterations = param.NumberOfTrees,
                MinimumExampleCountPerLeaf = param.MinimumExampleCountPerLeaf,
                LearningRate = param.LearningRate,
                NumberOfThreads = AutoMlUtils.GetNumberOfThreadFromEnvrionment(),
                LabelColumnName = param.LabelColumnName,
                FeatureColumnName = param.FeatureColumnName,
                ExampleWeightColumnName = param.ExampleWeightColumnName,
                Booster = new GradientBooster.Options()
                {
                    SubsampleFraction = param.SubsampleFraction,
                    FeatureFraction = param.FeatureFraction,
                    L1Regularization = param.L1Regularization,
                    L2Regularization = param.L2Regularization,
                },
                MaximumBinCountPerFeature = param.MaximumBinCountPerFeature,
            };

            return context.Regression.Trainers.LightGbm(option);
        }
    }
}
