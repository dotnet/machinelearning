// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Trainers.FastTree;

namespace Microsoft.ML.AutoML.CodeGen
{
    internal partial class FastTreeOva
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, FastTreeOption param)
        {
            var option = new FastTreeBinaryTrainer.Options()
            {
                NumberOfLeaves = param.NumberOfLeaves,
                NumberOfTrees = param.NumberOfTrees,
                MinimumExampleCountPerLeaf = param.MinimumExampleCountPerLeaf,
                LearningRate = param.LearningRate,
                LabelColumnName = param.LabelColumnName,
                FeatureColumnName = param.FeatureColumnName,
                ExampleWeightColumnName = param.ExampleWeightColumnName,
                NumberOfThreads = AutoMlUtils.GetNumberOfThreadFromEnvrionment(),
                MaximumBinCountPerFeature = param.MaximumBinCountPerFeature,
                FeatureFraction = param.FeatureFraction,
            };

            return context.MulticlassClassification.Trainers.OneVersusAll(context.BinaryClassification.Trainers.FastTree(option), labelColumnName: param.LabelColumnName);
        }
    }

    internal partial class FastTreeRegression
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, FastTreeOption param)
        {
            var option = new FastTreeRegressionTrainer.Options()
            {
                NumberOfLeaves = param.NumberOfLeaves,
                NumberOfTrees = param.NumberOfTrees,
                MinimumExampleCountPerLeaf = param.MinimumExampleCountPerLeaf,
                LearningRate = param.LearningRate,
                LabelColumnName = param.LabelColumnName,
                FeatureColumnName = param.FeatureColumnName,
                ExampleWeightColumnName = param.ExampleWeightColumnName,
                NumberOfThreads = AutoMlUtils.GetNumberOfThreadFromEnvrionment(),
                MaximumBinCountPerFeature = param.MaximumBinCountPerFeature,
                FeatureFraction = param.FeatureFraction,
            };

            return context.Regression.Trainers.FastTree(option);
        }
    }

    internal partial class FastTreeTweedieRegression
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, FastTreeOption param)
        {
            var option = new FastTreeTweedieTrainer.Options()
            {
                NumberOfLeaves = param.NumberOfLeaves,
                NumberOfTrees = param.NumberOfTrees,
                MinimumExampleCountPerLeaf = param.MinimumExampleCountPerLeaf,
                LearningRate = param.LearningRate,
                LabelColumnName = param.LabelColumnName,
                FeatureColumnName = param.FeatureColumnName,
                ExampleWeightColumnName = param.ExampleWeightColumnName,
                NumberOfThreads = AutoMlUtils.GetNumberOfThreadFromEnvrionment(),
                MaximumBinCountPerFeature = param.MaximumBinCountPerFeature,
                FeatureFraction = param.FeatureFraction,
            };

            return context.Regression.Trainers.FastTreeTweedie(option);
        }
    }

    internal partial class FastTreeBinary
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, FastTreeOption param)
        {
            var option = new FastTreeBinaryTrainer.Options()
            {
                NumberOfLeaves = param.NumberOfLeaves,
                NumberOfTrees = param.NumberOfTrees,
                MinimumExampleCountPerLeaf = param.MinimumExampleCountPerLeaf,
                LearningRate = param.LearningRate,
                LabelColumnName = param.LabelColumnName,
                FeatureColumnName = param.FeatureColumnName,
                ExampleWeightColumnName = param.ExampleWeightColumnName,
                NumberOfThreads = AutoMlUtils.GetNumberOfThreadFromEnvrionment(),
                MaximumBinCountPerFeature = param.MaximumBinCountPerFeature,
                FeatureFraction = param.FeatureFraction,
            };

            return context.BinaryClassification.Trainers.FastTree(option);
        }
    }
}
