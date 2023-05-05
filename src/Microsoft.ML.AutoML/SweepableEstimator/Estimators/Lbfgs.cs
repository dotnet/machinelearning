// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Trainers;

namespace Microsoft.ML.AutoML.CodeGen
{
    internal partial class LbfgsMaximumEntropyMulti
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, LbfgsOption param)
        {
            var option = new LbfgsMaximumEntropyMulticlassTrainer.Options()
            {
                L1Regularization = param.L1Regularization,
                L2Regularization = param.L2Regularization,
                LabelColumnName = param.LabelColumnName,
                FeatureColumnName = param.FeatureColumnName,
                ExampleWeightColumnName = param.ExampleWeightColumnName,
                NumberOfThreads = AutoMlUtils.GetNumberOfThreadFromEnvrionment(),
            };

            return context.MulticlassClassification.Trainers.LbfgsMaximumEntropy(option);
        }
    }

    internal partial class LbfgsPoissonRegressionRegression
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, LbfgsOption param)
        {
            var option = new LbfgsPoissonRegressionTrainer.Options()
            {
                L1Regularization = param.L1Regularization,
                L2Regularization = param.L2Regularization,
                LabelColumnName = param.LabelColumnName,
                FeatureColumnName = param.FeatureColumnName,
                ExampleWeightColumnName = param.ExampleWeightColumnName,
                NumberOfThreads = AutoMlUtils.GetNumberOfThreadFromEnvrionment(),
            };

            return context.Regression.Trainers.LbfgsPoissonRegression(option);
        }
    }

    internal partial class LbfgsLogisticRegressionBinary
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, LbfgsOption param)
        {
            var option = new LbfgsLogisticRegressionBinaryTrainer.Options()
            {
                L1Regularization = param.L1Regularization,
                L2Regularization = param.L2Regularization,
                LabelColumnName = param.LabelColumnName,
                FeatureColumnName = param.FeatureColumnName,
                ExampleWeightColumnName = param.ExampleWeightColumnName,
                NumberOfThreads = AutoMlUtils.GetNumberOfThreadFromEnvrionment(),
            };

            return context.BinaryClassification.Trainers.LbfgsLogisticRegression(option);
        }
    }

    internal partial class LbfgsLogisticRegressionOva
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, LbfgsOption param)
        {
            var option = new LbfgsLogisticRegressionBinaryTrainer.Options()
            {
                L1Regularization = param.L1Regularization,
                L2Regularization = param.L2Regularization,
                LabelColumnName = param.LabelColumnName,
                FeatureColumnName = param.FeatureColumnName,
                ExampleWeightColumnName = param.ExampleWeightColumnName,
                NumberOfThreads = AutoMlUtils.GetNumberOfThreadFromEnvrionment(),
            };

            var binaryTrainer = context.BinaryClassification.Trainers.LbfgsLogisticRegression(option);
            return context.MulticlassClassification.Trainers.OneVersusAll(binaryTrainer, param.LabelColumnName);
        }
    }
}
