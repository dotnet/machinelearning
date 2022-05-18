// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Trainers;

namespace Microsoft.ML.AutoML.CodeGen
{
    internal partial class SdcaRegression
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, SdcaOption param)
        {
            var option = new SdcaRegressionTrainer.Options()
            {
                LabelColumnName = param.LabelColumnName,
                FeatureColumnName = param.FeatureColumnName,
                ExampleWeightColumnName = param.ExampleWeightColumnName,
                L1Regularization = param.L1Regularization,
                L2Regularization = param.L2Regularization,
                NumberOfThreads = AutoMlUtils.GetNumberOfThreadFromEnvrionment(),
            };

            return context.Regression.Trainers.Sdca(option);
        }
    }

    internal partial class SdcaMaximumEntropyMulti
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, SdcaOption param)
        {
            var option = new SdcaMaximumEntropyMulticlassTrainer.Options()
            {
                LabelColumnName = param.LabelColumnName,
                FeatureColumnName = param.FeatureColumnName,
                ExampleWeightColumnName = param.ExampleWeightColumnName,
                L1Regularization = param.L1Regularization,
                L2Regularization = param.L2Regularization,
                NumberOfThreads = AutoMlUtils.GetNumberOfThreadFromEnvrionment(),
            };

            return context.MulticlassClassification.Trainers.SdcaMaximumEntropy(option);
        }
    }

    internal partial class SdcaLogisticRegressionBinary
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, SdcaOption param)
        {
            var option = new SdcaLogisticRegressionBinaryTrainer.Options()
            {
                LabelColumnName = param.LabelColumnName,
                FeatureColumnName = param.FeatureColumnName,
                ExampleWeightColumnName = param.ExampleWeightColumnName,
                L1Regularization = param.L1Regularization,
                L2Regularization = param.L2Regularization,
                NumberOfThreads = AutoMlUtils.GetNumberOfThreadFromEnvrionment(),
            };

            return context.BinaryClassification.Trainers.SdcaLogisticRegression(option);
        }
    }

    internal partial class SdcaLogisticRegressionOva
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, SdcaOption param)
        {
            var option = new SdcaLogisticRegressionBinaryTrainer.Options()
            {
                LabelColumnName = param.LabelColumnName,
                FeatureColumnName = param.FeatureColumnName,
                ExampleWeightColumnName = param.ExampleWeightColumnName,
                L1Regularization = param.L1Regularization,
                L2Regularization = param.L2Regularization,
                NumberOfThreads = AutoMlUtils.GetNumberOfThreadFromEnvrionment(),
            };

            var binaryTrainer = context.BinaryClassification.Trainers.SdcaLogisticRegression(option);
            return context.MulticlassClassification.Trainers.OneVersusAll(binaryEstimator: binaryTrainer, labelColumnName: param.LabelColumnName);
        }
    }
}
