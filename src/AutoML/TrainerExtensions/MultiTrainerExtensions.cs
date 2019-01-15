// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Learners;
using Microsoft.ML.LightGBM;
using Microsoft.ML.Trainers;
using Microsoft.ML.Training;

namespace Microsoft.ML.Auto
{
    using ITrainerEstimator = ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictor>, IPredictor>;
    using ITrainerEstimatorProducingFloat = ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictorProducing<float>>, IPredictorProducing<float>>;

    internal class AveragedPerceptronOvaExtension : ITrainerExtension
    {
        private static readonly ITrainerExtension _binaryLearnerCatalogItem = new AveragedPerceptronBinaryExtension();

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildAveragePerceptronParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var binaryTrainer = _binaryLearnerCatalogItem.CreateInstance(mlContext, sweepParams) as ITrainerEstimatorProducingFloat;
            return mlContext.MulticlassClassification.Trainers.OneVersusAll(binaryTrainer);
        }

        public TrainerName GetTrainerName()
        {
            return TrainerName.AveragedPerceptronOva;
        }
    }

    internal class FastForestOvaExtension : ITrainerExtension
    {
        private static readonly ITrainerExtension _binaryLearnerCatalogItem = new FastForestBinaryExtension();

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildFastForestParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var binaryTrainer = _binaryLearnerCatalogItem.CreateInstance(mlContext, sweepParams) as ITrainerEstimatorProducingFloat;
            return mlContext.MulticlassClassification.Trainers.OneVersusAll(binaryTrainer);
        }

        public TrainerName GetTrainerName()
        {
            return TrainerName.FastForestOva;
        }
    }

    internal class LightGbmMultiExtension : ITrainerExtension
    {
        private static readonly ITrainerExtension _binaryLearnerCatalogItem = new LightGbmBinaryExtension();

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildLightGbmParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            Action<LightGbmArguments> argsFunc = TrainerExtensionUtil.CreateLightGbmArgsFunc(sweepParams);
            return mlContext.MulticlassClassification.Trainers.LightGbm(advancedSettings: argsFunc);
        }

        public TrainerName GetTrainerName()
        {
            return TrainerName.LightGbmMulti;
        }
    }

    internal class LinearSvmOvaExtension : ITrainerExtension
    {
        private static readonly ITrainerExtension _binaryLearnerCatalogItem = new LinearSvmBinaryExtension();

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildLinearSvmParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var binaryTrainer = _binaryLearnerCatalogItem.CreateInstance(mlContext, sweepParams) as ITrainerEstimatorProducingFloat;
            return mlContext.MulticlassClassification.Trainers.OneVersusAll(binaryTrainer);
        }

        public TrainerName GetTrainerName()
        {
            return TrainerName.LinearSvmOva;
        }
    }

    internal class SdcaMultiExtension : ITrainerExtension
    {
        private static readonly ITrainerExtension _binaryLearnerCatalogItem = new SdcaBinaryExtension();

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildSdcaParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var argsFunc = TrainerExtensionUtil.CreateArgsFunc<SdcaMultiClassTrainer.Arguments>(sweepParams);
            return mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(advancedSettings: argsFunc);
        }

        public TrainerName GetTrainerName()
        {
            return TrainerName.SdcaMulti;
        }
    }


    internal class LogisticRegressionOvaExtension : ITrainerExtension
    {
        private static readonly ITrainerExtension _binaryLearnerCatalogItem = new LogisticRegressionBinaryExtension();

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildLogisticRegressionParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var binaryTrainer = _binaryLearnerCatalogItem.CreateInstance(mlContext, sweepParams) as ITrainerEstimatorProducingFloat;
            return mlContext.MulticlassClassification.Trainers.OneVersusAll(binaryTrainer);
        }

        public TrainerName GetTrainerName()
        {
            return TrainerName.LogisticRegressionOva;
        }
    }

    internal class SgdOvaExtension : ITrainerExtension
    {
        private static readonly ITrainerExtension _binaryLearnerCatalogItem = new SgdBinaryExtension();

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildSgdParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var binaryTrainer = _binaryLearnerCatalogItem.CreateInstance(mlContext, sweepParams) as ITrainerEstimatorProducingFloat;
            return mlContext.MulticlassClassification.Trainers.OneVersusAll(binaryTrainer);
        }

        public TrainerName GetTrainerName()
        {
            return TrainerName.StochasticGradientDescentOva;
        }
    }

    internal class SymSgdOvaExtension : ITrainerExtension
    {
        private static readonly ITrainerExtension _binaryLearnerCatalogItem = new SymSgdBinaryExtension();

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return _binaryLearnerCatalogItem.GetHyperparamSweepRanges();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var binaryTrainer = _binaryLearnerCatalogItem.CreateInstance(mlContext, sweepParams) as ITrainerEstimatorProducingFloat;
            return mlContext.MulticlassClassification.Trainers.OneVersusAll(binaryTrainer);
        }

        public TrainerName GetTrainerName()
        {
            return TrainerName.SymSgdOva;
        }
    }

    internal class FastTreeOvaExtension : ITrainerExtension
    {
        private static readonly ITrainerExtension _binaryLearnerCatalogItem = new FastTreeBinaryExtension();

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildFastTreeParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var binaryTrainer = _binaryLearnerCatalogItem.CreateInstance(mlContext, sweepParams) as ITrainerEstimatorProducingFloat;
            return mlContext.MulticlassClassification.Trainers.OneVersusAll(binaryTrainer);
        }

        public TrainerName GetTrainerName()
        {
            return TrainerName.FastTreeOva;
        }
    }

    internal class LogisticRegressionMultiExtension : ITrainerExtension
    {
        private static readonly ITrainerExtension _binaryLearnerCatalogItem = new LogisticRegressionBinaryExtension();

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildLogisticRegressionParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var argsFunc = TrainerExtensionUtil.CreateArgsFunc<MulticlassLogisticRegression.Arguments>(sweepParams);
            return mlContext.MulticlassClassification.Trainers.LogisticRegression(advancedSettings: argsFunc);
        }

        public TrainerName GetTrainerName()
        {
            return TrainerName.LogisticRegressionMulti;
        }
    }
}