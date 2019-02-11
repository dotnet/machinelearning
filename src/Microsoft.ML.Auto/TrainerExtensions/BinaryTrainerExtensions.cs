// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Learners;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.Online;
using Microsoft.ML.Trainers.SymSgd;
using Microsoft.ML.Training;

namespace Microsoft.ML.Auto
{
    using ITrainerEstimator = ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictor>, IPredictor>;

    internal class AveragedPerceptronBinaryExtension : ITrainerExtension
    {
        private const int DefaultNumIterations = 10;

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildAveragePerceptronParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var options = new AveragedPerceptronTrainer.Options();
            if (sweepParams == null)
            {
                options.NumIterations = DefaultNumIterations;
            }
            else
            {
                options = TrainerExtensionUtil.CreateOptions<AveragedPerceptronTrainer.Options>(sweepParams);
            }
            return mlContext.BinaryClassification.Trainers.AveragedPerceptron(options);
        }
    }

    internal class FastForestBinaryExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildFastForestParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var options = TrainerExtensionUtil.CreateOptions<FastForestClassification.Options>(sweepParams);
            return mlContext.BinaryClassification.Trainers.FastForest(options);
        }
    }

    internal class FastTreeBinaryExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildFastTreeParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var options = TrainerExtensionUtil.CreateOptions<FastTreeBinaryClassificationTrainer.Options>(sweepParams);
            return mlContext.BinaryClassification.Trainers.FastTree(options);
        }
    }

    internal class LightGbmBinaryExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildLightGbmParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var options = TrainerExtensionUtil.CreateLightGbmOptions(sweepParams);
            return mlContext.BinaryClassification.Trainers.LightGbm(options);
        }
    }

    internal class LinearSvmBinaryExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildLinearSvmParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var options = TrainerExtensionUtil.CreateOptions<LinearSvmTrainer.Options>(sweepParams);
            return mlContext.BinaryClassification.Trainers.LinearSupportVectorMachines(options);
        }
    }

    internal class SdcaBinaryExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildSdcaParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var options = TrainerExtensionUtil.CreateOptions<SdcaBinaryTrainer.Options>(sweepParams);
            return mlContext.BinaryClassification.Trainers.StochasticDualCoordinateAscent(options);
        }
    }

    internal class LogisticRegressionBinaryExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildLogisticRegressionParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var options = TrainerExtensionUtil.CreateOptions<LogisticRegression.Options>(sweepParams);
            return mlContext.BinaryClassification.Trainers.LogisticRegression(options);
        }
    }

    internal class SgdBinaryExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildSgdParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var options = TrainerExtensionUtil.CreateOptions<StochasticGradientDescentClassificationTrainer.Options>(sweepParams);
            return mlContext.BinaryClassification.Trainers.StochasticGradientDescent(options);
        }
    }

    internal class SymSgdBinaryExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildSymSgdParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var options = TrainerExtensionUtil.CreateOptions<SymSgdClassificationTrainer.Options>(sweepParams);
            return mlContext.BinaryClassification.Trainers.SymbolicStochasticGradientDescent(options);
        }
    }
}