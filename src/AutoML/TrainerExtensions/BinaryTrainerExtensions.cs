// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Training;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.Online;
using Microsoft.ML.Trainers.SymSgd;
using System;
using System.Collections.Generic;
using Microsoft.ML.LightGBM;
using Microsoft.ML.Learners;

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
            Action<AveragedPerceptronTrainer.Arguments> argsFunc = null;
            if (sweepParams == null)
            {
                argsFunc = (args) =>
                {
                    args.NumIterations = DefaultNumIterations;
                };
            }
            else
            {
                argsFunc = TrainerExtensionUtil.CreateArgsFunc<AveragedPerceptronTrainer.Arguments>(sweepParams);
            }
            return mlContext.BinaryClassification.Trainers.AveragedPerceptron(advancedSettings: argsFunc);
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
            var argsFunc = TrainerExtensionUtil.CreateArgsFunc<FastForestClassification.Arguments>(sweepParams);
            return mlContext.BinaryClassification.Trainers.FastForest(advancedSettings: argsFunc);
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
            var argsFunc = TrainerExtensionUtil.CreateArgsFunc<FastTreeBinaryClassificationTrainer.Arguments>(sweepParams);
            return mlContext.BinaryClassification.Trainers.FastTree(advancedSettings: argsFunc);
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
            Action<LightGbmArguments> argsFunc = TrainerExtensionUtil.CreateLightGbmArgsFunc(sweepParams);
            return mlContext.BinaryClassification.Trainers.LightGbm(advancedSettings: argsFunc);
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
            var argsFunc = TrainerExtensionUtil.CreateArgsFunc<LinearSvmTrainer.Arguments>(sweepParams);
            return mlContext.BinaryClassification.Trainers.LinearSupportVectorMachines(advancedSettings: argsFunc);
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
            var argsFunc = TrainerExtensionUtil.CreateArgsFunc<SdcaBinaryTrainer.Arguments>(sweepParams);
            return mlContext.BinaryClassification.Trainers.StochasticDualCoordinateAscent(advancedSettings: argsFunc);
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
            var argsFunc = TrainerExtensionUtil.CreateArgsFunc<LogisticRegression.Arguments>(sweepParams);
            return mlContext.BinaryClassification.Trainers.LogisticRegression(advancedSettings: argsFunc);
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
            var argsFunc = TrainerExtensionUtil.CreateArgsFunc<StochasticGradientDescentClassificationTrainer.Arguments>(sweepParams);
            return mlContext.BinaryClassification.Trainers.StochasticGradientDescent(advancedSettings: argsFunc);
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
            var argsFunc = TrainerExtensionUtil.CreateArgsFunc<SymSgdClassificationTrainer.Arguments>(sweepParams);
            return mlContext.BinaryClassification.Trainers.SymbolicStochasticGradientDescent(advancedSettings: argsFunc);
        }
    }
}