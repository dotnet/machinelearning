// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.HalLearners;
using Microsoft.ML.Trainers.Online;
using Microsoft.ML.Training;

namespace Microsoft.ML.Auto
{
    using ITrainerEstimator = ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictor>, IPredictor>;

    internal class FastForestRegressionExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildFastForestParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var argsFunc = TrainerExtensionUtil.CreateArgsFunc<FastForestRegression.Arguments>(sweepParams);
            return mlContext.Regression.Trainers.FastForest(advancedSettings: argsFunc);
        }

        public TrainerName GetTrainerName()
        {
            return TrainerName.FastForestRegression;
        }
    }

    internal class FastTreeRegressionExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildFastTreeParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var argsFunc = TrainerExtensionUtil.CreateArgsFunc<FastTreeRegressionTrainer.Arguments>(sweepParams);
            return mlContext.Regression.Trainers.FastTree(advancedSettings: argsFunc);
        }

        public TrainerName GetTrainerName()
        {
            return TrainerName.FastTreeRegression;
        }
    }

    internal class FastTreeTweedieRegressionExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildFastTreeTweedieParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var argsFunc = TrainerExtensionUtil.CreateArgsFunc<FastTreeTweedieTrainer.Arguments>(sweepParams);
            return mlContext.Regression.Trainers.FastTreeTweedie(advancedSettings: argsFunc);
        }

        public TrainerName GetTrainerName()
        {
            return TrainerName.FastTreeTweedieRegression;
        }
    }

    internal class LightGbmRegressionExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildLightGbmParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var argsFunc = TrainerExtensionUtil.CreateLightGbmArgsFunc(sweepParams);
            return mlContext.Regression.Trainers.LightGbm(advancedSettings: argsFunc);
        }

        public TrainerName GetTrainerName()
        {
            return TrainerName.LightGbmRegression;
        }
    }

    internal class OnlineGradientDescentRegressionExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildOnlineGradientDescentParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var argsFunc = TrainerExtensionUtil.CreateArgsFunc<AveragedLinearArguments>(sweepParams);
            return mlContext.Regression.Trainers.OnlineGradientDescent(advancedSettings: argsFunc);
        }

        public TrainerName GetTrainerName()
        {
            return TrainerName.OnlineGradientDescentRegression;
        }
    }

    internal class OrdinaryLeastSquaresRegressionExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildOrdinaryLeastSquaresParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var argsFunc = TrainerExtensionUtil.CreateArgsFunc<OlsLinearRegressionTrainer.Arguments>(sweepParams);
            return mlContext.Regression.Trainers.OrdinaryLeastSquares(advancedSettings: argsFunc);
        }

        public TrainerName GetTrainerName()
        {
            return TrainerName.OrdinaryLeastSquaresRegression;
        }
    }

    internal class PoissonRegressionExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildPoissonRegressionParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var argsFunc = TrainerExtensionUtil.CreateArgsFunc<PoissonRegression.Arguments>(sweepParams);
            return mlContext.Regression.Trainers.PoissonRegression(advancedSettings: argsFunc);
        }

        public TrainerName GetTrainerName()
        {
            return TrainerName.PoissonRegression;
        }
    }

    internal class SdcaRegressionExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildSdcaParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams)
        {
            var argsFunc = TrainerExtensionUtil.CreateArgsFunc<SdcaRegressionTrainer.Arguments>(sweepParams);
            return mlContext.Regression.Trainers.StochasticDualCoordinateAscent(advancedSettings: argsFunc);
        }

        public TrainerName GetTrainerName()
        {
            return TrainerName.SdcaRegression;
        }
    }
}