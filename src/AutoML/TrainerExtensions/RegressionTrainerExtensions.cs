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
            var options = TrainerExtensionUtil.CreateOptions<FastForestRegression.Options>(sweepParams);
            return mlContext.Regression.Trainers.FastForest(options);
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
            var options = TrainerExtensionUtil.CreateOptions<FastTreeRegressionTrainer.Options>(sweepParams);
            return mlContext.Regression.Trainers.FastTree(options);
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
            var options = TrainerExtensionUtil.CreateOptions<FastTreeTweedieTrainer.Options>(sweepParams);
            return mlContext.Regression.Trainers.FastTreeTweedie(options);
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
            var options = TrainerExtensionUtil.CreateLightGbmOptions(sweepParams);
            return mlContext.Regression.Trainers.LightGbm(options);
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
            var options = TrainerExtensionUtil.CreateOptions<OnlineGradientDescentTrainer.Options>(sweepParams);
            return mlContext.Regression.Trainers.OnlineGradientDescent(options);
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
            var options = TrainerExtensionUtil.CreateOptions<OlsLinearRegressionTrainer.Options>(sweepParams);
            return mlContext.Regression.Trainers.OrdinaryLeastSquares(options);
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
            var options = TrainerExtensionUtil.CreateOptions<PoissonRegression.Options>(sweepParams);
            return mlContext.Regression.Trainers.PoissonRegression(options);
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
            var options = TrainerExtensionUtil.CreateOptions<SdcaRegressionTrainer.Options>(sweepParams);
            return mlContext.Regression.Trainers.StochasticDualCoordinateAscent(options);
        }
    }
}