// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.HalLearners;
using Microsoft.ML.Trainers.Online;

namespace Microsoft.ML.Auto
{
    using ITrainerEstimator = ITrainerEstimator<ISingleFeaturePredictionTransformer<object>, object>;

    internal class FastForestRegressionExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildFastForestParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo)
        {
            var options = TrainerExtensionUtil.CreateOptions<FastForestRegression.Options>(sweepParams, columnInfo.LabelColumn);
            options.WeightColumn = columnInfo.WeightColumn;
            return mlContext.Regression.Trainers.FastForest(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumn, columnInfo.WeightColumn);
        }
    }

    internal class FastTreeRegressionExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildFastTreeParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo)
        {
            var options = TrainerExtensionUtil.CreateOptions<FastTreeRegressionTrainer.Options>(sweepParams, columnInfo.LabelColumn);
            options.WeightColumn = columnInfo.WeightColumn;
            return mlContext.Regression.Trainers.FastTree(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumn, columnInfo.WeightColumn);
        }
    }

    internal class FastTreeTweedieRegressionExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildFastTreeTweedieParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo)
        {
            var options = TrainerExtensionUtil.CreateOptions<FastTreeTweedieTrainer.Options>(sweepParams, columnInfo.LabelColumn);
            options.WeightColumn = columnInfo.WeightColumn;
            return mlContext.Regression.Trainers.FastTreeTweedie(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumn, columnInfo.WeightColumn);
        }
    }

    internal class LightGbmRegressionExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildLightGbmParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo)
        {
            var options = TrainerExtensionUtil.CreateLightGbmOptions(sweepParams, columnInfo);
            options.WeightColumn = columnInfo.WeightColumn;
            return mlContext.Regression.Trainers.LightGbm(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildLightGbmPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumn, columnInfo.WeightColumn);
        }
    }

    internal class OnlineGradientDescentRegressionExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildOnlineGradientDescentParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo)
        {
            var options = TrainerExtensionUtil.CreateOptions<OnlineGradientDescentTrainer.Options>(sweepParams, columnInfo.LabelColumn);
            return mlContext.Regression.Trainers.OnlineGradientDescent(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumn);
        }
    }

    internal class OrdinaryLeastSquaresRegressionExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildOrdinaryLeastSquaresParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo)
        {
            var options = TrainerExtensionUtil.CreateOptions<OlsLinearRegressionTrainer.Options>(sweepParams, columnInfo.LabelColumn);
            options.WeightColumn = columnInfo.WeightColumn;
            return mlContext.Regression.Trainers.OrdinaryLeastSquares(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumn, columnInfo.WeightColumn);
        }
    }

    internal class PoissonRegressionExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildPoissonRegressionParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo)
        {
            var options = TrainerExtensionUtil.CreateOptions<PoissonRegression.Options>(sweepParams, columnInfo.LabelColumn);
            options.WeightColumn = columnInfo.WeightColumn;
            return mlContext.Regression.Trainers.PoissonRegression(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumn, columnInfo.WeightColumn);
        }
    }

    internal class SdcaRegressionExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildSdcaParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo)
        {
            var options = TrainerExtensionUtil.CreateOptions<SdcaRegressionTrainer.Options>(sweepParams, columnInfo.LabelColumn);
            return mlContext.Regression.Trainers.StochasticDualCoordinateAscent(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumn);
        }
    }
}