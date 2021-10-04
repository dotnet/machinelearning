// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;

namespace Microsoft.ML.AutoML
{
    using ITrainerEstimator = ITrainerEstimator<IPredictionTransformer<object>, object>;

    internal class FastForestRegressionExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildFastForestParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo, IDataView validationSet)
        {
            var options = TrainerExtensionUtil.CreateOptions<FastForestRegressionTrainer.Options>(sweepParams, columnInfo.LabelColumnName);
            options.ExampleWeightColumnName = columnInfo.ExampleWeightColumnName;
            return mlContext.Regression.Trainers.FastForest(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumnName, columnInfo.ExampleWeightColumnName);
        }
    }

    internal class FastTreeRegressionExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildFastTreeParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo, IDataView validationSet)
        {
            var options = TrainerExtensionUtil.CreateOptions<FastTreeRegressionTrainer.Options>(sweepParams, columnInfo.LabelColumnName);
            options.ExampleWeightColumnName = columnInfo.ExampleWeightColumnName;
            return mlContext.Regression.Trainers.FastTree(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumnName, columnInfo.ExampleWeightColumnName);
        }
    }

    internal class FastTreeTweedieRegressionExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildFastTreeTweedieParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo, IDataView validationSet)
        {
            var options = TrainerExtensionUtil.CreateOptions<FastTreeTweedieTrainer.Options>(sweepParams, columnInfo.LabelColumnName);
            options.ExampleWeightColumnName = columnInfo.ExampleWeightColumnName;
            return mlContext.Regression.Trainers.FastTreeTweedie(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumnName, columnInfo.ExampleWeightColumnName);
        }
    }

    internal class LightGbmRegressionExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildLightGbmParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo, IDataView validationSet)
        {
            LightGbmRegressionTrainer.Options options = TrainerExtensionUtil.CreateLightGbmOptions<LightGbmRegressionTrainer.Options, float, RegressionPredictionTransformer<LightGbmRegressionModelParameters>, LightGbmRegressionModelParameters>(sweepParams, columnInfo);
            return mlContext.Regression.Trainers.LightGbm(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildLightGbmPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumnName, columnInfo.ExampleWeightColumnName, columnInfo.GroupIdColumnName);
        }
    }

    internal class OnlineGradientDescentRegressionExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildOnlineGradientDescentParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo, IDataView validationSet)
        {
            var options = TrainerExtensionUtil.CreateOptions<OnlineGradientDescentTrainer.Options>(sweepParams, columnInfo.LabelColumnName);
            return mlContext.Regression.Trainers.OnlineGradientDescent(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumnName);
        }
    }

    internal class OlsRegressionExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildOlsParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo, IDataView validationSet)
        {
            var options = TrainerExtensionUtil.CreateOptions<OlsTrainer.Options>(sweepParams, columnInfo.LabelColumnName);
            options.ExampleWeightColumnName = columnInfo.ExampleWeightColumnName;
            return mlContext.Regression.Trainers.Ols(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumnName, columnInfo.ExampleWeightColumnName);
        }
    }

    internal class LbfgsPoissonRegressionExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildLbfgsPoissonRegressionParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo, IDataView validationSet)
        {
            var options = TrainerExtensionUtil.CreateOptions<LbfgsPoissonRegressionTrainer.Options>(sweepParams, columnInfo.LabelColumnName);
            options.ExampleWeightColumnName = columnInfo.ExampleWeightColumnName;
            return mlContext.Regression.Trainers.LbfgsPoissonRegression(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumnName, columnInfo.ExampleWeightColumnName);
        }
    }

    internal class SdcaRegressionExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildSdcaParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo, IDataView validationSet)
        {
            var options = TrainerExtensionUtil.CreateOptions<SdcaRegressionTrainer.Options>(sweepParams, columnInfo.LabelColumnName);
            return mlContext.Regression.Trainers.Sdca(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumnName);
        }
    }
}
