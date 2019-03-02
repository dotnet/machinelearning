// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.HalLearners;
using Microsoft.ML.Trainers.Online;

namespace Microsoft.ML.Auto
{
    using ITrainerEstimator = ITrainerEstimator<ISingleFeaturePredictionTransformer<object>, object>;

    internal class AveragedPerceptronBinaryExtension : ITrainerExtension
    {
        private const int DefaultNumIterations = 10;

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildAveragePerceptronParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo)
        {
            AveragedPerceptronTrainer.Options options = null;
            if (sweepParams == null || !sweepParams.Any())
            {
                options = new AveragedPerceptronTrainer.Options();
                options.NumberOfIterations = DefaultNumIterations;
                options.LabelColumn = columnInfo.LabelColumn;
            }
            else
            {
                options = TrainerExtensionUtil.CreateOptions<AveragedPerceptronTrainer.Options>(sweepParams, columnInfo.LabelColumn);
                if (!sweepParams.Any(p => p.Name == "NumberOfIterations"))
                {
                    options.NumberOfIterations = DefaultNumIterations;
                }
            }
            return mlContext.BinaryClassification.Trainers.AveragedPerceptron(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            Dictionary<string, object> additionalProperties = null;

            if (sweepParams == null || !sweepParams.Any(p => p.Name != "NumberOfIterations"))
            {
                additionalProperties = new Dictionary<string, object>()
                {
                    { "NumberOfIterations", DefaultNumIterations.ToString() }
                };
            }

            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumn, additionalProperties: additionalProperties);
        }
    }

    internal class FastForestBinaryExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildFastForestParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo)
        {
            var options = TrainerExtensionUtil.CreateOptions<FastForestClassification.Options>(sweepParams, columnInfo.LabelColumn);
            options.WeightColumn = columnInfo.WeightColumn;
            return mlContext.BinaryClassification.Trainers.FastForest(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumn, columnInfo.WeightColumn);
        }
    }

    internal class FastTreeBinaryExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildFastTreeParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo)
        {
            var options = TrainerExtensionUtil.CreateOptions<FastTreeBinaryClassificationTrainer.Options>(sweepParams, columnInfo.LabelColumn);
            options.WeightColumn = columnInfo.WeightColumn;
            return mlContext.BinaryClassification.Trainers.FastTree(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumn, columnInfo.WeightColumn);
        }
    }

    internal class LightGbmBinaryExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildLightGbmParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo)
        {
            var options = TrainerExtensionUtil.CreateLightGbmOptions(sweepParams, columnInfo);
            return mlContext.BinaryClassification.Trainers.LightGbm(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildLightGbmPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumn, columnInfo.WeightColumn);
        }
    }

    internal class LinearSvmBinaryExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildLinearSvmParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo)
        {
            var options = TrainerExtensionUtil.CreateOptions<LinearSvmTrainer.Options>(sweepParams, columnInfo.LabelColumn);
            return mlContext.BinaryClassification.Trainers.LinearSupportVectorMachines(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumn);
        }
    }

    internal class SdcaBinaryExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildSdcaParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo)
        {
            var options = TrainerExtensionUtil.CreateOptions<SdcaBinaryTrainer.Options>(sweepParams, columnInfo.LabelColumn);
            return mlContext.BinaryClassification.Trainers.StochasticDualCoordinateAscent(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumn);
        }
    }

    internal class LogisticRegressionBinaryExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildLogisticRegressionParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo)
        {
            var options = TrainerExtensionUtil.CreateOptions<LogisticRegression.Options>(sweepParams, columnInfo.LabelColumn);
            options.WeightColumn = columnInfo.WeightColumn;
            return mlContext.BinaryClassification.Trainers.LogisticRegression(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumn, columnInfo.WeightColumn);
        }
    }

    internal class SgdBinaryExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildSgdParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo)
        {
            var options = TrainerExtensionUtil.CreateOptions<SgdBinaryTrainer.Options>(sweepParams, columnInfo.LabelColumn);
            options.WeightColumn = columnInfo.WeightColumn;
            return mlContext.BinaryClassification.Trainers.StochasticGradientDescent(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                 columnInfo.LabelColumn, columnInfo.WeightColumn);
        }
    }

    internal class SymSgdBinaryExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildSymSgdParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo)
        {
            var options = TrainerExtensionUtil.CreateOptions<SymSgdClassificationTrainer.Options>(sweepParams, columnInfo.LabelColumn);
            return mlContext.BinaryClassification.Trainers.SymbolicStochasticGradientDescent(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumn);
        }
    }
}
