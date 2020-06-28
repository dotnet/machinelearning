// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;

namespace Microsoft.ML.AutoML
{
    using ITrainerEstimator = ITrainerEstimator<IPredictionTransformer<object>, object>;

    internal class AveragedPerceptronBinaryExtension : ITrainerExtension
    {
        private const int DefaultNumIterations = 10;

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildAveragePerceptronParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo, IDataView validationSet)
        {
            AveragedPerceptronTrainer.Options options = null;
            if (sweepParams == null || !sweepParams.Any())
            {
                options = new AveragedPerceptronTrainer.Options();
                options.NumberOfIterations = DefaultNumIterations;
                options.LabelColumnName = columnInfo.LabelColumnName;
            }
            else
            {
                options = TrainerExtensionUtil.CreateOptions<AveragedPerceptronTrainer.Options>(sweepParams, columnInfo.LabelColumnName);
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
                    { "NumberOfIterations", DefaultNumIterations }
                };
            }

            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumnName, additionalProperties: additionalProperties);
        }
    }

    internal class FastForestBinaryExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildFastForestParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo, IDataView validationSet)
        {
            var options = TrainerExtensionUtil.CreateOptions<FastForestBinaryTrainer.Options>(sweepParams, columnInfo.LabelColumnName);
            options.ExampleWeightColumnName = columnInfo.ExampleWeightColumnName;
            return mlContext.BinaryClassification.Trainers.FastForest(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumnName, columnInfo.ExampleWeightColumnName);
        }
    }

    internal class FastTreeBinaryExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildFastTreeParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo, IDataView validationSet)
        {
            var options = TrainerExtensionUtil.CreateOptions<FastTreeBinaryTrainer.Options>(sweepParams, columnInfo.LabelColumnName);
            options.ExampleWeightColumnName = columnInfo.ExampleWeightColumnName;
            return mlContext.BinaryClassification.Trainers.FastTree(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumnName, columnInfo.ExampleWeightColumnName);
        }
    }

    internal class LightGbmBinaryExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildLightGbmParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo, IDataView validationSet)
        {
            LightGbmBinaryTrainer.Options options = TrainerExtensionUtil.CreateLightGbmOptions<LightGbmBinaryTrainer.Options, float, BinaryPredictionTransformer<CalibratedModelParametersBase<LightGbmBinaryModelParameters, PlattCalibrator>>, CalibratedModelParametersBase<LightGbmBinaryModelParameters, PlattCalibrator>>(sweepParams, columnInfo);
            return mlContext.BinaryClassification.Trainers.LightGbm(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildLightGbmPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumnName, columnInfo.ExampleWeightColumnName, columnInfo.GroupIdColumnName);
        }
    }

    internal class LinearSvmBinaryExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildLinearSvmParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo, IDataView validationSet)
        {
            var options = TrainerExtensionUtil.CreateOptions<LinearSvmTrainer.Options>(sweepParams, columnInfo.LabelColumnName);
            return mlContext.BinaryClassification.Trainers.LinearSvm(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumnName);
        }
    }

    internal class SdcaLogisticRegressionBinaryExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildSdcaParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo, IDataView validationSet)
        {
            var options = TrainerExtensionUtil.CreateOptions<SdcaLogisticRegressionBinaryTrainer.Options>(sweepParams, columnInfo.LabelColumnName);
            return mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumnName);
        }
    }

    internal class LbfgsLogisticRegressionBinaryExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildLbfgsLogisticRegressionParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo, IDataView validationSet)
        {
            var options = TrainerExtensionUtil.CreateOptions<LbfgsLogisticRegressionBinaryTrainer.Options>(sweepParams, columnInfo.LabelColumnName);
            options.ExampleWeightColumnName = columnInfo.ExampleWeightColumnName;
            return mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumnName, columnInfo.ExampleWeightColumnName);
        }
    }

    internal class SgdCalibratedBinaryExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildSgdParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo, IDataView validationSet)
        {
            var options = TrainerExtensionUtil.CreateOptions<SgdCalibratedTrainer.Options>(sweepParams, columnInfo.LabelColumnName);
            options.ExampleWeightColumnName = columnInfo.ExampleWeightColumnName;
            return mlContext.BinaryClassification.Trainers.SgdCalibrated(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                 columnInfo.LabelColumnName, columnInfo.ExampleWeightColumnName);
        }
    }

    internal class SymbolicSgdLogisticRegressionBinaryExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildSymSgdLogisticRegressionParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo, IDataView validationSet)
        {
            var options = TrainerExtensionUtil.CreateOptions<SymbolicSgdLogisticRegressionBinaryTrainer.Options>(sweepParams, columnInfo.LabelColumnName);
            return mlContext.BinaryClassification.Trainers.SymbolicSgdLogisticRegression(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumnName);
        }
    }
}
