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
    internal class LightGbmRankingExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildLightGbmParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo, IDataView validationSet)
        {
            LightGbmRankingTrainer.Options options = TrainerExtensionUtil.CreateLightGbmOptions<LightGbmRankingTrainer.Options,
                float, RankingPredictionTransformer<LightGbmRankingModelParameters>, LightGbmRankingModelParameters>(sweepParams, columnInfo);
            options.RowGroupColumnName = columnInfo.GroupIdColumnName;
            return mlContext.Ranking.Trainers.LightGbm(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildLightGbmPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumnName, columnInfo.ExampleWeightColumnName, columnInfo.GroupIdColumnName);
        }
    }

    internal class FastTreeRankingExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildFastTreeParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo, IDataView validationSet)
        {
            var options = TrainerExtensionUtil.CreateOptions<FastTreeRankingTrainer.Options>(sweepParams, columnInfo.LabelColumnName);
            options.RowGroupColumnName = columnInfo.GroupIdColumnName;
            return mlContext.Ranking.Trainers.FastTree(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            var property = new Dictionary<string, object>();
            property.Add(nameof(FastTreeRankingTrainer.Options.RowGroupColumnName), columnInfo.GroupIdColumnName);
            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumnName, additionalProperties: property);
        }
    }
}
