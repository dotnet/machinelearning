// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;

namespace Microsoft.ML.AutoML
{
    using ITrainerEstimator = ITrainerEstimator<ISingleFeaturePredictionTransformer<object>, object>;

    internal class RandomizedPcaExtension : ITrainerExtension
    {
        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return new List<SweepableParam>().AsEnumerable();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo)
        {
            return mlContext.AnomalyDetection.Trainers.RandomizedPca(rank: 1, ensureZeroMean: false);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumnName, columnInfo.ExampleWeightColumnName);
        }
    }
}