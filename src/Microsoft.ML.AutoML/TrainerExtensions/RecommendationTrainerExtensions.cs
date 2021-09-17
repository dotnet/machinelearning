// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Trainers;

namespace Microsoft.ML.AutoML
{
    using ITrainerEsitmator = ITrainerEstimator<IPredictionTransformer<object>, object>;

    internal class MatrixFactorizationExtension : ITrainerExtension
    {
        public ITrainerEsitmator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo, IDataView validationSet)
        {
            var options = TrainerExtensionUtil.CreateOptions<MatrixFactorizationTrainer.Options>(sweepParams);
            options.LabelColumnName = columnInfo.LabelColumnName;
            options.MatrixColumnIndexColumnName = columnInfo.UserIdColumnName;
            options.MatrixRowIndexColumnName = columnInfo.ItemIdColumnName;
            options.Quiet = true;
            return mlContext.Recommendation().Trainers.MatrixFactorization(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            var property = new Dictionary<string, object>();
            property.Add(nameof(MatrixFactorizationTrainer.Options.MatrixColumnIndexColumnName), columnInfo.UserIdColumnName);
            property.Add(nameof(MatrixFactorizationTrainer.Options.MatrixRowIndexColumnName), columnInfo.ItemIdColumnName);
            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams, columnInfo.LabelColumnName, additionalProperties: property);
        }

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildMatrixFactorizationParams();
        }
    }
}
