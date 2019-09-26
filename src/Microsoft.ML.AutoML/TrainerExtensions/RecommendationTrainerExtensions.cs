// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Trainers;

namespace Microsoft.ML.AutoML
{
    using ITrainerEsitmator = ITrainerEstimator<ISingleFeaturePredictionTransformer<object>, object>;

    internal class MatrixFactorizationExtension : ITrainerExtension
    {
        public ITrainerEsitmator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            var options = TrainerExtensionUtil.CreateOptions<MatrixFactorizationTrainer.Options>(sweepParams);
            options.LabelColumnName = columnInfo.LabelColumnName;
            options.MatrixColumnIndexColumnName = (string)AutoCatalog.ValuePairs[nameof(options.MatrixColumnIndexColumnName)];
            options.MatrixRowIndexColumnName = (string)AutoCatalog.ValuePairs[nameof(options.MatrixRowIndexColumnName)];
            return mlContext.Recommendation().Trainers.MatrixFactorization(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            var property = new Dictionary<string, object>();
            property.Add(nameof(MatrixFactorizationTrainer.Options.MatrixColumnIndexColumnName), AutoCatalog.ValuePairs[nameof(MatrixFactorizationTrainer.Options.MatrixColumnIndexColumnName)]);
            property.Add(nameof(MatrixFactorizationTrainer.Options.MatrixRowIndexColumnName), AutoCatalog.ValuePairs[nameof(MatrixFactorizationTrainer.Options.MatrixRowIndexColumnName)]);
            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams, columnInfo.LabelColumnName, additionalProperties:property);
        }

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildMatrixFactorizationParmas();
        }
    }
}
