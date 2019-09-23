using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.Recommender;

namespace Microsoft.ML.AutoML
{
    using ITrainerEsitmator = ITrainerEstimator<ISingleFeaturePredictionTransformer<object>, object>;

    internal class MatrixFactorizationExtension : ITrainerExtension
    {
        public ITrainerEsitmator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            // TODO
            // MatrixFactorizationTrainer.Options should inheriat from ABC TrainerInputBaseWithGroupId
            var options = TrainerExtensionUtil.CreateOptions<MatrixFactorizationTrainer.Options>(sweepParams, columnInfo.LabelColumnName);
            options.MatrixColumnIndexColumnName = (string)AutoCatalog.ValuePairs[nameof(options.MatrixColumnIndexColumnName)];
            options.MatrixRowIndexColumnName = (string)AutoCatalog.ValuePairs[nameof(options.MatrixRowIndexColumnName)];
            return mlContext.Recommendation().Trainers.MatrixFactorization(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            throw new NotImplementedException();
        }

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildMatrixFactorizationParmas();
        }
    }
}
