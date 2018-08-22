using Microsoft.ML.Core.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Runtime.Training
{
    public interface ITrainerEstimator<out TTransformer, out TPredictor>: IEstimator<TTransformer>
        where TTransformer: IPredictionTransformer<TPredictor>
        where TPredictor: IPredictor
    {
        TrainerInfo Info { get; }

        PredictionKind PredictionKind { get; }
    }
}
