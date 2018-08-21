using Microsoft.ML.Core.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Runtime.Training
{
    public interface ITrainerEstimator<TTransformer, TPredictor>: IEstimator<TTransformer>
        where TTransformer: IPredictionTransformer<TPredictor>
        where TPredictor: IPredictor
    {
        TrainerInfo TrainerInfo { get; }
    }
}
