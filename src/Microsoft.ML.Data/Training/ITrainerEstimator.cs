// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;

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
